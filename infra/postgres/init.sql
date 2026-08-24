-- RAGent :: system of record
-- Design rule: every generated sentence must be traceable to a pixel region of a
-- source page. That chain is documents -> pages -> blocks(bbox) -> chunk_blocks -> chunks.
-- Nothing in the pipeline is allowed to break a link in it.

CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS unaccent;

-- ---------------------------------------------------------------- enums

CREATE TYPE doc_status   AS ENUM ('pending','processing','ready','failed','quarantined');
CREATE TYPE stage_status AS ENUM ('pending','running','succeeded','failed','skipped');
CREATE TYPE block_kind   AS ENUM ('title','heading','paragraph','list','table','figure','caption','footnote','header','footer','page_number');
CREATE TYPE text_origin  AS ENUM ('native','ocr','vlm');

-- ---------------------------------------------------------------- documents

CREATE TABLE documents (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id       TEXT        NOT NULL DEFAULT 'default',
    -- content hash is the dedupe key: re-uploading the same filing is a no-op
    sha256          CHAR(64)    NOT NULL,
    source_uri      TEXT        NOT NULL,          -- s3://raw/<sha256>.pdf
    original_name   TEXT        NOT NULL,
    mime_type       TEXT        NOT NULL,
    byte_size       BIGINT      NOT NULL,
    page_count      INT,
    status          doc_status  NOT NULL DEFAULT 'pending',

    -- EDGAR / filing metadata. Doubles as the Qdrant payload filter set, so
    -- "compare FY24 vs FY25 for CIK X" is a filtered vector search, not a rerank hack.
    cik             TEXT,
    ticker          TEXT,
    company_name    TEXT,
    form_type       TEXT,                          -- 10-K, 10-Q, 8-K, DEF 14A, PRES
    fiscal_year     INT,
    fiscal_period   TEXT,                          -- FY, Q1..Q4
    filed_at        DATE,

    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    UNIQUE (tenant_id, sha256)
);

CREATE INDEX documents_filter_idx ON documents (tenant_id, cik, fiscal_year, form_type);
CREATE INDEX documents_status_idx ON documents (status) WHERE status <> 'ready';

-- ---------------------------------------------------------------- pipeline state
-- One row per (document, pipeline_version). Lets a doc resume mid-DAG after a
-- worker crash instead of re-OCRing 200 pages, and makes reprocessing on a new
-- pipeline version an explicit, comparable run rather than a destructive update.

CREATE TABLE ingest_runs (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id      UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    pipeline_version TEXT NOT NULL,
    status           stage_status NOT NULL DEFAULT 'pending',
    started_at       TIMESTAMPTZ,
    finished_at      TIMESTAMPTZ,
    error            TEXT,
    UNIQUE (document_id, pipeline_version)
);

CREATE TABLE ingest_stages (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id       UUID NOT NULL REFERENCES ingest_runs(id) ON DELETE CASCADE,
    stage        TEXT NOT NULL,      -- classify|parse|ocr|tables|figures|chunk|contextualize|embed
    status       stage_status NOT NULL DEFAULT 'pending',
    attempt      INT NOT NULL DEFAULT 0,
    started_at   TIMESTAMPTZ,
    finished_at  TIMESTAMPTZ,
    error        TEXT,
    -- per-stage timings, token spend, page counts -> feeds the cost dashboard
    metrics      JSONB NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE (run_id, stage)
);

CREATE INDEX ingest_stages_live_idx ON ingest_stages (status, stage) WHERE status IN ('pending','running');

-- ---------------------------------------------------------------- pages and blocks

CREATE TABLE pages (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id      UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_no          INT  NOT NULL,                -- 1-indexed
    width_pt         REAL NOT NULL,
    height_pt        REAL NOT NULL,
    rotation         INT  NOT NULL DEFAULT 0,
    render_uri       TEXT,                         -- s3://pages/<doc>/<page>.webp
    -- mean confidence of the embedded text layer. Drives SELECTIVE ocr: we only
    -- rasterise and OCR the regions a born-digital PDF got wrong, never the whole file.
    text_confidence  REAL,
    needs_ocr        BOOLEAN NOT NULL DEFAULT false,
    UNIQUE (document_id, page_no)
);

-- The provenance atom. bbox is normalised 0..1 against page dims so the web
-- viewer can highlight at any zoom without knowing the render scale.
CREATE TABLE blocks (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id    UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_id        UUID NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
    page_no        INT  NOT NULL,
    reading_order  INT  NOT NULL,                  -- resolved across columns
    kind           block_kind  NOT NULL,
    origin         text_origin NOT NULL,
    confidence     REAL,
    x0 REAL NOT NULL, y0 REAL NOT NULL, x1 REAL NOT NULL, y1 REAL NOT NULL,
    text           TEXT NOT NULL,
    -- heading trail at this point in the doc, e.g. {"Part II","Item 7","Liquidity"}
    section_path   TEXT[] NOT NULL DEFAULT '{}',
    CONSTRAINT bbox_normalised CHECK (x0 >= 0 AND y0 >= 0 AND x1 <= 1 AND y1 <= 1 AND x1 > x0 AND y1 > y0)
);

CREATE INDEX blocks_doc_order_idx ON blocks (document_id, page_no, reading_order);

-- ---------------------------------------------------------------- tables
-- Financial tables are kept as CELLS, not flattened to markdown. A question like
-- "what was FY25 gross margin" then resolves against typed numeric values instead
-- of hoping the LLM parses a mangled pipe-table back out of a chunk.

CREATE TABLE doc_tables (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id  UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    block_id     UUID NOT NULL REFERENCES blocks(id) ON DELETE CASCADE,
    page_no      INT  NOT NULL,
    caption      TEXT,
    n_rows       INT  NOT NULL,
    n_cols       INT  NOT NULL,
    units        TEXT,                             -- "USD thousands", "%"
    extractor    TEXT NOT NULL
);

CREATE TABLE table_cells (
    id            BIGSERIAL PRIMARY KEY,
    table_id      UUID NOT NULL REFERENCES doc_tables(id) ON DELETE CASCADE,
    row_idx       INT NOT NULL,
    col_idx       INT NOT NULL,
    row_span      INT NOT NULL DEFAULT 1,
    col_span      INT NOT NULL DEFAULT 1,
    is_header     BOOLEAN NOT NULL DEFAULT false,
    text          TEXT NOT NULL,
    -- parsed once at ingest: "(1,234)" -> -1234.0, "12.3%" -> 12.3
    numeric_value NUMERIC,
    UNIQUE (table_id, row_idx, col_idx)
);

CREATE INDEX table_cells_numeric_idx ON table_cells (table_id) WHERE numeric_value IS NOT NULL;

-- ---------------------------------------------------------------- chunks
-- `strategy` is what makes the Phase 2 bake-off possible: four chunkings of the
-- same corpus coexist and are evaluated head to head against one golden set.

CREATE TABLE chunks (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id    UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    strategy       TEXT NOT NULL,        -- fixed|recursive|layout|semantic
    seq            INT  NOT NULL,
    text           TEXT NOT NULL,
    -- one-line "where this sits in the document" preamble written at ingest.
    -- Embedded WITH the text (contextual retrieval); shown to the user WITHOUT it.
    context_prefix TEXT,
    token_count    INT  NOT NULL,
    section_path   TEXT[] NOT NULL DEFAULT '{}',
    page_from      INT  NOT NULL,
    page_to        INT  NOT NULL,
    tsv            tsvector GENERATED ALWAYS AS (
                       to_tsvector('english', coalesce(context_prefix,'') || ' ' || text)
                   ) STORED,
    UNIQUE (document_id, strategy, seq)
);

CREATE INDEX chunks_tsv_idx      ON chunks USING GIN (tsv);
CREATE INDEX chunks_strategy_idx ON chunks (strategy, document_id);

-- The join that keeps citations honest. Losing it would let a chunk exist without
-- knowing which pixels it came from, which is the failure mode this schema exists
-- to prevent.
CREATE TABLE chunk_blocks (
    chunk_id UUID NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    block_id UUID NOT NULL REFERENCES blocks(id) ON DELETE CASCADE,
    ordinal  INT  NOT NULL,
    PRIMARY KEY (chunk_id, block_id)
);

-- ---------------------------------------------------------------- conversations

CREATE TABLE conversations (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id  TEXT NOT NULL DEFAULT 'default',
    title      TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE messages (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL,      -- user|assistant
    content         TEXT NOT NULL,
    -- which model actually served this turn, and what it cost. The router's
    -- decisions stay auditable after the fact rather than asserted in a README.
    model           TEXT,
    input_tokens    INT,
    output_tokens   INT,
    cost_usd        NUMERIC(12,6),
    latency_ms      INT,
    trace_id        TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX messages_conv_idx ON messages (conversation_id, created_at);

-- Resolved at generation time, so the UI never has to re-derive a highlight.
CREATE TABLE citations (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    message_id  UUID NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    marker      INT  NOT NULL,          -- the [1] in the answer text
    chunk_id    UUID NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_no     INT  NOT NULL,
    -- union of the cited chunk block bboxes, precomputed for the viewer
    bboxes      JSONB NOT NULL,
    quote       TEXT,
    -- verifier score: did this claim actually follow from this chunk?
    grounding   REAL,
    UNIQUE (message_id, marker)
);

-- ---------------------------------------------------------------- triggers

CREATE OR REPLACE FUNCTION touch_updated_at() RETURNS trigger AS $fn$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$fn$ LANGUAGE plpgsql;

CREATE TRIGGER documents_touch BEFORE UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION touch_updated_at();
