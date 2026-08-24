-- RAGent :: system of record
-- Design rule: every generated sentence must be traceable back to its exact
-- location in the source. That chain is
--     documents -> pages -> blocks -> chunk_blocks -> chunks -> citations
-- and nothing in the pipeline may break a link in it.
--
-- A block locates itself one of two ways depending on the source format:
--   paged (pdf, images, converted office docs) -> page_no + normalised bbox
--   flow  (markdown, csv, plain text)          -> character offsets
-- The block_is_locatable constraint enforces that it does one or the other.

CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS unaccent;

-- ---------------------------------------------------------------- enums

CREATE TYPE doc_status   AS ENUM ('pending','processing','ready','failed','quarantined');
CREATE TYPE stage_status AS ENUM ('pending','running','succeeded','failed','skipped');
CREATE TYPE block_kind   AS ENUM ('title','heading','paragraph','list','table','figure','caption','footnote','header','footer','page_number');
CREATE TYPE text_origin  AS ENUM ('native','ocr','vlm');
CREATE TYPE format_family   AS ENUM ('pdf','image','office','web','flow');
-- How a citation resolves back to its source. Paged documents highlight a
-- pixel region; flow documents have no geometry, so they carry character
-- offsets instead. Two honest modes beat one that fabricates bounding
-- boxes for a Markdown file.
CREATE TYPE provenance_mode AS ENUM ('paged','flow');

-- ---------------------------------------------------------------- documents

CREATE TABLE documents (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id       TEXT        NOT NULL DEFAULT 'default',
    -- content hash is the dedupe key: re-uploading the same filing is a no-op
    sha256          CHAR(64)    NOT NULL,
    source_uri      TEXT        NOT NULL,          -- s3://raw/<sha256>.<ext>
    original_name   TEXT        NOT NULL,
    mime_type       TEXT        NOT NULL,
    byte_size       BIGINT      NOT NULL,
    page_count      INT,
    status          doc_status  NOT NULL DEFAULT 'pending',

    -- Set by the detect stage from magic bytes, never from the extension.
    format_family   format_family,
    provenance      provenance_mode,
    -- Office and HTML render to PDF before parsing; this is that artefact.
    converted_uri   TEXT,

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
    -- detect|convert|parse|parse_flow|ocr|tables|figures|chunk|contextualize|embed
    stage        TEXT NOT NULL,
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
    -- NULL for flow documents (markdown, csv, plain text): they have no pages.
    page_id        UUID REFERENCES pages(id) ON DELETE CASCADE,
    page_no        INT,
    reading_order  INT  NOT NULL,                  -- resolved across columns
    kind           block_kind  NOT NULL,
    origin         text_origin NOT NULL,
    confidence     REAL,

    -- Paged provenance: normalised 0..1 bbox on the rendered page.
    x0 REAL, y0 REAL, x1 REAL, y1 REAL,
    -- Flow provenance: character offsets into the extracted text.
    char_start INT, char_end INT,

    text           TEXT NOT NULL,
    -- heading trail at this point in the doc, e.g. {"Part II","Item 7","Liquidity"}
    section_path   TEXT[] NOT NULL DEFAULT '{}',

    CONSTRAINT bbox_normalised CHECK (
        x0 IS NULL OR (x0 >= 0 AND y0 >= 0 AND x1 <= 1 AND y1 <= 1 AND x1 > x0 AND y1 > y0)
    ),
    CONSTRAINT char_range_valid CHECK (
        char_start IS NULL OR (char_start >= 0 AND char_end > char_start)
    ),
    -- Every block must be locatable one way or the other. This is the invariant
    -- that keeps citations resolvable across every supported format.
    CONSTRAINT block_is_locatable CHECK (
        (x0 IS NOT NULL AND page_no IS NOT NULL) OR char_start IS NOT NULL
    )
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
    -- Paged documents fill the page range, flow documents the char range.
    page_from      INT,
    page_to        INT,
    char_start     INT,
    char_end       INT,
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
    -- Paged: page plus the union of the cited chunk's block bboxes,
    -- precomputed so the viewer never re-derives a highlight.
    page_no     INT,
    bboxes      JSONB,
    -- Flow: the character range the viewer highlights instead.
    char_start  INT,
    char_end    INT,
    quote       TEXT,
    -- verifier score: did this claim actually follow from this chunk?
    grounding   REAL,
    UNIQUE (message_id, marker),
    CONSTRAINT citation_is_locatable CHECK (
        (page_no IS NOT NULL AND bboxes IS NOT NULL) OR char_start IS NOT NULL
    )
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
