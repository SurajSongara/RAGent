"""Seed the demo corpus.

Uploads through the real API so the seed exercises the same path a user does —
no back door that works when the pipeline does not.

Two sources. Local fixtures are generated in-process and always work, covering
one document per format family so the routing is visible immediately. EDGAR
filings are fetched over the network and skipped cleanly if it is unavailable,
because `make seed` failing on a flaky connection would make the whole demo look
broken.

    python -m scripts.seed_edgar              # fixtures + EDGAR
    python -m scripts.seed_edgar --local      # fixtures only, no network
"""

from __future__ import annotations

import argparse
import asyncio
import io
import sys
import zipfile

import httpx

API = "http://localhost:8000"

# EDGAR requires a descriptive User-Agent naming a real contact; requests
# without one are throttled or refused outright.
EDGAR_UA = "RAGent demo seeder (github.com/SurajSongara/RAGent)"

# Primary filing documents. EDGAR serves these as HTML, which exercises the
# WEB -> convert -> PDF branch of the DAG rather than the easy PDF path.
EDGAR_FILINGS: list[tuple[str, str]] = [
    (
        "apple-10q-2025-q3.htm",
        "https://www.sec.gov/Archives/edgar/data/320193/000032019325000073/aapl-20250628.htm",
    ),
    (
        "microsoft-10k-2025.htm",
        "https://www.sec.gov/Archives/edgar/data/789019/000095017025100235/msft-20250630.htm",
    ),
]


MARKDOWN_FIXTURE = """# RAGent Demo Notes

## Overview

RAGent ingests documents of several kinds and answers questions about them with
citations that resolve back to the exact spot in the source.

## Retrieval

Retrieval is hybrid. A dense retriever runs against Qdrant with payload filters,
and a lexical retriever runs against Postgres. Their results are fused with
Reciprocal Rank Fusion, which combines them on rank rather than on score.

Fusing on rank matters because cosine similarity and ts_rank_cd are not on the
same scale, and normalising between them needs per-query calibration that drifts
whenever the corpus or the embedding model changes.

## Provenance

Paged documents carry a page number and a normalised bounding box. Flow
documents such as this one have no geometry at all, so they carry character
offsets instead, and the viewer highlights a text range.

### Why two modes

The alternative was laying flow text onto synthetic pages so that everything had
a bounding box. That would mean fabricating coordinates that correspond to
nothing real.

## Selective OCR

A born-digital PDF already has a perfect text layer. Re-running OCR over it
costs minutes per document and makes accuracy worse, so each page's text layer
is scored and only the pages that fall below the confidence threshold are sent
for recognition.
"""

CSV_FIXTURE = """segment,fiscal_year,revenue_usd_m,gross_margin_pct,notes
Services,2024,85200,71.2,record quarter
Services,2025,96400,73.8,driven by subscriptions
Hardware,2024,214100,36.5,unit volumes declined
Hardware,2025,208900,37.1,mix shifted to premium
Licensing,2024,12400,88.0,
Licensing,2025,14100,89.2,renewal rate improved
Other,2024,3100,22.4,includes one-time items
Other,2025,2700,19.8,
"""

TEXT_FIXTURE = """QUARTERLY OPERATING SUMMARY

Total revenue for the period increased 12% year over year, driven primarily by
higher services revenue and partially offset by a decline in hardware unit
volumes across certain geographic segments.

Gross margin percentage improved by 180 basis points compared with the prior
year period, reflecting a more favourable mix of higher margin services revenue
and continued operating leverage in the manufacturing organisation.

Operating expenses were (1,234) million for the quarter, which includes a
one-time restructuring charge of 210 million recorded in the second month of the
period. Excluding that charge, operating expenses would have been broadly flat.

The litigation described in Item 3 remains pending before the district court.
Management believes the claims are without merit and intends to defend them
vigorously.
"""


def docx_fixture() -> bytes:
    """A minimal but valid .docx, built in-process.

    Hand-rolling the OOXML rather than depending on python-docx keeps the seeder
    dependency-free, and the point is only to prove the Office route works.
    """
    body = "".join(
        f"<w:p><w:r><w:t xml:space='preserve'>{line}</w:t></w:r></w:p>"
        for line in [
            "Board Meeting Minutes",
            "",
            "The board reviewed the quarterly results. Revenue grew twelve percent "
            "year over year, ahead of the guidance issued in the prior quarter.",
            "",
            "The audit committee reported no material weaknesses in internal "
            "controls over financial reporting.",
            "",
            "The board approved a capital expenditure budget of 450 million for "
            "the coming fiscal year, allocated primarily to data centre capacity.",
        ]
    )
    document = (
        "<?xml version='1.0' encoding='UTF-8' standalone='yes'?>"
        "<w:document xmlns:w='http://schemas.openxmlformats.org/wordprocessingml/2006/main'>"
        f"<w:body>{body}</w:body></w:document>"
    )
    content_types = (
        "<?xml version='1.0' encoding='UTF-8' standalone='yes'?>"
        "<Types xmlns='http://schemas.openxmlformats.org/package/2006/content-types'>"
        "<Default Extension='rels' ContentType='application/vnd.openxmlformats-package.relationships+xml'/>"
        "<Default Extension='xml' ContentType='application/xml'/>"
        "<Override PartName='/word/document.xml' ContentType='application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml'/>"
        "</Types>"
    )
    rels = (
        "<?xml version='1.0' encoding='UTF-8' standalone='yes'?>"
        "<Relationships xmlns='http://schemas.openxmlformats.org/package/2006/relationships'>"
        "<Relationship Id='rId1' Type='http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument' Target='word/document.xml'/>"
        "</Relationships>"
    )

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("word/document.xml", document)
    return buffer.getvalue()


FIXTURES: list[tuple[str, bytes, str]] = [
    ("ragent-demo-notes.md", MARKDOWN_FIXTURE.encode(), "text/markdown"),
    ("segment-results.csv", CSV_FIXTURE.encode(), "text/csv"),
    ("operating-summary.txt", TEXT_FIXTURE.encode(), "text/plain"),
    (
        "board-minutes.docx",
        docx_fixture(),
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ),
]


async def push(client: httpx.AsyncClient, name: str, data: bytes, mime: str) -> None:
    try:
        response = await client.post(
            f"{API}/documents",
            files={"file": (name, data, mime)},
            timeout=120,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        print(f"  ✗ {name}: {exc}")
        return

    payload = response.json()
    marker = "•" if payload.get("status") == "duplicate" else "✓"
    print(f"  {marker} {name} ({payload.get('status')}) {payload.get('document_id', '')}")


async def fetch_edgar(client: httpx.AsyncClient) -> list[tuple[str, bytes, str]]:
    out: list[tuple[str, bytes, str]] = []
    for name, url in EDGAR_FILINGS:
        try:
            response = await client.get(
                url, headers={"User-Agent": EDGAR_UA}, timeout=60, follow_redirects=True
            )
            response.raise_for_status()
            out.append((name, response.content, "text/html"))
            print(f"  ↓ {name} ({len(response.content) // 1024} KB)")
        except httpx.HTTPError as exc:
            print(f"  ! skipping {name}: {exc}")
    return out


async def main(local_only: bool) -> int:
    async with httpx.AsyncClient() as client:
        try:
            health = await client.get(f"{API}/health", timeout=10)
            health.raise_for_status()
        except httpx.HTTPError:
            print(f"API is not reachable at {API}. Is the stack up? Try `make up`.")
            return 1

        info = health.json()
        print(
            f"API ready · embeddings={info.get('embedding_backend')} "
            f"· llm={'yes' if info.get('llm_configured') else 'no key'}\n"
        )

        print("Local fixtures (one per format family):")
        for name, data, mime in FIXTURES:
            await push(client, name, data, mime)

        if not local_only:
            print("\nEDGAR filings:")
            for name, data, mime in await fetch_edgar(client):
                await push(client, name, data, mime)

    print("\nSeeded. Watch ingest at http://localhost:3000")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="scripts.seed_edgar")
    parser.add_argument(
        "--local", action="store_true", help="skip EDGAR, use bundled fixtures only"
    )
    sys.exit(asyncio.run(main(parser.parse_args().local)))
