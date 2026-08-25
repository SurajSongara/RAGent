"use client";

import { useEffect, useRef, useState } from "react";
import type { Citation } from "@/lib/api";
import { getDocumentFile, getDocumentText } from "@/lib/api";

/**
 * The payoff of the whole provenance chain: click a citation, see exactly where
 * the claim came from.
 *
 * Two modes, because the two provenance models genuinely differ:
 *   paged — render the PDF page with pdf.js and draw the bbox on top.
 *   flow  — show the source text with the character range highlighted.
 *
 * Rendering happens in the browser rather than server-side. Bboxes are stored
 * normalised 0..1, so the overlay is just a percentage of whatever size the
 * canvas ended up — no scale negotiation with the backend, and it stays correct
 * at any zoom.
 */
export function CitationViewer({
  citation,
  onClose,
}: {
  citation: Citation | null;
  onClose: () => void;
}) {
  if (!citation) return null;

  return (
    <div className="viewer">
      <div className="pane-head">
        <div style={{ minWidth: 0 }}>
          <div style={{ fontWeight: 600, overflow: "hidden", textOverflow: "ellipsis" }}>
            [{citation.marker}] {citation.document_name}
          </div>
          {citation.section_path.length > 0 && (
            <div className="small muted">{citation.section_path.join(" › ")}</div>
          )}
        </div>
        <button onClick={onClose}>Close</button>
      </div>
      <div className="pane-body">
        {citation.provenance === "flow" ? (
          <FlowHighlight citation={citation} />
        ) : (
          <PagedHighlight citation={citation} />
        )}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ flow */

function FlowHighlight({ citation }: { citation: Citation }) {
  const [text, setText] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const markRef = useRef<HTMLElement>(null);

  useEffect(() => {
    let cancelled = false;
    setText(null);
    setError(null);
    getDocumentText(citation.document_id)
      .then((r) => !cancelled && setText(r.text))
      .catch((e) => !cancelled && setError(String(e)));
    return () => {
      cancelled = true;
    };
  }, [citation.document_id]);

  useEffect(() => {
    markRef.current?.scrollIntoView({ block: "center", behavior: "smooth" });
  }, [text, citation.chunk_id]);

  if (error) return <div className="muted small">Could not load source: {error}</div>;
  if (text === null) return <div className="muted small">Loading source…</div>;

  const start = citation.char_start ?? 0;
  const end = citation.char_end ?? 0;
  if (end <= start) return <div className="flow-text">{text}</div>;

  // A window around the span keeps very large files from locking the browser
  // up, while still showing enough context to read around the highlight.
  const PAD = 2000;
  const from = Math.max(0, start - PAD);
  const to = Math.min(text.length, end + PAD);

  return (
    <>
      <div className="small muted" style={{ marginBottom: 10 }}>
        characters {start.toLocaleString()}–{end.toLocaleString()}
        {from > 0 && " · trimmed for display"}
      </div>
      <div className="flow-text">
        {from > 0 && <span className="muted">…</span>}
        {text.slice(from, start)}
        <mark ref={markRef}>{text.slice(start, end)}</mark>
        {text.slice(end, to)}
        {to < text.length && <span className="muted">…</span>}
      </div>
    </>
  );
}

/* ----------------------------------------------------------------- paged */

function PagedHighlight({ citation }: { citation: Citation }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");
  const [message, setMessage] = useState("");

  useEffect(() => {
    let cancelled = false;
    const container = containerRef.current;
    if (!container) return;

    container.innerHTML = "";
    setStatus("loading");

    (async () => {
      const pdfjs = await import("pdfjs-dist");
      // Bundled worker rather than a CDN: the app has to work offline.
      pdfjs.GlobalWorkerOptions.workerSrc = new URL(
        "pdfjs-dist/build/pdf.worker.min.mjs",
        import.meta.url,
      ).toString();

      const { url } = await getDocumentFile(citation.document_id);
      const pdf = await pdfjs.getDocument({ url }).promise;
      if (cancelled) return;

      // One canvas per page the chunk touches — a chunk can straddle a break.
      const pageNos = [...new Set(citation.regions.map((r) => r.page_no))].sort(
        (a, b) => a - b,
      );

      for (const pageNo of pageNos) {
        if (cancelled || pageNo < 1 || pageNo > pdf.numPages) continue;

        const page = await pdf.getPage(pageNo);
        const viewport = page.getViewport({ scale: 1.6 });

        const wrap = document.createElement("div");
        wrap.className = "page-wrap";
        wrap.style.width = `${viewport.width}px`;
        wrap.style.height = `${viewport.height}px`;

        const canvas = document.createElement("canvas");
        canvas.width = viewport.width;
        canvas.height = viewport.height;
        wrap.appendChild(canvas);

        const label = document.createElement("div");
        label.className = "small muted";
        label.style.margin = "0 0 6px";
        label.textContent = `page ${pageNo}`;
        container.appendChild(label);
        container.appendChild(wrap);

        await page.render({ canvasContext: canvas.getContext("2d")!, viewport }).promise;
        if (cancelled) return;

        // Normalised coordinates mean the overlay is pure percentages — this is
        // exactly why they are stored 0..1 rather than in points.
        for (const region of citation.regions.filter((r) => r.page_no === pageNo)) {
          const box = document.createElement("div");
          box.className = "overlay";
          box.style.left = `${region.x0 * 100}%`;
          box.style.top = `${region.y0 * 100}%`;
          box.style.width = `${(region.x1 - region.x0) * 100}%`;
          box.style.height = `${(region.y1 - region.y0) * 100}%`;
          wrap.appendChild(box);
        }
      }

      if (!cancelled) {
        setStatus("ready");
        container.querySelector(".overlay")?.scrollIntoView({
          block: "center",
          behavior: "smooth",
        });
      }
    })().catch((error) => {
      if (cancelled) return;
      setMessage(String(error));
      setStatus("error");
    });

    return () => {
      cancelled = true;
    };
  }, [citation.chunk_id, citation.document_id]);

  return (
    <>
      {status === "loading" && <div className="muted small">Rendering page…</div>}
      {status === "error" && (
        // Falling back to the passage text keeps the citation useful even when
        // the document cannot be rendered.
        <div>
          <div className="banner">Could not render the page: {message}</div>
          <div style={{ marginTop: 12, whiteSpace: "pre-wrap" }}>{citation.text}</div>
        </div>
      )}
      <div ref={containerRef} />
      {status === "ready" && citation.regions.length === 0 && (
        <div className="banner">
          This chunk has no stored region, so there is nothing to highlight.
        </div>
      )}
    </>
  );
}
