"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { Doc } from "@/lib/api";
import { documentEvents, getDocument, listDocuments, upload } from "@/lib/api";

const ACCEPT =
  ".pdf,.png,.jpg,.jpeg,.tiff,.tif,.gif,.bmp,.webp," +
  ".docx,.xlsx,.pptx,.doc,.xls,.ppt,.odt,.ods,.odp,.rtf," +
  ".html,.htm,.md,.txt,.csv,.tsv,.json,.xml";

export function Sidebar({
  selected,
  onSelect,
}: {
  selected: string[];
  onSelect: (ids: string[]) => void;
}) {
  const [docs, setDocs] = useState<Doc[]>([]);
  const [over, setOver] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const refresh = useCallback(async () => {
    try {
      const { documents } = await listDocuments();
      setDocs(documents);
      return documents;
    } catch (e) {
      setError(String(e));
      return [];
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  /** Subscribe to every in-flight document so the stage strip updates live.
   *  Finished documents need no stream, so the set shrinks as ingest completes. */
  useEffect(() => {
    const active = docs.filter((d) => d.status === "pending" || d.status === "processing");
    if (!active.length) return;

    const closers = active.map((doc) =>
      documentEvents(doc.id, {
        snapshot: (data: Doc) => patch(doc.id, data),
        progress: async (event: { type: string }) => {
          if (event.type === "ready" || event.type === "quarantined") {
            await refresh();
          } else {
            patch(doc.id, await getDocument(doc.id).catch(() => null));
          }
        },
      }),
    );
    return () => closers.forEach((close) => close());

    function patch(id: string, next: Doc | null) {
      if (!next) return;
      setDocs((prev) => prev.map((d) => (d.id === id ? { ...d, ...next } : d)));
    }
  }, [docs.map((d) => `${d.id}:${d.status}`).join(","), refresh]);

  const send = useCallback(
    async (files: FileList | null) => {
      if (!files?.length) return;
      setBusy(true);
      setError(null);
      try {
        for (const file of Array.from(files)) await upload(file);
        await refresh();
      } catch (e) {
        setError(String(e));
      } finally {
        setBusy(false);
      }
    },
    [refresh],
  );

  const toggle = (id: string) =>
    onSelect(
      selected.includes(id) ? selected.filter((s) => s !== id) : [...selected, id],
    );

  return (
    <aside className="sidebar">
      <div className="pane-head">
        <div className="brand">
          RAGent<span>corpus</span>
        </div>
        {selected.length > 0 && (
          <button className="small" onClick={() => onSelect([])}>
            Clear ({selected.length})
          </button>
        )}
      </div>

      <div
        className={`drop${over ? " over" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setOver(false);
          send(e.dataTransfer.files);
        }}
        onClick={() => inputRef.current?.click()}
      >
        {busy ? "Uploading…" : "Drop files or click to upload"}
        <div className="small" style={{ marginTop: 4, opacity: 0.7 }}>
          PDF · images · Office · HTML · text
        </div>
        <input
          ref={inputRef}
          type="file"
          multiple
          accept={ACCEPT}
          hidden
          onChange={(e) => {
            send(e.target.files);
            e.target.value = "";
          }}
        />
      </div>

      {error && <div className="banner">{error}</div>}

      <div className="pane-body">
        {docs.length === 0 && (
          <div className="muted small" style={{ textAlign: "center", padding: "24px 0" }}>
            No documents yet.
          </div>
        )}
        {docs.map((doc) => (
          <div
            key={doc.id}
            className={`doc${selected.includes(doc.id) ? " selected" : ""}`}
            onClick={() => toggle(doc.id)}
          >
            <div className="doc-name">{doc.name}</div>
            <div className="doc-meta small">
              <span className={`pill ${doc.status}`}>{doc.status}</span>
              {doc.format_family && <span className="pill">{doc.format_family}</span>}
              {doc.provenance && <span className="pill">{doc.provenance}</span>}
              {doc.page_count ? <span className="muted">{doc.page_count}p</span> : null}
              {doc.chunk_count ? (
                <span className="muted">{doc.chunk_count} chunks</span>
              ) : null}
            </div>
            {doc.stages && doc.status !== "ready" && (
              <div className="stages">
                {doc.stages.map((stage) => (
                  <span
                    key={stage.name}
                    className={`stage ${stage.status}`}
                    title={stage.error ?? JSON.stringify(stage.metrics)}
                  >
                    {stage.name}
                  </span>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </aside>
  );
}
