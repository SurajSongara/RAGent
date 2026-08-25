"use client";

import { Fragment, useEffect, useRef, useState } from "react";
import type { Citation, Passage } from "@/lib/api";
import { askStream, health } from "@/lib/api";

type Turn = {
  question: string;
  answer: string;
  passages: Passage[];
  citations: Citation[];
  usage: { model: string; cost_usd: number; latency_ms: number } | null;
  done: boolean;
};

export function Ask({
  documentIds,
  onCite,
}: {
  documentIds: string[];
  onCite: (citation: Citation) => void;
}) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [llmReady, setLlmReady] = useState<boolean | null>(null);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    health()
      .then((h) => setLlmReady(h.llm_configured))
      .catch(() => setLlmReady(null));
  }, []);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns]);

  function submit() {
    const question = input.trim();
    if (!question || busy) return;

    setInput("");
    setBusy(true);
    const index = turns.length;
    setTurns((prev) => [
      ...prev,
      { question, answer: "", passages: [], citations: [], usage: null, done: false },
    ]);

    const update = (patch: Partial<Turn>) =>
      setTurns((prev) =>
        prev.map((turn, i) => (i === index ? { ...turn, ...patch } : turn)),
      );

    askStream(question, documentIds.length ? documentIds : null, {
      // Evidence arrives before the answer, so the wait shows real progress
      // instead of a spinner.
      passages: (data: Passage[]) => update({ passages: data }),
      delta: (data: { text: string }) =>
        setTurns((prev) =>
          prev.map((turn, i) =>
            i === index ? { ...turn, answer: turn.answer + data.text } : turn,
          ),
        ),
      citations: (data: Citation[]) => update({ citations: data }),
      usage: (data: Turn["usage"]) => update({ usage: data }),
      done: () => {
        update({ done: true });
        setBusy(false);
      },
    });
  }

  return (
    <main className="main">
      <div className="pane-head">
        <div className="brand">
          Ask
          <span>
            {documentIds.length
              ? `${documentIds.length} document${documentIds.length > 1 ? "s" : ""} selected`
              : "whole corpus"}
          </span>
        </div>
        {llmReady === false && (
          <span className="small muted">no API key · extractive mode</span>
        )}
      </div>

      <div className="messages">
        {turns.length === 0 && (
          <div className="empty">
            <p style={{ fontSize: 15, color: "var(--text)" }}>
              Ask a question about your documents.
            </p>
            <p className="small">
              Every claim is cited. Click a citation to see the exact region of the
              source it came from — a highlighted box on the page, or the precise
              character range for documents that have no pages.
            </p>
          </div>
        )}

        {turns.map((turn, i) => (
          <div className="turn" key={i}>
            <div className="turn-q">{turn.question}</div>

            {!turn.answer && !turn.done && (
              <div className="muted small">
                {turn.passages.length
                  ? `Found ${turn.passages.length} passages. Writing…`
                  : "Retrieving…"}
              </div>
            )}

            <div className="answer">
              <WithCitations text={turn.answer} citations={turn.citations} onCite={onCite} />
            </div>

            {turn.done && turn.passages.length > 0 && (
              <div className="evidence">
                <div className="small muted" style={{ marginBottom: 7 }}>
                  Retrieved evidence
                  {turn.usage &&
                    ` · ${turn.usage.model} · $${turn.usage.cost_usd.toFixed(4)} · ${turn.usage.latency_ms}ms`}
                </div>
                {turn.passages.map((passage, n) => (
                  <div
                    className="passage"
                    key={passage.chunk_id}
                    onClick={() => onCite({ ...passage, marker: n + 1 })}
                  >
                    <div className="passage-head">
                      <span className="cite">{n + 1}</span>
                      <strong className="small">{passage.document_name}</strong>
                      {passage.section_path.length > 0 && (
                        <span className="small muted">
                          {passage.section_path.join(" › ")}
                        </span>
                      )}
                      {/* Which retriever found it is the interesting part: hits
                          both retrievers agreed on are the strong ones. */}
                      <span className="mono muted">
                        {Object.keys(passage.retrievers).join("+") || "—"}
                      </span>
                    </div>
                    <div className="muted">
                      {passage.text.length > 260
                        ? `${passage.text.slice(0, 260)}…`
                        : passage.text}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
        <div ref={endRef} />
      </div>

      <div className="composer">
        <textarea
          value={input}
          placeholder="Ask about the corpus…"
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              submit();
            }
          }}
          rows={1}
        />
        <button onClick={submit} disabled={busy || !input.trim()}>
          {busy ? "…" : "Ask"}
        </button>
      </div>
    </main>
  );
}

/** Turn `[n]` markers into clickable chips, resolved against real citations. */
function WithCitations({
  text,
  citations,
  onCite,
}: {
  text: string;
  citations: Citation[];
  onCite: (c: Citation) => void;
}) {
  const byMarker = new Map(citations.map((c) => [c.marker, c]));
  const parts = text.split(/(\[\d{1,2}\])/g);

  return (
    <>
      {parts.map((part, i) => {
        const match = /^\[(\d{1,2})\]$/.exec(part);
        if (!match) return <Fragment key={i}>{part}</Fragment>;

        const citation = byMarker.get(Number(match[1]));
        // Markers still stream in before citations resolve, and a marker with no
        // matching passage was hallucinated — render it as plain text either way.
        if (!citation) return <Fragment key={i}>{part}</Fragment>;

        return (
          <span key={i} className="cite" onClick={() => onCite(citation)}>
            {match[1]}
          </span>
        );
      })}
    </>
  );
}
