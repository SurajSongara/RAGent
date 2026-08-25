const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type Region = { page_no: number; x0: number; y0: number; x1: number; y1: number };

export type Passage = {
  chunk_id: string;
  text: string;
  score: number;
  document_id: string;
  document_name: string;
  provenance: "paged" | "flow";
  section_path: string[];
  regions: Region[];
  char_start: number | null;
  char_end: number | null;
  retrievers: Record<string, number>;
};

export type Citation = Passage & { marker: number };

export type StageState = {
  name: string;
  status: "pending" | "running" | "succeeded" | "failed" | "skipped";
  error: string | null;
  metrics: Record<string, unknown>;
};

export type Doc = {
  id: string;
  name: string;
  status: string;
  mime_type: string;
  byte_size: number;
  page_count: number | null;
  format_family: string | null;
  provenance: "paged" | "flow" | null;
  chunk_count: number | null;
  created_at: string | null;
  stages?: StageState[];
};

async function json<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API}${path}`, init);
  if (!response.ok) {
    const detail = await response.text().catch(() => response.statusText);
    throw new Error(detail || `request failed: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export const listDocuments = () => json<{ documents: Doc[] }>("/documents");
export const getDocument = (id: string) => json<Doc>(`/documents/${id}`);
export const getDocumentText = (id: string) =>
  json<{ text: string }>(`/documents/${id}/text`);
export const getDocumentFile = (id: string) =>
  json<{ url: string; mime_type: string }>(`/documents/${id}/file`);
export const health = () =>
  json<{ status: string; llm_configured: boolean; embedding_backend: string }>("/health");

export async function upload(file: File): Promise<{ document_id: string; status: string }> {
  const body = new FormData();
  body.append("file", file);
  const response = await fetch(`${API}/documents`, { method: "POST", body });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

/** Server-sent events carry named event types, which EventSource exposes only
 *  via addEventListener — so the caller gets a typed callback per event name. */
export function streamEvents(
  path: string,
  handlers: Record<string, (data: any) => void>,
  init?: RequestInit,
): () => void {
  const controller = new AbortController();

  (async () => {
    const response = await fetch(`${API}${path}`, {
      ...init,
      signal: controller.signal,
      headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    });
    if (!response.body) return;

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // SSE frames are separated by a blank line; a partial frame stays in the
      // buffer until its terminator arrives.
      const frames = buffer.split("\n\n");
      buffer = frames.pop() ?? "";

      for (const frame of frames) {
        let event = "message";
        const dataLines: string[] = [];
        for (const line of frame.split("\n")) {
          if (line.startsWith("event:")) event = line.slice(6).trim();
          else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
        }
        if (!dataLines.length) continue;
        const handler = handlers[event];
        if (!handler) continue;
        try {
          handler(JSON.parse(dataLines.join("\n")));
        } catch {
          /* keep the stream alive through one malformed frame */
        }
      }
    }
  })().catch((error) => {
    if ((error as Error).name !== "AbortError") console.error(error);
  });

  return () => controller.abort();
}

export const askStream = (
  query: string,
  documentIds: string[] | null,
  handlers: Record<string, (data: any) => void>,
) =>
  streamEvents("/ask", handlers, {
    method: "POST",
    body: JSON.stringify({ query, document_ids: documentIds }),
  });

export const documentEvents = (id: string, handlers: Record<string, (data: any) => void>) =>
  streamEvents(`/documents/${id}/events`, handlers);
