"use client";

import { useState } from "react";
import { Ask } from "@/components/Ask";
import { CitationViewer } from "@/components/CitationViewer";
import { Sidebar } from "@/components/Sidebar";
import type { Citation } from "@/lib/api";

export default function Page() {
  const [selected, setSelected] = useState<string[]>([]);
  const [citation, setCitation] = useState<Citation | null>(null);

  return (
    <div className={`shell${citation ? " with-viewer" : ""}`}>
      <Sidebar selected={selected} onSelect={setSelected} />
      <Ask documentIds={selected} onCite={setCitation} />
      <CitationViewer citation={citation} onClose={() => setCitation(null)} />
    </div>
  );
}
