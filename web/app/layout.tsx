import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "RAGent",
  description: "Document intelligence with grounded, clickable citations.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
