"use client";

import { type ReactNode } from "react";

export type GuidelinePassage = {
  ref_id: string;
  doc_title: string;
  source: string;
  url?: string | null;
  chunk_id: string;
  text: string;
  score: number;
};

export type GuidelineCitationClaim = {
  claim: string;
  ref_id?: string | null;
  supported: boolean;
  evidence_span?: string | null;
  note?: string | null;
};

export type GuidelineVerifier = {
  claims: GuidelineCitationClaim[];
  total_claims: number;
  supported_count: number;
  unsupported_count: number;
  faithfulness: number;
};

export type GuidelineReference = {
  id: string;
  title: string;
  source: string;
  url?: string;
  note?: string;
};

export type GuidelineRagResult = {
  question: string;
  draft_answer: string;
  references?: GuidelineReference[];
  passages?: GuidelinePassage[];
  uncertainty?: string;
  used_fallback?: boolean;
  model?: string;
  verifier?: GuidelineVerifier | null;
};

const C = {
  border: "#e2e8f0",
  sub: "#64748b",
  ok: "#15803d",
  okBg: "#dcfce7",
  bad: "#b91c1c",
  badBg: "#fee2e2",
  accent: "#1d4ed8",
  accentBg: "#eff6ff",
  warnBg: "#fffbeb",
  warnBorder: "#fde68a",
};

function Badge({ children, bg, fg }: { children: ReactNode; bg: string; fg: string }) {
  return (
    <span
      style={{
        background: bg,
        color: fg,
        fontSize: 12,
        fontWeight: 600,
        padding: "2px 8px",
        borderRadius: 999,
        whiteSpace: "nowrap",
      }}
    >
      {children}
    </span>
  );
}

/** Render answer text with [REF#] tags highlighted. */
function AnswerWithCitations({ text }: { text: string }): ReactNode {
  const parts = text.split(/(\[REF\d+\])/g);
  return (
    <p style={{ margin: 0, lineHeight: 1.6, fontSize: 15, color: "#0f172a", whiteSpace: "pre-wrap" }}>
      {parts.map((part, i) =>
        /^\[REF\d+\]$/.test(part) ? (
          <span
            key={i}
            style={{
              background: C.accentBg,
              color: C.accent,
              fontWeight: 600,
              fontSize: 12,
              padding: "1px 5px",
              borderRadius: 4,
              margin: "0 1px",
            }}
          >
            {part}
          </span>
        ) : (
          <span key={i}>{part}</span>
        )
      )}
    </p>
  );
}

export function GuidelineRagCard({ result }: { result: GuidelineRagResult | null }) {
  if (!result) {
    return (
      <div style={{ color: C.sub, fontSize: 14, padding: 16 }}>
        No guideline RAG result yet. Ask a clinical question to ground an answer in the corpus.
      </div>
    );
  }

  const passages = result.passages ?? [];
  const verifier = result.verifier ?? null;
  const faithPct = verifier ? Math.round(verifier.faithfulness * 100) : null;

  return (
    <div
      style={{
        border: `1px solid ${C.border}`,
        borderRadius: 12,
        background: "#fff",
        padding: 18,
        display: "flex",
        flexDirection: "column",
        gap: 16,
      }}
    >
      {/* header */}
      <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
        <strong style={{ fontSize: 16, color: "#0f172a" }}>Guideline RAG</strong>
        {result.model ? <Badge bg="#f1f5f9" fg={C.sub}>{result.model}</Badge> : null}
        {result.used_fallback ? (
          <Badge bg={C.warnBg} fg="#92400e">extractive fallback (LLM offline)</Badge>
        ) : (
          <Badge bg={C.okBg} fg={C.ok}>local LLM</Badge>
        )}
        {faithPct !== null ? (
          <Badge bg={faithPct >= 100 ? C.okBg : C.badBg} fg={faithPct >= 100 ? C.ok : C.bad}>
            faithfulness {faithPct}% ({verifier!.supported_count}/{verifier!.total_claims})
          </Badge>
        ) : null}
      </div>

      {/* question */}
      <div style={{ fontSize: 13, color: C.sub }}>
        <span style={{ fontWeight: 600 }}>Q:</span> {result.question}
      </div>

      {/* answer */}
      <div
        style={{
          border: `1px solid ${C.border}`,
          borderLeft: `3px solid ${C.accent}`,
          borderRadius: 8,
          padding: "12px 14px",
          background: "#f8fafc",
        }}
      >
        <AnswerWithCitations text={result.draft_answer} />
      </div>

      {result.uncertainty ? (
        <div
          style={{
            background: C.warnBg,
            border: `1px solid ${C.warnBorder}`,
            borderRadius: 8,
            padding: "8px 12px",
            fontSize: 13,
            color: "#92400e",
          }}
        >
          ⚠ {result.uncertainty}
        </div>
      ) : null}

      {/* verifier per-claim */}
      {verifier && verifier.claims.length > 0 ? (
        <div>
          <div style={{ fontSize: 13, fontWeight: 600, color: C.sub, marginBottom: 6 }}>
            Citation faithfulness check
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            {verifier.claims.map((claim, i) => (
              <div
                key={i}
                style={{
                  display: "flex",
                  gap: 8,
                  alignItems: "flex-start",
                  fontSize: 13,
                  padding: "6px 10px",
                  borderRadius: 6,
                  background: claim.supported ? C.okBg : C.badBg,
                }}
              >
                <span style={{ color: claim.supported ? C.ok : C.bad, fontWeight: 700 }}>
                  {claim.supported ? "✓" : "✕"}
                </span>
                <span style={{ color: "#0f172a" }}>
                  <span style={{ fontWeight: 600 }}>[{claim.ref_id}] </span>
                  {claim.claim}
                  {claim.note ? (
                    <span style={{ color: C.sub, fontSize: 12 }}> — {claim.note}</span>
                  ) : null}
                </span>
              </div>
            ))}
          </div>
        </div>
      ) : null}

      {/* retrieved passages */}
      {passages.length > 0 ? (
        <div>
          <div style={{ fontSize: 13, fontWeight: 600, color: C.sub, marginBottom: 6 }}>
            Retrieved evidence ({passages.length})
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {passages.map((p) => (
              <div
                key={p.chunk_id}
                style={{ border: `1px solid ${C.border}`, borderRadius: 8, padding: "10px 12px" }}
              >
                <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 4, flexWrap: "wrap" }}>
                  <Badge bg={C.accentBg} fg={C.accent}>{p.ref_id}</Badge>
                  <span style={{ fontSize: 13, fontWeight: 600, color: "#0f172a" }}>{p.doc_title}</span>
                  <Badge bg="#f1f5f9" fg={C.sub}>cos {p.score.toFixed(3)}</Badge>
                </div>
                <div style={{ fontSize: 12, color: C.sub, marginBottom: 4 }}>
                  {p.source}
                  {p.url ? (
                    <>
                      {" · "}
                      <a href={p.url} target="_blank" rel="noreferrer" style={{ color: C.accent }}>
                        source
                      </a>
                    </>
                  ) : null}
                </div>
                <div style={{ fontSize: 13, color: "#334155", lineHeight: 1.5 }}>
                  {p.text.length > 320 ? p.text.slice(0, 320) + "…" : p.text}
                </div>
              </div>
            ))}
          </div>
        </div>
      ) : null}
    </div>
  );
}

export default GuidelineRagCard;
