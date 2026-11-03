import { useEffect, useId, useState } from "react";
import { Evidence, Report } from "../types/api";
import { VERIFICATION_LABELS } from "./stageMeta";
import { verificationDetail, verificationStatus } from "./verificationDetails";

export function VerificationPanel({ checks, report, evidence = [], onEvidence }: {
  checks: Report["verification"]; report?: Report | null; evidence?: Evidence[]; onEvidence?: (item: Evidence) => void;
}) {
  const [selected, setSelected] = useState<string | null>(null);
  const panelId = useId();
  useEffect(() => setSelected(null), [report?.run_id]);
  const check = checks.find(item => item.kind === selected);
  const detail = check ? verificationDetail(check, report) : null;
  function evidenceLink(id: string) {
    const item = evidence.find(entry => entry.evidence_id === id);
    return item && onEvidence
      ? <button type="button" className="verification-evidence-link" key={id} onClick={() => onEvidence(item)} title={item.title || id}>{item.title || id}</button>
      : <code key={id}>{id}</code>;
  }
  return <div className="verification-panel">
    <div className="verification-chips" aria-label="选择验证项目查看依据">
      {checks.length === 0 && <span className="pending">检查中…</span>}
      {checks.map(item => <button type="button" key={item.kind}
        className={`${!item.required ? "pending" : item.passed === true ? "pass" : item.passed === false ? "fail" : "pending"}${selected === item.kind ? " is-selected" : ""}`}
        aria-expanded={selected === item.kind} aria-controls={`${panelId}-${item.kind}`}
        onClick={() => setSelected(current => current === item.kind ? null : item.kind)}>
        {item.passed === true ? "✓" : item.passed === false ? "!" : "·"} {VERIFICATION_LABELS[item.kind] ?? item.kind}
      </button>)}
    </div>
    {!check && checks.length > 0 && <p className="verification-hint">点击任一检查项，查看判定条件、满足或未满足的内容。</p>}
    {check && detail && <section className="verification-detail" id={`${panelId}-${check.kind}`} aria-label={`${VERIFICATION_LABELS[check.kind] ?? check.kind}判定依据`}>
      <header><h3>{VERIFICATION_LABELS[check.kind] ?? check.kind}</h3><span>{verificationStatus(check)}</span></header>
      <p>{detail.explanation}</p>
      {detail.expected && <p><b>判定条件：</b>{detail.expected}</p>}
      {detail.boundary && <p className="verification-boundary">{detail.boundary}</p>}
      {detail.metrics.length > 0 && <dl className="verification-metrics">{detail.metrics.map(metric => <div key={metric.label}><dt>{metric.label}</dt><dd>{metric.value}</dd></div>)}</dl>}
      {detail.entries.map((entry, index) => <article className="verification-detail-entry" key={index}>
        <div><strong>{entry.label}</strong>{entry.status && <span>{entry.status}</span>}</div>
        {entry.requestQuote && <p className="verification-request">用户原话：{entry.requestQuote}</p>}
        {entry.text && <p>{entry.text}</p>}
        {entry.quotes?.map((quote, i) => <blockquote key={i}>{quote}</blockquote>)}
        {Boolean(entry.evidenceIds?.length) && <div className="verification-evidence-links">{entry.evidenceIds?.map(evidenceLink)}</div>}
      </article>)}
      {detail.entries.length === 0 && detail.emptyMessage && <p>{detail.emptyMessage}</p>}
      <details className="verification-raw"><summary>查看原始判定记录</summary><pre>{JSON.stringify(check.evidence, null, 2)}</pre></details>
    </section>}
  </div>;
}
