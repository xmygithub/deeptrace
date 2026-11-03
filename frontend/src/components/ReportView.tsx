import { ReactNode, useEffect, useMemo, useRef, useState } from "react";
import DOMPurify from "dompurify";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Evidence, Report } from "../types/api";
import { VerificationPanel } from "./VerificationPanel";

type ReportTab = "report" | "evidence" | "verification";

export function ReportView({ report, evidence, onEvidence }: {report?: Report | null; evidence: Evidence[]; onEvidence: (item: Evidence) => void}) {
  const [tab, setTab] = useState<ReportTab>("report");
  const articleRef = useRef<HTMLElement>(null);
  useEffect(() => setTab("report"), [report?.run_id]);
  const sanitized = useMemo(() => DOMPurify.sanitize(report?.content ?? "", { ALLOWED_TAGS: [] }), [report?.content]);
  const citationIds = useMemo(() => {
    const ids: string[] = [];
    for (const match of sanitized.matchAll(/\[(?:\^)?(ev_[\w-]+)\]/g)) if (!ids.includes(match[1])) ids.push(match[1]);
    return ids;
  }, [sanitized]);
  const citationNumbers = useMemo(() => new Map(citationIds.map((id, index) => [id, index + 1])), [citationIds]);
  const markdown = useMemo(() => sanitized.replace(/\[(?:\^)?(ev_[\w-]+)\](?!\()/g, (_, id: string) => `[${citationNumbers.get(id) ?? "?"}](#evidence-${id})`), [sanitized, citationNumbers]);
  const headings = useMemo(() => report?.sections?.length ? report.sections : [...sanitized.matchAll(/^(#{1,4})\s+(.+?)\s*$/gm)].map((match, index) => ({id: `report-section-${index}`, title: match[2], level: match[1].length})), [report?.sections, sanitized]);
  const evidenceById = useMemo(() => new Map(evidence.map((item) => [item.evidence_id, item])), [evidence]);
  const indexedEvidence = report?.evidence_index ?? evidence.map((item) => ({...item, url: typeof item.metadata.url === "string" ? item.metadata.url : item.source_id}));
  if (!report?.content) return null;

  function openCitation(id: string) {
    const item = evidenceById.get(id);
    if (item) onEvidence(item); else setTab("evidence");
  }
  function jumpToHeading(index: number) {
    articleRef.current?.querySelectorAll("h1, h2, h3, h4")[index]?.scrollIntoView({behavior: "smooth", block: "start"});
  }

  return <section className="report-view">
    {report.delivery_review && <details className="report-delivery-note"><summary>交付要求检查 · {report.delivery_review.passed ? "已覆盖清单要求" : "仍有未完成要求"}</summary><p>模型辅助检查，不代表事实准确率。展开各项查看判定依据。</p>{report.delivery_review.items.map(item => <details key={item.requirement_id}><summary>{({covered: "已覆盖", partial: "部分覆盖", missing: "未覆盖", insufficient_evidence: "基础证据不足"} as Record<string, string>)[item.status] ?? item.status} · {report.delivery_review?.requirements.find(req => req.id === item.requirement_id)?.text ?? item.requirement_id}</summary><p>{item.reason}</p>{item.report_quote && <blockquote>{item.report_quote}</blockquote>}</details>)}</details>}
    {!report.delivery_review && Boolean(report.research_gaps?.length) && <div className="context-notice"><strong>检索阶段记录的缺口（尚未针对最终报告复核）：</strong>{report.research_gaps?.map((gap, index) => <p key={index}>{gap}</p>)}</div>}
    <div className="report-toolbar">
      <div className="report-tabs" role="tablist" aria-label="报告视图">
        <button className={tab === "report" ? "active" : ""} onClick={() => setTab("report")}>报告</button>
        <button className={tab === "evidence" ? "active" : ""} onClick={() => setTab("evidence")}>证据 <span>{indexedEvidence.length}</span></button>
        <button className={tab === "verification" ? "active" : ""} onClick={() => setTab("verification")}>验证 <span>{report.verification.filter((item) => item.required && item.passed).length}/{report.verification.filter((item) => item.required).length}</span></button>
      </div>
      <span className={`report-mode ${report.status !== "completed" ? "draft" : report.report_mode === "budget_fallback" ? "fallback" : ""}`}>{report.status !== "completed" ? "研究草稿 · 尚未通过交付校验" : report.report_mode === "budget_fallback" ? "预算保护报告" : "研究报告 · 已通过完成校验"}</span>
    </div>

    {tab === "report" && <div className="report-reading-layout">
      <nav className="report-toc" aria-label="报告目录"><strong>目录</strong>{headings.map((heading, index) => <button key={`${heading.id}-${index}`} className={`level-${heading.level}`} onClick={() => jumpToHeading(index)}>{heading.title}</button>)}</nav>
      <article className="report-article" ref={articleRef}>
        {report.report_mode === "budget_fallback" && <div className="report-notice"><strong>预算保护模式</strong><span>正文仅呈现可确定交付的主要发现；完整证据保存在“证据”视图。</span></div>}
        <ReactMarkdown remarkPlugins={[remarkGfm]} components={{
          a: ({href, children}) => {
            const match = href?.match(/^#evidence-(ev_[\w-]+)$/);
            if (match) return <button className="citation-link" title={evidenceById.get(match[1])?.title ?? match[1]} onClick={() => openCitation(match[1])}>{children}</button>;
            if (!href || !/^https?:\/\//i.test(href)) return <span>{children}</span>;
            return <a href={href} target="_blank" rel="noopener noreferrer">{children}</a>;
          },
          table: ({children}) => <div className="report-table-scroll"><table>{children}</table></div>,
        }}>{markdown}</ReactMarkdown>
      </article>
      <aside className="report-citation-rail"><strong>本文引用</strong>{citationIds.slice(0, 12).map((id) => <button key={id} onClick={() => openCitation(id)}><span>[{citationNumbers.get(id)}]</span><em>{evidenceById.get(id)?.title || "证据详情"}</em></button>)}{citationIds.length === 0 && <p>正文暂无可解析引用。</p>}</aside>
    </div>}

    {tab === "evidence" && <div className="report-evidence-panel">
      <header><div><h3>证据台账</h3><p>全部证据独立保存；正文只显示支撑结论的引用。</p></div><span>{indexedEvidence.length} 条</span></header>
      <div className="report-evidence-grid">{indexedEvidence.map((entry, index) => {
        const item = evidenceById.get(entry.evidence_id); const url = entry.url || entry.source_id;
        return <article key={entry.evidence_id}><div className="report-evidence-head"><span>{citationNumbers.has(entry.evidence_id) ? `[${citationNumbers.get(entry.evidence_id)}]` : `E${index + 1}`}</span><small>{entry.provider} · {entry.score.toFixed(2)}</small></div><h4>{entry.title || "未命名来源"}</h4><p>{entry.summary || "暂无摘要"}</p><footer><span>{safeDomain(url)}</span>{item && <button onClick={() => onEvidence(item)}>查看证据</button>}</footer></article>;
      })}</div>
    </div>}

    {tab === "verification" && <div className="report-verification-panel"><header><h3>交付验证</h3><p>点击各项查看满足、不满足的内容与判定依据。</p></header><VerificationPanel checks={report.verification} report={report} evidence={evidence} onEvidence={onEvidence} /></div>}
  </section>;
}

function safeDomain(value?: string | null): ReactNode {
  if (!value) return "未知来源";
  try { return new URL(value).hostname; } catch { return value.slice(0, 48); }
}
