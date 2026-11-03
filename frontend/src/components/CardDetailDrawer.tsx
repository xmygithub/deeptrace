import { useEffect, useRef } from "react";
import { motion } from "motion/react";
import { ContextInspector, Evidence, Report, Run } from "../types/api";
import { ContextInspectorView } from "./ContextInspectorView";
import { ReportView } from "./ReportView";
import { STAGES, VERIFICATION_LABELS } from "./stageMeta";
import { StageFeed } from "./stageFeed";
import { VerificationPanel } from "./VerificationPanel";
import { RUN_LABELS } from "./RunOverview";

interface CardDetailDrawerProps {
  stageId: string;
  run: Run | null;
  feed: StageFeed;
  report?: Report | null;
  evidence: Evidence[];
  context?: ContextInspector | null;
  onEvidence: (item: Evidence) => void;
  onClose: () => void;
}

function IterationList({ feed }: { feed: StageFeed }) {
  const listRef = useRef<HTMLDivElement>(null);
  const rows = [...feed.iterations];
  const running = feed.runningIteration;
  const live = feed.liveSearches.filter((s) => s.iteration_index === running);
  const hasLive = running != null && !rows.some((entry) => entry.index === running);

  // 新迭代到达时自动滚动到底部
  useEffect(() => {
    if (!listRef.current) return;
    const frame = requestAnimationFrame(() => listRef.current?.scrollTo({ top: listRef.current.scrollHeight }));
    return () => cancelAnimationFrame(frame);
  }, [rows.length, live.length]);

  return <div className="drawer-iter-list" ref={listRef}>
    {rows.length === 0 && !hasLive && <div className="drawer-empty">等待深度研究迭代开始…</div>}
    {rows.map((entry) => (
      <div key={entry.index} className="iter-block">
        <div className="iter-block-head">
          <span className="iter-block-title">第 {entry.index + 1} 轮迭代</span>
          <span className="iter-block-meta">{entry.queries.length} 次搜索 · {entry.total_results} 条结果{entry.answer_char_count != null ? ` · 答案 ${entry.answer_char_count} 字符` : ""}</span>
        </div>
        {entry.queries.map((query, index) => (
          <div key={index} className="iter-query">
            <span className="iter-query-text">"{query.query}"</span>
            <span className={`iter-query-meta ${query.found_useful ? "ok" : "miss"}`}>{query.result_count} 条{query.found_useful ? " · 有用" : " · 未命中"}</span>
          </div>
        ))}
        {entry.info_snippets.length > 0 && (
          <div className="iter-snippets">
            {entry.info_snippets.map((snippet, index) => <span key={index} className="iter-snippet">{snippet}</span>)}
          </div>
        )}
      </div>
    ))}
    {hasLive && (
      <div className="iter-block is-live">
        <div className="iter-block-head">
          <span className="iter-block-title">第 {running! + 1} 轮迭代</span>
          <span className="pulse-dot" />
        </div>
        {live.map((search, index) => (
          <div key={index} className="iter-query">
            <span className="iter-query-text">"{search.query}"</span>
            <span className="iter-query-meta ok">搜索中…</span>
          </div>
        ))}
        {live.length === 0 && <div className="iter-query"><span className="iter-query-text">正在思考检索方向…</span></div>}
      </div>
    )}
  </div>;
}

function ToolList({ feed }: { feed: StageFeed }) {
  return <div className="drawer-tool-list">
    {feed.tools.length === 0 && <div className="drawer-empty">暂无工具调用</div>}
    {[...feed.tools].reverse().map((tool, index) => (
      <div key={index} className="tool-row">
        <span className="tool-name">{tool.tool_name}</span>
        <span className="tool-count">{tool.failed ? "失败" : tool.running ? "调用中" : "已完成"}</span>
        {tool.query && <span className="tool-query">"{tool.query}"</span>}
        {tool.result_count != null && <span className="tool-count">{tool.result_count} 条结果</span>}
      </div>
    ))}
  </div>;
}

export function CardDetailDrawer({ stageId, run, feed, report, evidence, context, onEvidence, onClose }: CardDetailDrawerProps) {
  const stage = STAGES.find((item) => item.id === stageId) ?? STAGES[2];
  return <motion.div className="drawer-backdrop" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.18 }} onClick={onClose}>
    <motion.aside
      className={`card-detail-drawer${stageId === "context_building" ? " context-drawer" : ""}${stageId === "reporting" || (stageId === "completed" && report?.content) ? " report-drawer" : ""}`}
      initial={{ x: "100%" }}
      animate={{ x: 0 }}
      exit={{ x: "100%" }}
      transition={{ type: "spring", stiffness: 320, damping: 32 }}
      onClick={(event) => event.stopPropagation()}
    >
      <header className="card-detail-head">
        <div className="card-detail-title">
          <span className="stage-icon">{stage.icon}</span>
          <h2>{stage.label}</h2>
          <span className={`stage-state is-${run?.status}`}>{RUN_LABELS[run?.status ?? ""] ?? run?.status ?? ""}</span>
        </div>
        <button className="icon-button close" onClick={onClose} aria-label="关闭">×</button>
      </header>

      {run?.error_message && <div className="inline-error">{run.error_code ? `${run.error_code}: ` : ""}{run.error_message}</div>}

      {stageId === "context_building" && (
        <div className="drawer-body">
          <p className="drawer-desc">检查实际进入本 Run 的冻结 Memory、稳定/动态上下文块、按需历史召回、压缩与局部编辑保护。</p>
          <ContextInspectorView context={context} />
        </div>
      )}

      {stageId === "planning" && (
        <div className="drawer-body">
          <p className="drawer-desc">研究计划由多个子任务构成，按依赖关系调度；失败处理受重试次数与运行预算约束。</p>
          {feed.planTasks.length === 0 && <div className="drawer-empty">等待计划生成…</div>}
          {feed.planTasks.map((task) => (
            <div key={task.task_id} className="plan-task">
              <span className="plan-task-type">{task.task_type}</span>
              <span className="plan-task-desc">{task.description}</span>
              <span className="plan-task-id">{task.task_id}</span>
            </div>
          ))}
        </div>
      )}

      {stageId === "executing" && (
        <div className="drawer-body">
          <section className="drawer-section"><h3>迭代过程</h3><IterationList feed={feed} /></section>
          <section className="drawer-section"><h3>工具调用</h3><ToolList feed={feed} /></section>
          {Boolean(feed.researchStops?.length) && <section className="drawer-section"><h3>研究阶段停止记录</h3><p className="drawer-desc">以下记录反映检索停止时的情况；后续报告或补写可能已经回答，最终结果见交付覆盖检查。</p>{feed.researchStops?.map((stop) => <details className="research-stop-entry" key={stop.task_id}><summary>{stop.task_id} · {({resolved: "基础事实采集结束", ready_for_synthesis: "基础事实已具备，转入分析写作", search_limit: "达到研究轮次上限", no_progress: "连续两轮没有新增发现", repeated_query: "下一查询重复，停止重复检索", no_evidence: "未获得来源证据", no_query: "未生成可执行的补查查询", no_next_query: "没有可执行的下一查询"} as Record<string, string>)[stop.stop_reason] ?? stop.stop_reason}</summary>{stop.open_questions.map((question, index) => <p key={index}>当时记录的事实缺口：{question}</p>)}{stop.analysis_tasks?.map((task, index) => <p key={`analysis-${index}`}>交给报告阶段的分析任务：{task.question}</p>)}</details>)}</section>}
          {report?.gap_review_status === "reviewed" && <section className="drawer-section"><h3>最终交付仍未满足的要求</h3>{report.final_gaps?.length ? report.final_gaps.map(gap => <div key={gap.requirement_id}><strong>{gap.requirement}</strong><p>{gap.reason}</p></div>) : <p>当前报告的交付覆盖审查未发现未满足项；这不等于事实准确性已验证。</p>}</section>}
          <section className="drawer-section"><h3>证据采集</h3>
            {evidence.length === 0 && <div className="drawer-empty">尚未采集到证据</div>}
            {[...evidence].sort((a, b) => b.created_at.localeCompare(a.created_at)).map((item) => (
              <button key={item.evidence_id} type="button" className="evidence-card" onClick={() => onEvidence(item)}>
                <div className="evidence-card-body">
                  <span className="evidence-title">{item.title || "未命名来源"}</span>
                  <span className="evidence-summary">{item.summary}</span>
                  <span className="evidence-meta">
                    <span className={`source-badge ${item.source_mode}`}>{item.source_mode === "web" ? "Web" : "私有库"}</span>
                    <span className="evidence-provider">{item.provider}</span>
                    <span className="evidence-score">{item.score.toFixed(2)}</span>
                  </span>
                </div>
              </button>
            ))}
          </section>
        </div>
      )}

      {stageId === "reporting" && (
        <div className="drawer-body">
          {!report?.content && <div className="drawer-empty">{run?.status === "paused" ? "报告阶段已暂停，恢复后将从这里继续。" : run?.status === "cancelled" ? "任务已取消，报告未保存。" : "报告生成中…"}</div>}
          <ReportView report={report} evidence={evidence} onEvidence={onEvidence} />
        </div>
      )}

      {stageId === "verifying" && (
        <div className="drawer-body">
          <p className="drawer-desc">点击检查项查看具体判定依据。引用、交付覆盖和证据处理分别核查；通过不代表所有事实均已验证。</p>
          <VerificationPanel checks={feed.verification} report={report} evidence={evidence} onEvidence={onEvidence} />
          {feed.verifyFailures.length > 0 && (
            <div className="verify-failures">
              {feed.recovery && <div className="verify-failure">
                <span className="verify-failure-kind">恢复动作</span>
                <span className="verify-failure-msg">{feed.recovery.attemptsExhausted
                  ? `${feed.recovery.action === "repair_report" ? "报告修复" : "重新规划"} ${feed.recovery.attempt ?? feed.recovery.maxAttempts ?? 0} 次后仍未通过`
                  : `${feed.recovery.action === "repair_report" ? "修复报告" : feed.recovery.action === "replan" ? "重新规划检索" : "无法自动修复"}${feed.recovery.attempt != null ? ` · 第 ${feed.recovery.attempt}/${feed.recovery.maxAttempts ?? "?"} 次` : ""}`}</span>
              </div>}
              {feed.verifyFailures.map((failure, index) => (
                <div key={index} className="verify-failure">
                  <span className="verify-failure-kind">{VERIFICATION_LABELS[failure.kind] ?? failure.kind}</span>
                  {failure.message && <span className="verify-failure-msg">{failure.message}</span>}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {stageId === "completed" && (
        <div className="drawer-body">
          <p className="drawer-desc">研究完成，报告通过必需的交付检查；非阻塞参考提示单独展示。点击引用可查看来源。</p>
          <dl className="drawer-stats">
            <div><dt>迭代轮次</dt><dd>{feed.iterations.length + (feed.runningIteration != null ? 1 : 0)}</dd></div>
            <div><dt>工具调用</dt><dd>{run?.usage?.usage?.tool_calls ?? feed.tools.length}</dd></div>
            <div><dt>采集证据</dt><dd>{feed.contextCount}</dd></div>
          </dl>
          {report?.content && <ReportView report={report} evidence={evidence} onEvidence={onEvidence} />}
        </div>
      )}

      {run?.status === "cancelled" && stageId !== "reporting" && (
        <div className="drawer-body"><div className="drawer-empty">任务已取消，不会继续调用模型或生成报告。</div></div>
      )}
      {run?.status === "paused" && stageId !== "reporting" && (
        <div className="drawer-body"><div className="drawer-empty">任务已暂停；恢复后将从当前卡片对应的安全阶段继续。</div></div>
      )}
    </motion.aside>
  </motion.div>;
}
