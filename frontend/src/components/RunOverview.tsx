import { StageFeed } from "./stageFeed";
import { STAGES, StageState } from "./stageMeta";
import { CacheStats, ContextInspector, Run } from "../types/api";

export const RUN_LABELS: Record<string, string> = {
  queued: "等待开始", context_building: "构建上下文", planning: "正在规划", executing: "正在研究",
  reporting: "生成报告", verifying: "正在验证", completed: "研究完成", failed: "运行失败",
  budget_exhausted: "预算耗尽", paused: "已暂停", pausing: "正在暂停", cancelled: "已取消",
  needs_user_input: "等待补充信息", interrupted: "运行中断", retrying: "正在重试", replanning: "重新规划",
};
const STATE_LABELS: Record<StageState, string> = {pending: "未开始", busy: "进行中", done: "已完成", error: "异常", paused: "已暂停", cancelled: "已取消"};

function number(value?: number) { return value == null ? "—" : value.toLocaleString(); }

export function RunOverview({ run, feed, evidenceCount, context, cache, stageState, onStage }: {
  run: Run | null; feed: StageFeed; evidenceCount?: number; context?: ContextInspector; cache?: CacheStats;
  stageState: (id: string) => StageState; onStage: (id: string) => void;
}) {
  const usage = run?.usage?.usage;
  const limits = run?.usage?.limits;
  const elapsed = usage?.elapsed_seconds;
  const toolLimit = limits?.max_tool_calls;
  const toolCount = usage?.tool_calls;
  const stopped = ["completed", "failed", "cancelled", "budget_exhausted", "paused", "interrupted"].includes(run?.status ?? "");
  // A newly-created Run can return only {run_id, ready: false, status}.
  const readyContext = context?.ready ? context : undefined;
  return <aside className="run-overview" aria-label="运行概览">
    <div className="overview-heading"><span>运行概览</span><span className={`run-badge status-${run?.status ?? "idle"}`}>{run ? RUN_LABELS[run.status] ?? "处理中" : "尚未开始"}</span></div>
    <dl className="run-metrics">
      <div><dt>运行耗时</dt><dd>{elapsed == null ? "—" : elapsed < 60 ? `${elapsed.toFixed(1)}` : `${Math.floor(elapsed / 60)}:${String(Math.floor(elapsed % 60)).padStart(2, "0")}`}<small>{elapsed == null ? "" : elapsed < 60 ? "秒" : "分:秒"}</small></dd></div>
      <div><dt>Token 用量</dt><dd>{number(usage?.llm_tokens)}</dd></div>
      <div><dt>工具调用</dt><dd>{number(toolCount)}{toolLimit != null && <small>/ {number(toolLimit)}</small>}</dd></div>
      <div><dt>已采集证据</dt><dd>{number(evidenceCount)}{evidenceCount != null && <small>条</small>}</dd></div>
    </dl>
    {toolCount != null && toolLimit != null && toolLimit > 0 && <div className="overview-budget"><div><span>工具预算已用</span><span>{Math.round(toolCount / toolLimit * 100)}%</span></div><progress value={toolCount} max={toolLimit} aria-label="工具预算已用" /></div>}
    {run?.error_message && <div className="run-alert"><strong>{RUN_LABELS[run.status] ?? "运行提示"}</strong><p>{run.error_message}</p>{run.error_code && <code>{run.error_code}</code>}</div>}
    <section className="overview-section"><div className="overview-section-title"><h2>执行阶段</h2><span>点击查看详情</span></div>
      <ol className="run-timeline">{STAGES.map((stage, index) => {
        const state = stageState(stage.id);
        return <li key={stage.id} className={`is-${state}`}><button type="button" disabled={!run || state === "pending"} onClick={() => onStage(stage.id)}><span className="timeline-marker">{state === "done" ? "✓" : state === "error" ? "!" : String(index + 1).padStart(2, "0")}</span><span className="timeline-label">{stage.label}</span><span className="timeline-state">{STATE_LABELS[state]}</span></button></li>;
      })}</ol>
    </section>
    {feed.tools.length > 0 && <section className="overview-section"><div className="overview-section-title"><h2>最近工具调用</h2><button onClick={() => onStage("executing")}>全部 ↗</button></div><div className="overview-tools">{feed.tools.slice(-3).reverse().map((tool, index) => <button key={index} onClick={() => onStage("executing")}><span className={`tool-dot${tool.failed ? " failed" : tool.running ? stopped ? " unconfirmed" : " running" : ""}`} /><span><strong>{tool.tool_name}</strong><small>{tool.query || tool.output_preview || (tool.result_count != null ? `返回 ${tool.result_count} 条结果` : "查看执行详情")}</small></span><em>{tool.failed ? "失败" : tool.running ? stopped ? "结果待确认" : "执行中" : "已返回"}</em></button>)}</div></section>}
    <section className="overview-section">
      <div className="overview-section-title"><h2>上下文与策略</h2></div>
      <button className="context-entry" disabled={!readyContext} onClick={() => onStage("context_building")}>
        <span><strong>上下文检查</strong><small>{readyContext
          ? `${number(readyContext.total_input_tokens)} tokens · ${readyContext.curated_memory.length} 条精选记忆`
          : context || run ? "上下文正在构建…" : "开始研究后可查看"}</small></span><span>↗</span>
      </button>
      {readyContext?.selected_skill && <details className="overview-skill">
        <summary>已加载技能 · {String(readyContext.selected_skill.name)}</summary>
        <p>版本 {String(readyContext.selected_skill.version)}</p>
        <p>使用策略的模块：{readyContext.skill_policy_consumers?.join("、") || "未记录"}</p>
        <pre>{JSON.stringify(readyContext.active_skill_policy ?? readyContext.selected_skill.machine_policy ?? {}, null, 2)}</pre>
      </details>}
    </section>
    <details className="overview-metadata"><summary>运行信息与全局缓存</summary><dl><div><dt>工作流</dt><dd>{run ? run.workflow_mode === "deep_research" ? "迭代深度研究" : "计划执行报告" : "—"}</dd></div><div><dt>检索来源</dt><dd>{run ? run.source_mode === "web" ? "联网搜索" : "私有知识库" : "—"}</dd></div></dl>{run && <p className="run-id">运行编号：{run.run_id}</p>}{cache && <p>服务进程累计缓存命中率 {(cache.hit_rate * 100).toFixed(1)}% · {cache.totals.requests} 次请求（跨任务累计）</p>}</details>
  </aside>;
}
