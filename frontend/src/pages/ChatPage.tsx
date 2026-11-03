import { FormEvent, KeyboardEvent, ReactNode, useEffect, useMemo, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { AnimatePresence, motion } from "motion/react";
import { api } from "../api/client";
import { CanvasBoard } from "../components/CanvasBoard";
import { CardDetailDrawer } from "../components/CardDetailDrawer";
import { ClarificationCard } from "../components/ClarificationCard";
import { ErrorState } from "../components/ErrorState";
import { EvidenceDrawer } from "../components/EvidenceDrawer";
import { FloatingComposer } from "../components/FloatingComposer";
import { MessageStrip } from "../components/MessageStrip";
import { ReportView } from "../components/ReportView";
import { RunOverview, RUN_LABELS } from "../components/RunOverview";
import { SourceSelector } from "../components/SourceSelector";
import { WhiteboardDrawer } from "../components/WhiteboardDrawer";
import { arrivedStages, STAGES, StageMeta, StageState, stageCardId } from "../components/stageMeta";
import { deriveStageFeed, StageFeed } from "../components/stageFeed";
import { useStagePositions } from "../hooks/useStagePositions";
import { useRunEvents } from "../hooks/useRunEvents";
import { recoveryNotice } from "../hooks/runStatus";
import { Evidence, Run, SourceMode, WorkflowMode } from "../types/api";

const TERMINAL = ["completed", "failed", "cancelled", "budget_exhausted", "paused"];
const ERROR_STATUS = ["failed", "budget_exhausted", "interrupted"];
/** 终态/回退阶段（本身不是画板卡片，需回溯到最后一个真实阶段） */
const NON_CARD_STAGES = ["failed", "cancelled", "budget_exhausted", "interrupted", "retrying", "replanning"];

/** 千/百万位缩写，用于 token 计数展示 */
function formatTokens(value: number | undefined): string {
  const n = value ?? 0;
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`;
  return String(n);
}

/** 命中率百分比（无观测时返回 null，不展示） */
function cacheHitRate(hit: number | undefined, miss: number | undefined): number | null {
  const total = (hit ?? 0) + (miss ?? 0);
  return total > 0 ? Math.round((100 * (hit ?? 0)) / total) : null;
}

function stageStateOf(id: string, currentStage: string | null | undefined, status: string | undefined): StageState {
  if (!status) return "pending";
  if (status === "completed") return "done";
  const current = stageCardId(currentStage);
  const currentIndex = current ? STAGES.findIndex((stage) => stage.id === current) : -1;
  const idIndex = STAGES.findIndex((stage) => stage.id === id);
  if (idIndex < currentIndex) return "done";
  if (idIndex === currentIndex) {
    if (status === "cancelled") return "cancelled";
    if (status === "paused") return "paused";
    return ERROR_STATUS.includes(status) ? "error" : "busy";
  }
  return "pending";
}

/** 执行阶段卡片：迭代轮次实时滚动列表 + 底部预算条 */
function ExecCard({ feed, run }: {feed: StageFeed; run: Run | null}) {
  const listRef = useRef<HTMLDivElement>(null);
  const running = feed.runningIteration;
  const hasLive = running != null && !feed.iterations.some((entry) => entry.index === running);
  const lastLive = hasLive ? feed.liveSearches.filter((search) => search.iteration_index === running).slice(-1)[0] : undefined;

  const rows = feed.iterations.map((entry) => ({ key: `${entry.index}-done`, label: `第 ${entry.index + 1} 轮`, sub: `${entry.queries.length} 次搜索 · ${entry.total_results} 条结果`, live: false }));
  if (hasLive && running != null) rows.push({ key: `${running}-live`, label: `第 ${running + 1} 轮`, sub: lastLive ? `正在搜索 "${lastLive.query}"` : "正在思考检索方向…", live: true });

  const isPlanWorkflow = run?.workflow_mode === "plan_execute_report";
  const planRows = feed.tools.map((tool, index) => ({
    key: `${tool.task_id ?? "tool"}-${index}`,
    label: tool.failed ? `失败 · ${tool.tool_name}` : tool.tool_name,
    sub: tool.query ?? (tool.result_count != null ? `${tool.result_count} 条结果` : tool.output_preview ?? "工具已返回"),
    live: Boolean(tool.running) && !TERMINAL.includes(run?.status ?? ""),
    failed: Boolean(tool.failed),
  }));
  const displayRows = isPlanWorkflow ? planRows : rows;

  useEffect(() => {
    if (!listRef.current) return;
    const frame = requestAnimationFrame(() => listRef.current?.scrollTo({ top: listRef.current.scrollHeight }));
    return () => cancelAnimationFrame(frame);
  }, [displayRows.length, lastLive?.query]);

  const usage = run?.usage?.usage;
  const limits = run?.usage?.limits;
  const toolPct = usage?.tool_calls && limits?.max_tool_calls ? Math.min(100, Math.round((usage.tool_calls / limits.max_tool_calls) * 100)) : 0;
  const cachePct = cacheHitRate(usage?.prefix_cache_hit_tokens, usage?.prefix_cache_miss_tokens);

  const executionSummary = run?.status === "failed"
    ? ["NO_SOURCE_EVIDENCE", "INSUFFICIENT_SOURCE_EVIDENCE"].includes(run.error_code ?? "")
      ? "私域资料不足，研究已停止"
      : "研究执行失败"
    : run?.status === "budget_exhausted"
      ? "研究预算耗尽，执行已停止"
      : run?.status === "paused"
        ? "执行已暂停，恢复后继续未完成任务"
        : `正在执行计划任务 · ${feed.taskCount}/${feed.planTasks.length || feed.taskStartedCount || 1}`;
  return <div className="exec-card">
    <div className="iter-rows" ref={listRef}>
      {displayRows.length === 0 && <div className="iter-row is-empty">{isPlanWorkflow ? executionSummary : "等待深度研究迭代开始…"}</div>}
      {displayRows.map((row) => (
        <div key={row.key} className={`iter-row${row.live ? " is-live" : ""}${"failed" in row && row.failed ? " is-failed" : ""}`}>
          <span className="iter-row-label">{row.label}</span>
          <span className="iter-row-sub">{row.sub}</span>
          {row.live && <span className="pulse-dot" />}
        </div>
      ))}
    </div>
    {feed.tools.length > 0 && <div className="budget-row compact"><span>调用工具</span><span className="budget-num">{[...new Set(feed.tools.map((tool) => tool.tool_name))].join("、")}</span></div>}
    {usage && limits && usage.tool_calls != null && (
      <div className="budget-row compact">
        <span>工具调用</span>
        <span className="budget-fill-track"><i className="budget-fill" style={{ width: `${toolPct}%` }} /></span>
        <span className="budget-num">{usage.tool_calls} / {limits.max_tool_calls}</span>
      </div>
    )}
    {cachePct != null && (
      <div className="budget-row compact">
        <span>前缀缓存</span>
        <span className={`cache-pct${cachePct >= 50 ? " is-good" : ""}`}>{cachePct}%</span>
        <span className="budget-num">命中 {formatTokens(usage?.prefix_cache_hit_tokens)} · 未命中 {formatTokens(usage?.prefix_cache_miss_tokens)} · {usage?.prefix_cache_requests} 次请求</span>
      </div>
    )}
  </div>;
}

function CardSummary({ lines, meta }: {lines: string[]; meta?: string}) {
  return <div className="stage-summary">
    {lines.map((line, index) => <span key={index} className="stage-summary-line">{line}</span>)}
    {meta && <span className="stage-summary-meta">{meta}</span>}
  </div>;
}

export function ChatPage({ sessionId }: {sessionId?: string}) {
  const queryClient = useQueryClient();
  const [text, setText] = useState("");
  const [source, setSource] = useState<SourceMode>(() => (localStorage.getItem("source_mode") as SourceMode) || "graphrag");
  const [workflow, setWorkflow] = useState<WorkflowMode>("deep_research");
  const [runId, setRunId] = useState<string | null>(null);
  const [selectedEvidence, setSelectedEvidence] = useState<Evidence | null>(null);
  const [selectedStage, setSelectedStage] = useState<string | null>(null);
  const [whiteboardOpen, setWhiteboardOpen] = useState(false);
  const [workspaceView, setWorkspaceView] = useState<"report" | "canvas" | "messages" | "overview">("report");
  const [composerCollapsed, setComposerCollapsed] = useState(() => localStorage.getItem("chat.composerCollapsed") === "1");
  const [error, setError] = useState<unknown>(null);
  const [sending, setSending] = useState(false);
  const [pausing, setPausing] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  const [resuming, setResuming] = useState(false);
  const dragMovedRef = useRef(false);

  const capabilities = useQuery({ queryKey: ["capabilities"], queryFn: api.capabilities });
  const detail = useQuery({ queryKey: ["session", sessionId], queryFn: () => api.session(sessionId!), enabled: Boolean(sessionId) });
  const { events, run: observedRun, connection, refresh, restart } = useRunEvents(runId);
  // Changing the selected record must not briefly show the previous run's status or metrics.
  const run = observedRun?.run_id === runId ? observedRun : null;
  const evidence = useQuery({ queryKey: ["evidence", runId], queryFn: () => api.evidence(runId!), enabled: Boolean(runId), refetchInterval: run && !TERMINAL.includes(run.status) ? 5000 : false });
  const report = useQuery({ queryKey: ["report", runId], queryFn: () => api.report(runId!), enabled: Boolean(runId), refetchInterval: run && !TERMINAL.includes(run.status) ? 5000 : false });
  const contextInspector = useQuery({ queryKey: ["context", runId], queryFn: () => api.context(runId!), enabled: Boolean(runId), refetchInterval: run && !TERMINAL.includes(run.status) ? 5000 : false });
  const cacheStats = useQuery({ queryKey: ["cacheStats"], queryFn: api.cacheStats, refetchInterval: 5000 });
  const recovery = useMemo(() => recoveryNotice(run ? events : []), [run, events]);
  useEffect(() => { if (recovery && !recovery.waiting) setResuming(false); }, [recovery?.waiting]);

  useEffect(() => { localStorage.setItem("source_mode", source); }, [source]);
  useEffect(() => { localStorage.setItem("chat.composerCollapsed", composerCollapsed ? "1" : "0"); }, [composerCollapsed]);
  useEffect(() => {
    const desktop = window.matchMedia("(min-width: 1051px)");
    const sync = () => { if (desktop.matches) setWorkspaceView(value => value === "overview" ? "report" : value); };
    desktop.addEventListener("change", sync);
    return () => desktop.removeEventListener("change", sync);
  }, []);
  useEffect(() => {
    setWorkspaceView("report"); setRunId(null); setSelectedEvidence(null); setSelectedStage(null); setPausing(false); setCancelling(false); setResuming(false); setError(null);
  }, [sessionId]);
  useEffect(() => {
    if (!detail.data || detail.data.session_id !== sessionId || runId) return;
    const latest = detail.data.runs.find((item) => !TERMINAL.includes(item.status)) ?? detail.data.runs[0];
    if (latest) setRunId(latest.run_id);
  }, [detail.data, runId, sessionId]);
  useEffect(() => {
    if (!run) return;
    setSource(run.source_mode); setWorkflow(run.workflow_mode);
    if (run.status === "paused" || (!run.pause_requested && run.status !== "pausing")) setPausing(false);
    if (run.status === "cancelled") setCancelling(false);
    if (run.status !== "paused" && run.status !== "pausing") setResuming(false);
  }, [run?.run_id, run?.status, run?.pause_requested]);
  useEffect(() => {
    if (!capabilities.data || capabilities.data.sources[source]?.available) return;
    const fallback = ([capabilities.data.default_source_mode, "graphrag", "web"] as SourceMode[]).find((mode) => capabilities.data!.sources[mode]?.available);
    if (fallback) setSource(fallback);
  }, [capabilities.data, source]);
  useEffect(() => {
    if (run && [...TERMINAL, "needs_user_input"].includes(run.status)) {
      queryClient.invalidateQueries({ queryKey: ["session", sessionId] });
      queryClient.invalidateQueries({ queryKey: ["report", runId] });
      queryClient.invalidateQueries({ queryKey: ["evidence", runId] });
      queryClient.invalidateQueries({ queryKey: ["context", runId] });
    }
  }, [run?.status]);

  const messageRuns = useMemo(() => new Map(detail.data?.runs.map((item) => [item.run_id, item]) ?? []), [detail.data]);
  const feed = useMemo(() => deriveStageFeed(run ? events : [], report.data), [events, report.data, run]);
  // 终态/回退阶段回溯到最后一个真实阶段（failed 时显示失败发生处的卡片，不渲染幻影卡）
  const effectiveStage = useMemo(() => {
    const raw = run?.current_stage;
    if (!raw || !NON_CARD_STAGES.includes(raw)) return raw;
    const last = [...events].reverse().find((event) =>
      (event.event_type === "run.stage_changed" || event.event_type === "run.started")
      && typeof event.status === "string"
      && STAGES.some((stage) => stage.id === event.status)
    );
    return last ? String(last.status) : raw;
  }, [run?.current_stage, events]);
  const arrived = useMemo(() => {
    if (!run) return [];
    const reached = [effectiveStage, ...events.flatMap((event) => [event.stage, typeof event.status === "string" ? event.status : undefined])]
      .filter((value): value is string => Boolean(value));
    const highest = reached.reduce((best, value) => {
      const index = STAGES.findIndex((stage) => stage.id === value);
      return index > best.index ? { value, index } : best;
    }, { value: effectiveStage ?? "", index: STAGES.findIndex((stage) => stage.id === effectiveStage) });
    return arrivedStages(highest.value);
  }, [effectiveStage, events, run]);
  const { positions, update: updatePosition } = useStagePositions(runId, arrived);

  async function submit(event: FormEvent) {
    event.preventDefault();
    const sourceAvailable = capabilities.data?.sources[source]?.available ?? false;
    const resumable = run?.status === "paused" || run?.status === "pausing" || Boolean(run?.pause_requested) || pausing;
    if (!sessionId || !text.trim() || sending || (!sourceAvailable && !resumable) || (run && !TERMINAL.includes(run.status) && !resumable)) return;
    setSending(true); setError(null);
    try {
      const result = await api.send(sessionId, { client_message_id: crypto.randomUUID(), content: text.trim(), source_mode: source, workflow_mode: workflow });
      const sameRun = result.run_id === runId;
      setText(""); setRunId(result.run_id);
      if (sameRun) restart();
      await queryClient.invalidateQueries({ queryKey: ["session", sessionId] });
    } catch (caught) { setError(caught); } finally { setSending(false); }
  }

  function onComposerKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault(); event.currentTarget.form?.requestSubmit();
    }
  }

  async function pauseRun() {
    if (!run || pausing) return;
    setPausing(true); setError(null);
    try { await api.pause(run.run_id); await refresh(); }
    catch (caught) { setError(caught); setPausing(false); }
  }

  async function cancelRun() {
    if (!run || cancelling) return;
    setCancelling(true); setError(null);
    try { await api.cancel(run.run_id); await refresh(); }
    catch (caught) { setError(caught); setCancelling(false); }
  }

  async function resumeRun() {
    if (!run || (!["paused", "pausing"].includes(run.status) && !run.pause_requested && !pausing) || resuming) return;
    setResuming(true); setError(null);
    try { await api.resume(run.run_id); restart(); await refresh(); }
    catch (caught) { setError(caught); setResuming(false); }
  }

  function openEvidence(item: Evidence) { setSelectedEvidence(item); }
  function openMessageReport(messageRunId: string) { setRunId(messageRunId); setSelectedEvidence(null); setSelectedStage(null); setWorkspaceView("report"); }

  function cardContent(stage: StageMeta): ReactNode {
    switch (stage.id) {
      case "context_building": {
        const info = feed.contextInfo;
        return <CardSummary lines={info ? [`已复用 ${info.usedMessages} 条当前会话消息`, `精选 Memory ${info.memories} 条 · 历史召回 ${info.historicalRecall} 个 Session`] : ["构建检索上下文…"]} meta={info ? `上下文 ${info.tokens} tokens` : `证据 ${feed.contextCount}`} />;
      }
      case "planning": {
        const tasks = feed.planTasks;
        return <CardSummary
          lines={tasks.length > 0 ? tasks.slice(0, 2).map((task) => `${task.task_type}: ${task.description}`) : ["正在规划研究步骤…"]}
          meta={tasks.length > 2 ? `${tasks.length} 个任务 · 另有 ${tasks.length - 2} 项` : `${tasks.length} 个任务`}
        />;
      }
      case "executing": return <ExecCard feed={feed} run={run} />;
      case "reporting": return <CardSummary
        lines={[run?.status === "paused"
          ? "报告阶段已暂停，恢复后继续"
          : run?.status === "cancelled"
          ? "任务已取消，未生成报告"
          : run?.status === "reporting" && feed.recovery?.action === "repair_report"
            ? `正在修复报告 · 第 ${feed.recovery.attempt ?? 1}/${feed.recovery.maxAttempts ?? "?"} 次`
            : feed.reportChars != null ? `报告已生成 · ${feed.reportChars} 字符` : "正在生成研究报告…"]}
        meta={`${feed.taskCount} 个任务完成`}
      />;
      case "verifying": {
        const done = feed.verification.filter((check) => check.passed != null).length;
        const passed = feed.verification.filter((check) => check.passed === true).length;
        const lines = run?.status === "paused" ? ["验证阶段已暂停，恢复后继续"] : done > 0 ? [`已核查 ${done} 项 · ${passed} 项通过`] : ["正在逐项验证…"];
        if (feed.verifyFailures.length > 0) {
          if (feed.recovery?.attemptsExhausted && feed.recovery.action === "repair_report") lines.push(`报告自动修复 ${feed.recovery.attempt ?? feed.recovery.maxAttempts ?? 0} 次后仍未通过`);
          else if (feed.recovery?.attemptsExhausted && feed.recovery.action === "replan") lines.push(`重新规划 ${feed.recovery.attempt ?? feed.recovery.maxAttempts ?? 0} 次后仍未通过`);
          else if (feed.recovery?.action === "repair_report") lines.push(`失败 ${feed.verifyFailures.length} 项，将修复报告 · 第 ${feed.recovery.attempt ?? 1}/${feed.recovery.maxAttempts ?? "?"} 次`);
          else if (feed.recovery?.action === "replan") lines.push(`失败 ${feed.verifyFailures.length} 项，将重新规划检索 · 第 ${feed.recovery.attempt ?? 1}/${feed.recovery.maxAttempts ?? "?"} 次`);
          else lines.push(`失败 ${feed.verifyFailures.length} 项，无法自动修复`);
        }
        return <CardSummary lines={lines} />;
      }
      case "completed": return <CardSummary
        lines={[feed.verifyFailures.length === 0 ? "全部验证通过，研究完成" : "研究结束"]}
        meta={`证据 ${feed.contextCount} · 工具 ${run?.usage?.usage?.tool_calls ?? feed.tools.length}`}
      />;
    }
  }

  if (!sessionId) return <main className="empty-chat workspace-welcome"><div><span className="welcome-kicker">DEEPTRACE / RESEARCH WORKSPACE</span><span className="hero-mark">D</span><h1>从一个问题，走向有据可查的结论。</h1><p>在左侧新建研究会话，选择信息来源，开始探索。</p><div className="welcome-features"><div><span>01</span><strong>研究与规划</strong><p>迭代探索，或拆分任务逐步研究。</p></div><div><span>02</span><strong>可靠执行</strong><p>追踪进度、控制预算、暂停与恢复。</p></div><div><span>03</span><strong>证据与报告</strong><p>查看来源、研究结论与交付检查。</p></div></div></div></main>;
  return <main className="chat-page research-workspace">
    <header className="chat-header">
      <div className="workspace-title"><span className="eyebrow">研究工作台 <span>/</span> {run?.source_mode === "web" ? "联网搜索" : run ? "私有知识库" : "新研究"}</span><h1 title={detail.data?.title}>{detail.data?.title ?? "加载中…"}</h1></div>
      <div className="chat-header-actions">
        <span className={`connection-dot${connection === "reconnecting" ? " reconnecting" : ""}`}>{run && TERMINAL.includes(run.status) ? "记录已保存" : connection === "reconnecting" ? "连接重试中" : connection === "connected" ? "实时同步" : "等待连接"}</span>
        <button type="button" className="whiteboard-button" onClick={() => setWhiteboardOpen(true)}>运行日志 ↗</button>
        <button type="button" className="icon-btn" onClick={() => setComposerCollapsed(value => !value)} title={composerCollapsed ? "展开输入框" : "隐藏输入框"} aria-label={composerCollapsed ? "展开输入框" : "隐藏输入框"}>{composerCollapsed ? "＋" : "⌄"}</button>
      </div>
    </header>
    <div className={`workspace-layout view-${workspaceView}`}>
      <section className="workspace-main" aria-label="研究内容">
        <div className="workspace-toolbar"><nav aria-label="工作区视图"><button aria-pressed={workspaceView === "report"} onClick={() => setWorkspaceView("report")}>研究报告</button><button aria-pressed={workspaceView === "canvas"} onClick={() => setWorkspaceView("canvas")}>流程白板</button><button aria-pressed={workspaceView === "messages"} onClick={() => setWorkspaceView("messages")}>会话记录 <span>{detail.data?.messages.length ?? 0}</span></button><button className="mobile-overview-tab" aria-pressed={workspaceView === "overview"} onClick={() => setWorkspaceView("overview")}>运行概览</button></nav>
        {Boolean(detail.data?.runs.length) && <select aria-label="选择研究记录" value={runId ?? ""} onChange={event => {setRunId(event.target.value); setSelectedStage(null); setSelectedEvidence(null);}}>{detail.data?.runs.map((item, index) => <option key={item.run_id} value={item.run_id}>第 {detail.data!.runs.length - index} 次研究 · {RUN_LABELS[item.status] ?? "处理中"}</option>)}</select>}</div>
        <div className="workspace-scroll">
          {recovery && <div className="context-notice" role="status"><strong>{recovery.waiting ? "等待恢复运行" : "恢复等待已结束"}</strong><p>{recovery.message}</p></div>}
          <ErrorState error={detail.error} />
          {workspaceView === "report" && (report.data?.content ? <ReportView report={report.data} evidence={evidence.data?.items ?? []} onEvidence={openEvidence} /> : <div className="research-placeholder"><span className={`placeholder-icon${run && !TERMINAL.includes(run.status) ? " is-working" : ""}`}>◈</span><span className="eyebrow">{run ? "研究进展" : "准备就绪"}</span><h2>{run ? RUN_LABELS[run.status] ?? "正在处理研究任务" : "让研究从这里开始"}</h2><p>{run ? run.status === "completed" ? "当前研究尚无可展示的报告，请查看运行详情。" : TERMINAL.includes(run.status) ? "本次运行尚未生成报告。可在右侧查看停止位置与运行详情。" : "研究报告将在这里生成。右侧可以实时查看执行阶段、工具调用和已采集的证据。" : "在下方输入研究问题。报告、证据和验证结果会汇集在这个工作区。"}</p>{run && <button className="placeholder-detail" onClick={() => setSelectedStage(stageCardId(effectiveStage) ?? "context_building")}>查看当前阶段 ↗</button>}{Boolean(evidence.data?.items.length) && <div className="pending-evidence"><h3>已采集的证据 <span>{evidence.data?.items.length}</span></h3>{evidence.data?.items.slice(0, 5).map(item => <button key={item.evidence_id} onClick={() => openEvidence(item)}><span>↗</span>{item.title || item.source_id}</button>)}</div>}<ErrorState error={report.error} /></div>)}
          {workspaceView === "canvas" && <><div className="canvas-caption"><span>执行流程白板</span><p>拖动卡片调整布局，点击查看阶段详情。</p></div><CanvasBoard runId={runId} arrived={arrived} positions={positions} onPositionChange={updatePosition} stageState={id => stageStateOf(id, effectiveStage, run?.status)} cardContent={cardContent} onSelectCard={setSelectedStage} dragMovedRef={dragMovedRef} />{!run && <div className="empty-state">开始研究后，这里会展示各阶段的执行过程。</div>}</>}
          {workspaceView === "messages" && <MessageStrip collapsed={false} onToggle={() => setWorkspaceView("report")} messages={detail.data?.messages ?? []} runs={messageRuns} onOpenReport={openMessageReport} />}
        </div>
        <div className="workspace-compose-area">
    <ErrorState error={error} />
    <AnimatePresence mode="wait" initial={false}>
      {composerCollapsed ? (
        <FloatingComposer key="fab" onClick={() => setComposerCollapsed(false)} />
      ) : (
        <motion.form key="composer" className="composer" onSubmit={submit}
          initial={{ opacity: 0, y: 24 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 24, scale: 0.9 }}
          transition={{ type: "spring", stiffness: 320, damping: 26 }}
        >
          <div className="composer-tools"><SourceSelector value={source} onChange={setSource} capabilities={capabilities.data} /><select aria-label="研究工作流" value={workflow} onChange={(event) => setWorkflow(event.target.value as WorkflowMode)}><option value="deep_research">深度研究 · DeepResearch</option><option value="plan_execute_report">计划执行报告 · PER</option></select></div>
          {source === "web" && capabilities.data?.sources.web.reason && <small className="capability-note">{capabilities.data.sources.web.reason}</small>}
          <textarea value={text} onChange={(event) => setText(event.target.value)} onKeyDown={onComposerKeyDown} aria-label="研究问题" placeholder="提出研究问题，或基于已有结果继续追问…" rows={2} />
          <div className="composer-actions">
            <span>{(["paused", "pausing"].includes(run?.status ?? "") || run?.pause_requested || pausing) ? "输入“继续 / 接着做 / 按计划执行”等，或点击恢复" : "Enter 发送 · Shift+Enter 换行 · 本次研究使用所选来源"}</span>
            <div className="composer-control-actions">
              {(["paused", "pausing"].includes(run?.status ?? "") || run?.pause_requested || pausing) ? <>
                <button type="button" className="resume" disabled={resuming} onClick={resumeRun}>{resuming ? "正在恢复…" : run?.status === "paused" ? "恢复运行" : "撤销暂停并继续"}</button>
                <button className="send" disabled={sending || !text.trim()}>{sending ? "发送中…" : "发送"}</button>
                <button type="button" className="stop" disabled={cancelling} onClick={cancelRun}>{cancelling ? "正在取消…" : "取消"}</button>
              </> : run && !TERMINAL.includes(run.status) ? <>
                <button type="button" className="pause" disabled={pausing || run.status === "pausing"} onClick={pauseRun}>{pausing || run.status === "pausing" ? "正在暂停…" : "暂停"}</button>
                <button type="button" className="stop" disabled={cancelling || run.cancellation_requested} onClick={cancelRun}>{cancelling || run.cancellation_requested ? "正在取消…" : "取消"}</button>
              </> : <button className="send" disabled={sending || !text.trim() || !(capabilities.data?.sources[source]?.available ?? false)}>{sending ? "发送中…" : "发送研究 →"}</button>}
            </div>
          </div>
        </motion.form>
      )}
    </AnimatePresence>

    {run?.status === "needs_user_input" && <ClarificationCard question={run.error_message} onSubmit={async (content) => { await api.clarify(run.run_id, content); await refresh(); }} />}
        </div>
      </section>
      <RunOverview run={run} feed={feed} evidenceCount={evidence.data?.items.length} context={contextInspector.data} cache={cacheStats.data} stageState={id => stageStateOf(id, effectiveStage, run?.status)} onStage={setSelectedStage} />
    </div>
    <AnimatePresence>{selectedStage && run && <CardDetailDrawer stageId={selectedStage} run={run} feed={feed} report={report.data} evidence={evidence.data?.items ?? []} context={contextInspector.data} onEvidence={openEvidence} onClose={() => setSelectedStage(null)} />}</AnimatePresence>
    <EvidenceDrawer item={selectedEvidence} onClose={() => setSelectedEvidence(null)} />
    <WhiteboardDrawer sessionId={sessionId} open={whiteboardOpen} live={Boolean(run && !TERMINAL.includes(run.status))} onClose={() => setWhiteboardOpen(false)} />
  </main>;
}
