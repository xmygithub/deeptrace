import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../api/client";
import { ErrorState } from "../components/ErrorState";
import { EvolutionReviewDetail } from "../types/api";
import { criticPresentation, checkLabel, evolutionLabel, formatCount, formatDelta, formatPercent, formatSeconds, holdoutLabel } from "./evolutionDisplay";

const PIPELINE = [
  ["run", "研究记录", "收集任务、工具、证据与校验结果"],
  ["review_pack", "轨迹整理", "整理执行片段与来源记录"],
  ["proposal", "生成提案", "建议新建、更新或不生成技能"],
  ["critic", "独立审查", "检查复用价值、依据与安全"],
  ["validation", "安全检查", "检查权限、来源和轨迹引用"],
  ["candidate", "候选技能", "生成技能文档和执行规则"],
  ["evaluation", "对照评测", "比较未启用与启用后的表现"],
  ["deployment", "分阶段启用", "影子验证、灰度试用与正式启用"],
] as const;

const EVENT_NAMES: Record<string, string> = {
  "learning.review.queued": "学习任务入队", "learning.review.started": "后台复盘开始",
  "learning.review_pack.built": "轨迹整理完成", "learning.review.ignored": "本次轨迹无需生成技能",
  "learning.review.failed": "学习任务失败", "learning.review.retried": "学习任务重试",
  "skill.proposal.created": "技能提案已生成", "skill.proposal.revised": "技能提案已修订",
  "skill.critic.completed": "独立审查完成", "skill.critic.rejected": "技能提案被驳回",
  "skill.validation.completed": "安全检查完成", "skill.validation.failed": "安全检查未通过",
  "skill.candidate.created": "候选技能已生成", "skill.candidate_created": "候选技能已生成", "skill.evaluation.started": "对照评测开始",
  "skill.evaluation.case_started": "评测用例开始", "skill.evaluation.arm_completed": "一组评测执行完成",
  "skill.evaluation.evaluation_completed": "对照评测完成", "skill.shadow_started": "进入影子验证",
  "skill.canary_started": "进入灰度试用", "skill.activated": "技能已正式启用",
  "skill.suspended": "技能已暂停", "skill.rolled_back": "技能版本已回滚",
};

function record(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {};
}
function list(value: unknown): Array<Record<string, unknown>> {
  return Array.isArray(value) ? value.filter((item) => item && typeof item === "object") as Array<Record<string, unknown>> : [];
}
function json(value: unknown) { return <pre className="evo-json">{JSON.stringify(value ?? {}, null, 2)}</pre>; }
function time(value: unknown) { return value ? new Date(String(value)).toLocaleString() : "—"; }

function RawRecord({ value, label = "查看原始记录" }: { value: unknown; label?: string }) {
  return <details className="evo-raw"><summary>{label}</summary>{json(value)}</details>;
}

function StatusBadge({ value }: { value: unknown }) {
  return <span className={`evo-status ${String(value ?? "pending")}`} title={String(value ?? "")}>{evolutionLabel(value)}</span>;
}

function RunResult({ value }: { value: Record<string, unknown> }) {
  return <><dl className="evo-result-facts">
    <div><dt>完成校验</dt><dd>{checkLabel(value.completed)}</dd></div>
    <div><dt>引用 ID 有效率</dt><dd>{formatPercent(value.citation_integrity)}</dd></div>
    <div><dt>Token 用量</dt><dd>{formatCount(value.llm_tokens)}</dd></div>
    <div><dt>运行耗时</dt><dd>{formatSeconds(value.wall_time_seconds)}</dd></div>
  </dl><RawRecord value={value} /></>;
}

function stageState(detail: EvolutionReviewDetail, key: string) {
  const failedStage = String(record(record(detail.diagnostics).review_failure).failed_stage ?? "");
  const failedStageKeys: Record<string, string> = {building_pack: "review_pack", proposing: "proposal", criticizing: "critic", revising: "proposal", writing_candidate: "candidate"};
  if (detail.status === "failed" && failedStageKeys[failedStage] === key) return "blocked";
  if (key === "critic") return criticPresentation(detail).stage;
  if (detail.proposal?.decision === "ignore" && ["validation", "candidate", "evaluation", "deployment"].includes(key)) return "skipped";
  if (key === "validation" && detail.validation?.passed === false) return "blocked";
  if (key === "evaluation" && ["failed", "rejected"].includes(String(detail.evaluation?.status))) return "blocked";
  const exists: Record<string, boolean> = {
    run: Boolean(detail.run), review_pack: Object.keys(detail.review_pack ?? {}).length > 0,
    proposal: Object.keys(detail.proposal ?? {}).length > 0, critic: Object.keys(detail.critic ?? {}).length > 0,
    validation: Object.keys(detail.validation ?? {}).length > 0, candidate: Boolean(detail.candidate),
    evaluation: Boolean(detail.evaluation), deployment: (detail.deployments ?? []).length > 0,
  };
  if (exists[key]) return "done";
  const checkpoint = String(record(detail.checkpoint).stage ?? detail.status);
  const aliases: Record<string, string> = {building_pack: "review_pack", proposing: "proposal", criticizing: "critic", revising: "proposal", writing_candidate: "candidate"};
  return aliases[checkpoint] === key || checkpoint === key ? "active" : "pending";
}

function EvaluationPanel({detail}: {detail: EvolutionReviewDetail}) {
  const evaluation = record(detail.evaluation); const metrics = record(evaluation.metrics);
  const pairs = list(metrics.pairs);
  if (!Object.keys(evaluation).length) return <div className="evo-empty">尚未进行候选技能的对照评测。</div>;
  return <div className="evo-eval">
    <p className="evo-explanation">同一问题分别不加载和加载候选技能，比较完成校验通过率与执行开销；通过率不等于答案准确率。</p>
    <div className="evo-metrics compact">
      <div><small>评测状态</small><strong>{evolutionLabel(evaluation.status)}</strong></div>
      <div><small>未启用通过率</small><strong>{formatPercent(metrics.active_contract_pass_rate)}</strong></div>
      <div><small>启用后通过率</small><strong>{formatPercent(metrics.contract_pass_rate)}</strong></div>
      <div><small>Token 差值</small><strong>{formatDelta(metrics.token_delta)}</strong></div>
      <div><small>耗时差值</small><strong>{formatDelta(metrics.latency_delta_seconds, " 秒")}</strong></div>
      <div title="单独保留的测试用例；没有相关用例时显示未评测"><small>保留测试集</small><strong>{holdoutLabel(metrics)}</strong></div>
    </div>
    <p className="evo-explanation">差值 = 启用候选技能 − 未启用；正数表示消耗增加，负数表示消耗减少。</p>
    <div className="evo-pairs">
      {pairs.map((pair, index) => { const control = record(pair.control); const treatment = record(pair.treatment); const testCase = record(pair.case); return <details key={index}>
        <summary><span>用例 {index + 1} · {evolutionLabel(testCase.split ?? "eval")}</span><b className={treatment.completed === true ? "ok" : treatment.completed === false ? "bad" : ""}>启用后：{checkLabel(treatment.completed)}</b></summary>
        <p>{String(testCase.query ?? "隐藏用例")}</p>
        <div className="evo-compare"><section><h5>未启用候选技能</h5><RunResult value={control} /></section><section><h5>启用候选技能后</h5><RunResult value={treatment} /></section></div>
      </details>; })}
    </div>
    {!pairs.length && <div className="evo-empty">没有保存逐项对照记录，无法展示用例比较。</div>}
  </div>;
}

function Detail({detail, onRetry, retrying}: {detail: EvolutionReviewDetail; onRetry: () => void; retrying: boolean}) {
  const pack = detail.review_pack ?? {}; const proposal = detail.proposal ?? {}; const critic = detail.critic ?? {};
  const validation = detail.validation ?? {}; const candidate = record(detail.candidate); const candidatePayload = record(candidate.payload);
  const spec = record(candidatePayload.spec); const content = String(candidatePayload.content ?? "");
  const model = record(record(pack.context_and_budget_metrics).model_snapshot);
  const diagnostics = record(detail.diagnostics);
  const researchFailures = list(diagnostics.research_failures);
  const reviewFailure = record(diagnostics.review_failure);
  const episodes = list(pack.episodes).filter(episode => !(
    detail.diagnostics && episode.summary === "运行中出现失败事件。"
  ));
  const criticView = criticPresentation(detail);
  const errors = Array.isArray(validation.errors) ? validation.errors : [];
  const warnings = Array.isArray(validation.warnings) ? validation.warnings : [];
  const checked = typeof validation.passed === "boolean";
  return <section className="evo-detail">
    <header className="evo-detail-head"><div><span className="eyebrow">复盘详情</span><h2>{String(proposal.title ?? "技能复盘记录")}</h2><p>{detail.goal}</p>{Boolean(proposal.name ?? proposal.target_skill_id) && <p className="evo-skill-id">技能标识：{String(proposal.name ?? proposal.target_skill_id)}</p>}</div><div className="evo-detail-status"><StatusBadge value={detail.status} /><code title="复盘编号">{detail.review_id}</code>{["failed", "rejected"].includes(detail.status) && <button onClick={onRetry} disabled={retrying}>{retrying ? "重试中…" : "重新复盘"}</button>}</div></header>
    <div className="evo-pipeline">{PIPELINE.map(([key, label, sub]) => { const state = stageState(detail, key); return <div className={`evo-stage ${state}`} key={key} title={state === "blocked" ? "未通过" : state === "done" ? "已处理" : state === "active" ? "进行中" : state === "skipped" ? "无需执行" : "未开始"}><i>{state === "done" ? "✓" : state === "blocked" ? "!" : ""}</i><strong>{label}</strong><small>{key === "critic" ? criticView.description : state === "skipped" ? "本次流程无需执行" : sub}</small></div>; })}</div>

    {Object.keys(reviewFailure).length > 0 && <div className="evo-issues" role="alert">
      <p><b>后续 Skill 复盘失败</b> · 研究任务：{evolutionLabel(record(detail.run).status)}</p>
      <p>{String(reviewFailure.summary)} 阶段：{evolutionLabel(reviewFailure.failed_stage)}{reviewFailure.stage_inferred === true ? "（根据已保存的阶段结果推断）" : ""}；失败尝试：{String(reviewFailure.attempts)} 次。</p>
      <p>这是研究结束后的技能提炼流程，研究结果请查看原会话；候选技能和对照评测以各阶段实际记录为准。</p>
      <details><summary>查看复盘错误详情</summary><pre className="evo-json">{String(detail.error_message ?? "未保存")}</pre></details>
    </div>}
    <div className="evo-detail-grid">
      <article><h3>01 · 研究执行记录</h3><div className="evo-facts"><span>运行编号 <code>{detail.run_id}</code></span><span>模型 <b>{String(model.llm_model ?? model.model ?? model.model_name ?? "—")}</b></span><span>工作流 <b>{evolutionLabel(record(detail.run).workflow_mode)}</b></span><span>来源 <b>{evolutionLabel(record(detail.run).source_mode)}</b></span></div><RawRecord label="查看完成校验与资源记录" value={{completion_contract: pack.completion_contract, context_and_budget_metrics: pack.context_and_budget_metrics}} /></article>
      <article><h3>02 · 轨迹整理</h3><div className="evo-facts"><span>执行片段 <b>{Array.isArray(pack.episodes) ? episodes.length : "—"}</b></span><span>已加载技能 <b>{Array.isArray(pack.loaded_skills) ? pack.loaded_skills.length : "—"}</b></span><span>关联文件 <b>{Array.isArray(pack.artifact_refs) ? pack.artifact_refs.length : "—"}</b></span></div>{episodes.map((episode, index) => <p className="evo-explanation" key={index}>{String(episode.summary ?? "执行片段已记录")}</p>)}{Boolean(detail.diagnostics) && <div className="evo-explanation">{researchFailures.length ? researchFailures.map((failure, index) => <p key={index}><b>{evolutionLabel(failure.stage)} · {String(failure.event_type)}</b><br />{String(failure.message)}<br /><small>原始事件：{String(failure.event_id)}{failure.tool_call_id ? ` · 工具调用：${String(failure.tool_call_id)}` : ""}</small></p>) : <p>研究执行记录中未发现失败事件；后续复盘失败单独列示。</p>}</div>}<RawRecord value={pack} /></article>
      <article><h3>03 · 技能提案</h3><div className="evo-verdict"><StatusBadge value={proposal.decision ?? "pending"} /><p>{String(proposal.rationale ?? "等待生成提案")}</p></div><RawRecord label="查看提案与建议规则" value={proposal} /></article>
      <article><h3>04 · 独立审查</h3><p className="evo-explanation">{criticView.description}</p>{record(detail.critic_source).kind === "event" && <p className="evo-explanation">由对应历史审查事件恢复展示；未改写原记录。{record(detail.critic_source).partial === true ? "历史事件未保存的详细审查字段不作补填。" : ""}</p>}<div className="evo-verdict"><StatusBadge value={criticView.label} /><p>修订次数：{detail.revision_count} · 阻塞问题：{Array.isArray(critic.blocking_issues) ? critic.blocking_issues.length : "—"}</p></div>{Array.isArray(critic.blocking_issues) && critic.blocking_issues.length > 0 && <ul className="evo-issues">{critic.blocking_issues.map((issue, index) => <li key={index}>{String(issue)}</li>)}</ul>}<RawRecord label="查看审查评分与修改建议" value={critic} /></article>
      <article><h3>05 · 安全检查</h3><p className="evo-explanation">检查轨迹引用、工具权限、来源约束与停止条件。</p><div className="gate-list"><div><b>检查结果</b><span className={validation.passed === true ? "ok" : validation.passed === false ? "bad" : ""}>{checkLabel(validation.passed)}</span></div><div><b>阻塞问题</b><span>{checked ? errors.length : "—"}</span></div><div><b>提示信息</b><span>{checked ? warnings.length : "—"}</span></div></div>{errors.length > 0 && <ul className="evo-issues">{errors.map((error, index) => <li key={index}>{String(error)}</li>)}</ul>}<RawRecord label="查看安全检查记录" value={validation} /></article>
      <article><h3>06 · 候选技能</h3>{Object.keys(candidate).length ? <><div className="evo-facts"><span>状态 <b>{evolutionLabel(candidate.status)}</b></span><span>版本 <b>{String(candidate.version ?? "—")}</b></span><span>可用工具 <b>{Array.isArray(spec.allowed_tools) ? spec.allowed_tools.join(", ") : "—"}</b></span></div><details className="evo-raw"><summary>查看技能文档（SKILL.md）</summary><pre className="evo-skill-content">{content}</pre></details><RawRecord label="查看技能执行规则" value={candidatePayload.machine_policy ?? spec.machine_policy} /></> : <div className="evo-empty">尚未生成候选技能。</div>}</article>
      <article className="wide"><h3>07 · 技能启用前后对照评测</h3><EvaluationPanel detail={detail} /></article>
      <article className="wide"><h3>08 · 分阶段启用记录</h3>{detail.deployments?.length ? <div className="deployment-track">{detail.deployments.map((item, index) => <div key={String(item.deployment_id ?? index)} className={`deployment-node ${item.status}`}><strong>{evolutionLabel(item.stage)}</strong><span>流量比例 {formatPercent(typeof item.allocation_percent === "number" ? item.allocation_percent / 100 : undefined)}</span><small>{time(item.started_at)} → {item.stopped_at ? time(item.stopped_at) : "尚未结束"}</small><RawRecord label="查看阶段指标" value={item.metrics} /></div>)}</div> : <div className="evo-empty">尚未进入分阶段启用流程。</div>}</article>
      <article className="wide"><h3>复盘事件记录</h3><div className="evo-timeline">{detail.timeline?.length ? detail.timeline.map((item) => <details key={item.id}><summary><time>{time(item.created_at)}</time><b title={item.event_type}>{EVENT_NAMES[item.event_type] ?? "其他复盘事件"}</b><code title={item.stage ?? ""}>{evolutionLabel(item.stage)}</code></summary><RawRecord value={item.payload} /></details>) : <div className="evo-empty">这条历史复盘没有保存阶段事件；新研究任务会记录完整时间线。</div>}</div></article>
    </div>
  </section>;
}

export function EvolutionPage() {
  const client = useQueryClient(); const [status, setStatus] = useState(""); const [q, setQ] = useState(""); const [selected, setSelected] = useState<string | null>(null);
  const overview = useQuery({queryKey: ["evolution-overview"], queryFn: api.evolutionOverview, refetchInterval: 5000});
  const reviews = useQuery({queryKey: ["evolution-reviews", status, q], queryFn: () => api.evolutionReviews(status, q), refetchInterval: 3000});
  useEffect(() => { if (!selected && reviews.data?.items.length) setSelected(reviews.data.items[0].review_id); }, [reviews.data, selected]);
  const detail = useQuery({queryKey: ["evolution-review", selected], queryFn: () => api.evolutionReview(selected!), enabled: Boolean(selected), refetchInterval: (query) => ["completed", "rejected", "failed"].includes(String(query.state.data?.status)) ? false : 2000});
  const retry = useMutation({mutationFn: () => api.retryLearningReview(selected!), onSuccess: async () => { await Promise.all([client.invalidateQueries({queryKey: ["evolution-reviews"]}), client.invalidateQueries({queryKey: ["evolution-review", selected]})]); }});
  const funnelMax = useMemo(() => Math.max(1, ...(overview.data?.funnel.map((item) => item.count) ?? [1])), [overview.data]);
  return <main className="management-page evolution-page"><header><span className="eyebrow">从研究记录提炼可复用技能</span><h1>技能复盘与评测</h1><p>查看研究复盘、候选技能、对照评测和启用记录。候选技能经评测与人工批准后才能启用。</p></header>
    <ErrorState error={overview.error ?? reviews.error ?? detail.error ?? retry.error} />
    <section className="evo-metrics">{[
      ["复盘任务", overview.data?.reviews.total], ["候选技能", overview.data?.candidates.total],
      ["对照评测", overview.data?.evaluations.real_replay], ["灰度试用", overview.data?.deployments.canary],
      ["已启用技能", overview.data ? overview.data.versions.by_status.active ?? 0 : undefined], ["累计 Token 差值", overview.data?.impact.token_delta],
    ].map(([label, value]) => <div key={String(label)}><small>{label}</small><strong>{label === "累计 Token 差值" ? formatDelta(value) : formatCount(value)}</strong></div>)}</section>
    <p className="evo-explanation">累计 Token 差值汇总已记录的对照评测：启用候选技能减去未启用，正数表示用量增加，负数表示用量减少。</p>
    <section className="evo-funnel"><h2>流程阶段统计</h2><div>{overview.data?.funnel.map((item) => <span key={item.stage}><i style={{width: `${item.count / funnelMax * 100}%`}} /><b title={item.stage}>{evolutionLabel(item.stage)}</b><em>{item.count}</em></span>)}</div></section>
    <section className="evo-workbench"><aside className="evo-review-list"><header><h2>复盘任务</h2><span>{formatCount(reviews.data?.total)}</span></header><div className="evo-filters"><input aria-label="搜索复盘任务" value={q} onChange={(event) => setQ(event.target.value)} placeholder="搜索任务或技能" /><select aria-label="筛选复盘状态" value={status} onChange={(event) => setStatus(event.target.value)}><option value="">全部状态</option><option value="queued">等待处理</option><option value="completed">已完成</option><option value="rejected">已驳回</option><option value="failed">失败</option></select></div><div className="evo-review-scroll">{reviews.data?.items.map((item) => <button key={item.review_id} className={selected === item.review_id ? "selected" : ""} onClick={() => setSelected(item.review_id)}><div><strong title={item.skill_name ?? ""}>{item.goal || item.skill_name || "研究轨迹复盘"}</strong><StatusBadge value={item.status} /></div><p>{item.skill_name ? `技能：${item.skill_name}` : item.goal}</p><small>{evolutionLabel(item.source_mode)} · {evolutionLabel(item.workflow_mode)} · {time(item.updated_at)}</small><code title="复盘编号">{item.review_id}</code></button>)}{reviews.data && !reviews.data.items.length && <div className="evo-empty">暂无匹配的复盘任务。可以调整筛选条件，或先执行一次研究任务。</div>}</div></aside>
      <div className="evo-detail-host">{detail.data ? <Detail detail={detail.data} onRetry={() => retry.mutate()} retrying={retry.isPending} /> : <div className="evo-empty large">选择一条复盘任务，查看提案、审查和评测结果。</div>}</div>
    </section>
  </main>;
}
