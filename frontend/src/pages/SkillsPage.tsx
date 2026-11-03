import { MouseEvent, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../api/client";
import { ErrorState } from "../components/ErrorState";
import { evolutionLabel } from "./evolutionDisplay";

type SkillRef = {kind: "candidate"; id: string} | {kind: "version"; name: string; version: string};

export function SkillsPage() {
  const client = useQueryClient();
  const query = useQuery({queryKey: ["skills"], queryFn: api.skills});
  const [selected, setSelected] = useState<SkillRef | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [mutationError, setMutationError] = useState<unknown>(null);
  const [notice, setNotice] = useState("");
  const detail = useQuery({
    queryKey: ["skill-detail", selected],
    queryFn: () => {
      if (!selected) throw new Error("未选择 Skill");
      return selected.kind === "candidate"
        ? api.skillCandidate(selected.id)
        : api.skill(selected.name, selected.version);
    },
    enabled: Boolean(selected),
    refetchInterval: selected?.kind === "candidate" ? 3000 : false,
  });
  const reviewId = String((detail.data?.review as Record<string, unknown> | undefined)?.review_id ?? "");
  const evaluationProgress = useQuery({
    queryKey: ["skill-evaluation-progress", reviewId],
    queryFn: () => api.evolutionReview(reviewId),
    enabled: Boolean(reviewId) && selected?.kind === "candidate",
    refetchInterval: 3000,
  });
  const evaluationEvents = (evaluationProgress.data?.timeline ?? []).filter(item =>
    item.event_type.startsWith("skill.evaluation.") && item.payload.candidate_id === (selected?.kind === "candidate" ? selected.id : ""));
  const startIndex = evaluationEvents.map(item => item.event_type).lastIndexOf("skill.evaluation.started");
  const currentEvents = startIndex >= 0 ? evaluationEvents.slice(startIndex) : [];
  const lastProgress = currentEvents[currentEvents.length - 1];
  const completedArms = currentEvents.filter(item => item.event_type === "skill.evaluation.arm_completed").length;
  const caseCount = Array.isArray((detail.data?.payload as Record<string, unknown> | undefined)?.eval_cases)
    ? ((detail.data?.payload as Record<string, unknown>).eval_cases as unknown[]).length : 0;

  async function refresh() {
    await Promise.all([
      client.invalidateQueries({queryKey: ["skills"]}),
      client.invalidateQueries({queryKey: ["skill-detail"]}),
    ]);
  }
  async function mutate(key: string, action: () => Promise<Record<string, unknown>>, success: string, event?: MouseEvent) {
    event?.stopPropagation(); setBusy(key); setMutationError(null); setNotice(success === "Skill 评测完成" ? "正在运行对照评测，每题执行两组，可能需要数分钟；详情会更新进度，请勿重复提交。" : "");
    try { const result = await action(); setNotice(result.status === "failed" ? "评测未通过，请查看对照结果和失败检查项。" : success); await refresh(); }
    catch (error) { setMutationError(error); }
    finally { setBusy(null); }
  }
  async function promoteCandidate(id: string, name: string, version: string, event: MouseEvent) {
    event.stopPropagation();
    const expected = `PROMOTE ${name}@${version}`;
    const confirmation = window.prompt(`人工启用会改变后续 Agent 行为。请输入：${expected}`) ?? "";
    if (confirmation !== expected) { setMutationError(new Error("确认短语不匹配，已取消启用")); return; }
    await mutate(`candidate:${id}`, () => api.promoteSkillCandidate(id, confirmation), "Skill 已进入 Shadow", event);
  }
  async function advanceVersion(name: string, version: string, targetStage: "canary" | "active", event: MouseEvent) {
    event.stopPropagation(); const expected = `ADVANCE ${name}@${version} TO ${targetStage}`;
    const confirmation = window.prompt(`请输入部署确认短语：${expected}`) ?? "";
    if (confirmation !== expected) { setMutationError(new Error("确认短语不匹配，已取消部署")); return; }
    await mutate(`version:${name}`, () => api.deploySkill(name, version, targetStage, confirmation), `Skill 已进入 ${targetStage}`, event);
  }
  async function suspendVersion(name: string, version: string, event: MouseEvent) {
    event.stopPropagation(); const expected = `SUSPEND ${name}@${version}`;
    const confirmation = window.prompt(`请输入暂停确认短语：${expected}`) ?? "";
    if (confirmation !== expected) { setMutationError(new Error("确认短语不匹配，已取消暂停")); return; }
    const reason = window.prompt("请输入暂停原因")?.trim() || "人工暂停";
    await mutate(`version:${name}`, () => api.suspendSkill(name, version, reason, confirmation), "Skill 已暂停", event);
  }

  return <main className="management-page skills-page"><header><span className="eyebrow">管理可复用的研究策略</span><h1>研究技能</h1><p>查看候选技能与版本详情。候选通过评测后，由你确认是否启用。</p></header>
    <ErrorState error={query.error ?? mutationError} />{notice && <div className="skill-notice">{notice}</div>}
    <section className="registry-grid"><div><h2>候选</h2>{query.data?.candidates.length ? query.data.candidates.map((item) => {
      const id = String(item.candidate_id); const candidateStatus = String(item.status); const key = `candidate:${id}`;
      return <article className="registry-card clickable" key={id} onClick={() => setSelected({kind: "candidate", id})}>
        <strong>{String(item.name)} · {String(item.proposed_version)}</strong><span>{evolutionLabel(candidateStatus)}</span><small>来源研究：{String(item.run_id)}</small>
        <div className="row-actions"><button disabled={busy != null} onClick={(event) => { setSelected({kind: "candidate", id}); setNotice("已请求对照评测：每题分别运行不加载和加载 Skill 两组，可能需要数分钟。详情中会更新进度。"); return mutate(key, () => api.evaluateSkillCandidate(id), "Skill 评测完成", event); }}>{busy === key ? "评测中…" : "评测"}</button><button disabled={candidateStatus !== "evaluated" || busy != null} onClick={(event) => promoteCandidate(id, String(item.name), String(item.proposed_version), event)}>{busy === key ? "处理中…" : "启用"}</button></div>
      </article>;
    }) : <div className="empty-state">没有候选 Skill</div>}</div><div><h2>版本</h2>{query.data?.versions.length ? query.data.versions.map((item) => {
      const name = String(item.name); const version = String(item.version); const key = `version:${name}`;
      return <article className="registry-card clickable" key={String(item.skill_version_id)} onClick={() => setSelected({kind: "version", name, version})}>
        <strong>{name} · {version}</strong><span>{evolutionLabel(item.status)}</span><div className="row-actions">
          {item.status === "shadow" ? <button disabled={busy != null} onClick={(event) => advanceVersion(name, version, "canary", event)}>小范围试用</button> : null}
          {item.status === "canary" ? <button disabled={busy != null} onClick={(event) => advanceVersion(name, version, "active", event)}>正式启用</button> : null}
          {item.status === "active" ? <button disabled={busy != null} onClick={(event) => mutate(key, () => api.rollbackSkill(name), "Skill 已回滚", event)}>{busy === key ? "回滚中…" : "回滚"}</button> : null}
          {["shadow", "canary", "active"].includes(String(item.status)) ? <button disabled={busy != null} onClick={(event) => suspendVersion(name, version, event)}>暂停</button> : null}
        </div>
      </article>;
    }) : <div className="empty-state">没有已注册版本</div>}</div></section>
    {selected && <aside className="skill-drawer" aria-label="Skill 详情"><header><div><span className="eyebrow">技能详情</span><h2>{String(detail.data?.name ?? "Skill 详情")} · {String(detail.data?.version ?? "")}</h2></div><button className="icon-button" onClick={() => setSelected(null)} aria-label="关闭 Skill 详情">×</button></header>
      <div className="skill-drawer-body">{detail.isLoading && <div className="drawer-empty">正在加载 Skill…</div>}<ErrorState error={detail.error} />{detail.data && <>
        <div className="skill-detail-meta"><span>{evolutionLabel(detail.data.status)}</span><code>{String(detail.data.candidate_id ?? detail.data.skill_version_id ?? "")}</code></div>
        {lastProgress && <section aria-live="polite"><h3>对照评测进度</h3><p>已完成 {completedArms} / {caseCount * 2 || "—"} 组执行</p><p>{lastProgress.event_type.endsWith("evaluation_completed") ? "评测已结束，结果见下方。" : lastProgress.event_type.endsWith("case_started") ? `正在执行第 ${Number(lastProgress.payload.case_index) + 1} 题 · ${lastProgress.payload.arm === "control" ? "不加载 Skill" : "加载 Skill"}` : "正在处理评测结果…"}</p><p>最近记录：{new Date(lastProgress.created_at).toLocaleString()}。进度来自后台事件；长时间不更新时请检查服务，避免重复提交。</p></section>}
        <ErrorState error={evaluationProgress.error} />
        <pre>{String((detail.data.payload as Record<string, unknown> | undefined)?.content ?? detail.data.content ?? "")}</pre>
        {detail.data.evaluation ? <section><h3>最近评测</h3><pre>{JSON.stringify(detail.data.evaluation, null, 2)}</pre></section> : <div className="context-notice">尚无评测结果。</div>}
        {detail.data.review ? <section><h3>学习复盘</h3><pre>{JSON.stringify(detail.data.review, null, 2)}</pre></section> : null}
      </>}</div>
    </aside>}
  </main>;
}
