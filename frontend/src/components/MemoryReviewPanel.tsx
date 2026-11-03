import { useQuery } from "@tanstack/react-query";
import { api } from "../api/client";
import { ErrorState } from "./ErrorState";

const STATUS: Record<string, string> = {candidates: "已生成候选", no_candidates: "未发现合格候选", skipped: "未进行模型提取", failed: "提取未完成"};
const REASONS: Record<string, string> = {
  reviewed: "已检查用户原话", no_durable_signal: "未发现明确的长期偏好或项目陈述", model_not_configured: "记忆提取未启用或未配置模型",
  evaluation: "评测运行不写记忆", missing_run: "运行不存在", missing_message: "用户消息不存在",
  TimeoutError: "超过记忆提取时限", BudgetExceeded: "达到记忆提取独立预算", ValueError: "模型返回内容无法解析",
};
const FILTERS: Record<string, string> = {
  noop: "模型判定无需变化", duplicate: "与已有记忆重复", invalid_structure: "结构不符合要求", low_confidence: "模型自评分未达门槛",
  missing_quote: "缺少用户原话依据", ineligible_source: "假设、临时要求或缺少实际归属", invalid_kind: "记忆类型不匹配",
  missing_update_target: "未指定待替代条目", policy_rejected: "不符合记忆保存规则",
};

export function MemoryReviewPanel() {
  const query = useQuery({queryKey: ["memory-reviews"], queryFn: api.memoryReviews, refetchInterval: 5000});
  return <details className="context-group memory-review-panel">
    <summary><strong>最近的记忆提取记录</strong><span>{query.data?.total ?? 0} 次</span></summary>
    <p className="context-notice">研究成功与否不决定用户事实能否提取。候选仍需人工确认；普通研究问题通常不会产生长期记忆。</p>
    <ErrorState error={query.error} />
    {!query.isPending && !query.data?.items.length && <p className="drawer-empty compact">暂无新版提取记录。更新后发送新问题才会生成记录，旧运行不会自动补写。</p>}
    {query.data?.items.map((item) => <article className="memory-review-row" key={`${item.run_id}:${item.created_at}`}>
      <div><strong>{STATUS[item.status] ?? item.status}</strong><span>{new Date(item.created_at).toLocaleString()}</span></div>
      <p>{REASONS[item.reason] ?? `调用未完成（${item.reason}）`} · 新候选 {item.candidate_count ?? 0} 条 · 模型调用 {item.model_calls ?? 0} 次</p>
      {item.partial && <p className="danger">仅完成部分分段检查，未覆盖的内容不能视为已提取。</p>}
      {Boolean(Object.keys(item.filtered ?? {}).length) && <p>过滤情况：{Object.entries(item.filtered ?? {}).map(([key, count]) => `${FILTERS[key] ?? key} ${count}`).join("；")}</p>}
      <small>运行 <code>{item.run_id}</code> · 独立预算占用 {item.budget_charged_tokens ?? 0} / {item.budget_limit_tokens ?? 0} tokens（含未知用量的保守记账，非账单）</small>
    </article>)}
  </details>;
}
