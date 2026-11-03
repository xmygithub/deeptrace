/** Presentation-only labels; API values and saved records remain unchanged. */
const LABELS: Record<string, string> = {
  queued: "等待处理", pending: "待处理", running: "处理中", building_pack: "整理轨迹中",
  proposing: "生成提案中", criticizing: "审查中", revising: "修订提案中",
  writing_candidate: "生成候选技能中", retry_wait: "等待重试", evaluating: "评测中",
  completed: "已完成", failed: "失败", rejected: "已驳回", cancelled: "已取消",
  passed: "通过", pass: "通过", reject: "驳回", revise: "需要修订",
  create: "新建技能", patch: "更新技能", ignore: "无需生成技能",
  candidate: "候选技能", evaluated: "评测通过", promoted: "已批准",
  shadow: "影子验证", canary: "灰度试用", active: "已启用", suspended: "已暂停",
  previous: "上一版本", archived: "已归档", inactive: "未启用",
  web: "联网搜索", graphrag: "私域知识库",
  deep_research: "深度研究", plan_execute_report: "计划执行报告",
  run: "研究记录", review_pack: "轨迹整理", proposal: "生成提案", critic: "独立审查",
  validation: "安全检查", evaluation: "对照评测",
  deployment: "分阶段启用", learning: "技能复盘",
  Run: "研究记录", "Run Review": "研究复盘", Proposed: "已生成提案",
  "Critic Passed": "审查通过", Candidate: "候选技能", Evaluated: "评测通过", Active: "已启用",
  eval: "评测集", holdout: "保留测试集", train: "开发集", regression: "回归测试集",
};

export function evolutionLabel(value: unknown): string {
  if (value === null || value === undefined || value === "") return "未记录";
  return LABELS[String(value)] ?? String(value);
}

function isNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

export function formatCount(value: unknown): string {
  return isNumber(value) ? value.toLocaleString("zh-CN", { maximumFractionDigits: 0 }) : "—";
}

export function formatPercent(value: unknown): string {
  return isNumber(value) ? `${(value * 100).toLocaleString("zh-CN", { maximumFractionDigits: 1 })}%` : "未记录";
}

export function formatDelta(value: unknown, unit = ""): string {
  if (!isNumber(value)) return "未记录";
  return `${value > 0 ? "+" : ""}${value.toLocaleString("zh-CN", { maximumFractionDigits: unit ? 2 : 0 })}${unit}`;
}

export function formatSeconds(value: unknown): string {
  return isNumber(value) ? `${value.toLocaleString("zh-CN", { maximumFractionDigits: 2 })} 秒` : "未记录";
}

export function checkLabel(value: unknown): string {
  return value === true ? "通过" : value === false ? "未通过" : "未检查";
}

export function holdoutLabel(metrics: Record<string, unknown>): string {
  const pairs = Array.isArray(metrics.pairs) ? metrics.pairs : [];
  const hasHoldout = pairs.some((pair) => pair && typeof pair === "object"
    && (pair as { case?: { split?: string } }).case?.split === "holdout");
  if (!hasHoldout) return "未评测";
  return metrics.holdout_passed === true ? "通过" : metrics.holdout_passed === false ? "未通过" : "未记录";
}

export function criticPresentation(detail: {critic_state?: unknown; proposal?: Record<string, unknown>; critic?: Record<string, unknown>}) {
  const fallback = detail.proposal?.decision === "ignore"
    ? detail.critic?.decision ? "reviewed_then_ignored" : "skipped"
    : ({pass: "passed", revise: "needs_revision", reject: "rejected"} as Record<string, string>)[String(detail.critic?.decision)] ?? "not_started";
  const state = String(detail.critic_state ?? fallback);
  const views: Record<string, {stage: string; label: string; description: string}> = {
    passed: {stage: "done", label: "通过", description: "独立审查已通过"},
    rejected: {stage: "blocked", label: "驳回", description: "提案未通过独立审查"},
    needs_revision: {stage: "blocked", label: "需要修订", description: "审查要求修改提案"},
    running: {stage: "active", label: "审查中", description: "正在执行独立审查"},
    skipped: {stage: "skipped", label: "无需审查", description: "提案阶段判断无需沉淀技能，未进入审查"},
    reviewed_then_ignored: {stage: "done", label: "已审查，提案已撤回", description: "根据审查意见修订后，决定不沉淀技能"},
    not_started: {stage: "pending", label: "尚未开始", description: "等待技能提案生成后进入审查"},
    not_reached: {stage: "skipped", label: "未进入审查", description: "前序阶段失败，未执行独立审查"},
    failed: {stage: "blocked", label: "审查异常", description: "审查执行失败，请查看错误详情"},
    unavailable: {stage: "pending", label: "审查记录缺失", description: "没有足够记录确认结果，请勿据此判断通过"},
  };
  return views[state] ?? views.unavailable;
}
