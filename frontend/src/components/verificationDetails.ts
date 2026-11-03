import type { Report } from "../types/api";

type Check = Report["verification"][number];
export interface DetailEntry {
  label: string; status?: string; text?: string; quotes?: string[]; evidenceIds?: string[]; requestQuote?: string;
}
export interface VerificationDetail {
  explanation: string; boundary?: string; expected?: string;
  metrics: Array<{label: string; value: string}>; entries: DetailEntry[]; emptyMessage?: string;
}
const object = (value: unknown): Record<string, unknown> => value !== null && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {};
const list = (value: unknown): unknown[] => Array.isArray(value) ? value : [];
const strings = (value: unknown): string[] => list(value).filter((item): item is string => typeof item === "string");
const value = (item: unknown): string => item === undefined || item === null ? "未记录" : typeof item === "object" ? JSON.stringify(item) : String(item);
export const COVERAGE_STATUS: Record<string, string> = {covered: "已满足", partial: "部分满足", missing: "未满足", insufficient_evidence: "基础证据不足"};
export const verificationStatus = (check: Check): string => {
  const details = object(object(object(check.evidence).observed).details);
  const status = details.status;
  const label = check.kind === "report_consistency" && status === "error" ? "检查异常" : check.kind === "report_consistency" && status === "not_checked" ? "待重新检查" : check.passed === true ? check.kind === "report_consistency" && list(details.warnings).length ? "通过（有提示）" : "通过" : check.passed === false ? "未通过" : "待判定";
  return `${label}${check.required ? " · 必需检查" : " · 非阻塞参考"}`;
};

export function verificationDetail(check: Check, report?: Report | null): VerificationDetail {
  const root = object(check.evidence);
  const observed = object(root.observed);
  const actual = "actual" in observed ? observed.actual : root.observed;
  const record = object(actual);
  const detail: VerificationDetail = {explanation: typeof root.explanation === "string" ? root.explanation : "历史记录未保存判定说明。",
    expected: observed.expected === undefined ? undefined : value(observed.expected), metrics: [], entries: []};
  const metric = (label: string, item: unknown) => detail.metrics.push({label, value: value(item)});
  const entries = (items: unknown, label: string, status: string) => strings(items).forEach((text, index) => detail.entries.push({label: `${label} ${index + 1}`, status, text}));

  switch (check.kind) {
    case "delivery_coverage": {
      if (root.report_binding_status === "unverified") {
        detail.explanation = "这份覆盖记录未能与当前报告版本匹配，不能作为当前报告已满足要求的依据。";
        detail.emptyMessage = "可展开原始记录查看历史判定；需重新审查当前报告才能得到有效结论。";
        break;
      }
      const review = Array.isArray(record.items) ? record : object(report?.delivery_review);
      const requirements = list(review.requirements).map(object);
      detail.boundary = "这是模型辅助的交付覆盖审查，依据定位到报告正文；不等于事实准确率或来源权威性验证。";
      list(review.items).map(object).forEach(item => {
        const requirement = requirements.find(req => req.id === item.requirement_id);
        const quotes = list(item.report_quotes).map(object).map(quote => quote.text).filter((text): text is string => typeof text === "string" && Boolean(text));
        detail.entries.push({label: typeof requirement?.text === "string" ? requirement.text : value(item.requirement_id),
          status: COVERAGE_STATUS[value(item.status)] ?? value(item.status), text: value(item.reason),
          quotes: quotes.length ? quotes : typeof item.report_quote === "string" && item.report_quote ? [item.report_quote] : [],
          requestQuote: typeof requirement?.request_quote === "string" ? requirement.request_quote : undefined});
      });
      metric("已满足", Array.isArray(review.items) ? review.items.filter(item => object(item).status === "covered").length : undefined);
      metric("检查项", Array.isArray(review.items) ? review.items.length : undefined);
      list(object(review.reconciliation).removed_limitations).map(object).forEach(item => detail.entries.push({
        label: "已校正的历史缺口表述", status: "正文已回答", text: `${value(item.text)}\n判定理由：${value(item.reason)}`,
        quotes: list(item.report_quotes).map(object).map(quote => quote.text).filter((text): text is string => typeof text === "string")
      }));
      detail.emptyMessage = "这条历史记录没有保存逐项交付依据。";
      break;
    }
    case "citation_integrity": {
      const missing = new Set(strings(record.missing));
      const ids = [...new Set(strings(record.citations))];
      metric("正文引用次数", Array.isArray(record.citations) ? record.citations.length : undefined);
      metric("引用的不同证据", Array.isArray(record.citations) ? ids.length : undefined);
      metric("未知证据 ID", Array.isArray(record.missing) ? missing.size : undefined);
      ids.forEach(id => detail.entries.push({label: id, status: missing.has(id) ? "ID 不存在" : "ID 有效", evidenceIds: [id]}));
      entries(record.uncited_claims, "未引用的内容", "未满足");
      entries(record.unresolved_citations, "未解析的引用", "未满足");
      detail.boundary = "ID 有效与逐段有引用，不代表引用来源在语义上支持该段结论。";
      detail.emptyMessage = "该记录没有保存引用或未满足项的明细，请核对原始记录。";
      break;
    }
    case "claim_support":
      metric("检查的主张段落", record.claims); metric("带有效引用的段落", record.supported);
      detail.boundary = "本项实际检查主张的引用覆盖，尚未进行逐主张的语义事实核验。";
      list(record.claim_details).map(object).forEach(item => detail.entries.push({label: "被检查的正文", status: item.passed === true ? "引用覆盖满足" : "缺少有效引用",
        text: `${value(item.text)}${item.truncated ? "（记录已截断）" : ""}`, evidenceIds: strings(item.evidence_ids)}));
      detail.emptyMessage = "此历史记录只保存了统计数，没有保存逐段明细。";
      break;
    case "min_evidence":
      metric("要求至少", observed.expected); metric("实际台账记录", actual);
      detail.boundary = "这是证据数量检查，不代表每条资料都相关、可靠或充分支撑结论。";
      break;
    case "source_match": {
      const modes = strings(actual);
      metric("冻结的信息源", observed.expected); metric("已检查证据数", Array.isArray(actual) ? modes.length : undefined);
      metric("来源不匹配数", Array.isArray(actual) && observed.expected !== undefined ? modes.filter(mode => mode !== observed.expected).length : undefined);
      [...new Set(modes)].forEach(mode => detail.entries.push({label: mode, text: `${modes.filter(item => item === mode).length} 条证据`, status: mode === observed.expected ? "来源匹配" : "来源不匹配"}));
      break;
    }
    case "required_section":
      entries(record.headings, "已识别章节", "已存在"); entries(record.missing, "缺失的必需章节", "未满足");
      detail.boundary = "本项定位章节与格式；论述是否充分需结合最终交付覆盖检查。";
      break;
    case "source_diversity":
      metric("记录的来源标识数", Array.isArray(record.sources) ? record.sources.length : undefined); metric("是否披露来源局限", typeof record.limitation_disclosed === "boolean" ? record.limitation_disclosed ? "是" : "否" : undefined);
      entries(record.sources, "来源标识", "已记录");
      detail.boundary = "本项记录来源标识的多样性，不验证信息独立性或权威性。";
      break;
    case "report_consistency": {
      const saved = object(observed.details);
      const statuses: Record<string, string> = {passed: "检查通过", failed: "内容检查未通过", error: "检查器执行异常", not_checked: "当前版本尚未检查"};
      metric("检查状态", typeof saved.status === "string" ? statuses[saved.status] ?? saved.status : typeof actual === "boolean" ? actual ? "未报告不一致" : "检查未通过（历史记录未区分原因）" : actual);
      if (saved.method) metric("检查方式", ({deterministic: "确定性检查", reporter: "报告阶段检查", "llm-assisted": "模型辅助检查"} as Record<string, string>)[value(saved.method)] ?? value(saved.method));
      if (saved.report_hash) metric("检查的报告版本", saved.report_hash);
      if (saved.policy_version) {
        metric("审查策略", saved.policy_version);
        metric("阻断问题", list(saved.issues).length);
        metric("提示与未核验项", list(saved.warnings).length);
      }
      if (saved.error) detail.entries.push({label: "检查异常原因", status: "尚无有效内容结论", text: value(saved.error)});
      list(saved.issues).map(object).forEach(item => detail.entries.push({label: value(item.location ?? item.type ?? item.kind ?? "检查问题"), text: value(item.description ?? item.message ?? item), status: "阻断问题", quotes: [...(typeof item.report_quote === "string" ? [item.report_quote] : []), ...list(item.source_quotes).map(object).map(q => `来源 ${value(q.evidence_id)}：${value(q.text)}`)], evidenceIds: list(item.source_quotes).map(object).map(q => value(q.evidence_id))}));
      list(saved.warnings).map(object).forEach(item => detail.entries.push({label: item.kind === "insufficient_context" ? "当前材料不足以核验" : "改进建议", text: value(item.description), status: "提示 · 不自动重写", quotes: typeof item.report_quote === "string" ? [item.report_quote] : []}));
      list(saved.corrections).map(object).forEach(item => detail.entries.push({label: "检查器建议（未自动采用）", text: `${value(item.original)}\n建议：${value(item.corrected)}\n理由：${value(item.reason)}`}));
      detail.boundary = "实质错误与提示分开记录；材料不足不等于事实错误或已经核实正确。历史记录仍按当时方法解释，DeepResearch可能仅执行确定性检查。";
      break;
    }
    case "evidence_card_coverage": {
      const labels: Record<string, string> = {ledger_count: "台账", card_count: "证据卡", routed_count: "已路由", processed_count: "已处理", annex_count: "结构化索引"};
      Object.entries(labels).forEach(([key, label]) => metric(label, record[key]));
      const stages: Record<string, string> = {cards: "证据卡缺失", routed: "路由缺失", processed: "处理缺失", annex: "索引缺失"};
      Object.entries(object(record.missing_by_stage)).forEach(([stage, ids]) => strings(ids).forEach(id => detail.entries.push({label: stages[stage] ?? stage, status: "未满足", evidenceIds: [id]})));
      detail.emptyMessage = "没有记录到分阶段缺失 ID；可在下方原始记录核对统计。";
      detail.boundary = "全量覆盖指处理链路中的证据 ID 集合一致，不代表所有证据都必须出现在正文。";
      break;
    }
    case "research_coverage":
      entries(record.open_questions, "研究阶段记录的事实缺口", "历史记录");
      detail.boundary = "这些缺口产生于检索阶段，最终是否解决请查看最终交付覆盖检查。";
      break;
    default:
      if (actual !== undefined) metric("记录的观测结果", actual);
      detail.emptyMessage = "该检查没有结构化展示模板，请展开原始记录查看已有依据。";
  }
  return detail;
}
