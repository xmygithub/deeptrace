import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import ts from "typescript";

const source = readFileSync(new URL("../src/components/verificationDetails.ts", import.meta.url), "utf8");
const { outputText } = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 } });
const { verificationDetail, verificationStatus } = await import(`data:text/javascript;base64,${Buffer.from(outputText).toString("base64")}`);
const check = (kind, actual, expected, passed = true) => ({kind, required: true, passed, evidence: {observed: {actual, expected}, explanation: "保存的判定说明"}});

test("delivery coverage keeps all actual report quotes, unmet requirements and request text", () => {
  const data = {requirements: [{id: "req1", text: "比较两者", request_quote: "比较"}, {id: "req2", text: "给建议"}],
    items: [{requirement_id: "req1", status: "covered", reason: "已比较", report_quotes: [{text: "实际段落A"}, {text: "实际段落B"}]},
      {requirement_id: "req2", status: "partial", reason: "只有原则，没有操作建议"}]};
  const result = verificationDetail({kind: "delivery_coverage", evidence: {observed: data}});
  assert.deepEqual(result.entries[0].quotes, ["实际段落A", "实际段落B"]);
  assert.equal(result.entries[0].requestQuote, "比较");
  assert.equal(result.entries[1].status, "部分满足");
  assert.equal(result.entries[1].text, "只有原则，没有操作建议");
});

test("citation failure points to missing IDs and the actual uncited paragraph", () => {
  const result = verificationDetail(check("citation_integrity", {citations: ["ev_a", "ev_a", "ev_missing"], missing: ["ev_missing"], uncited_claims: ["没有引用的结论段落"], unresolved_citations: ["result_bad"]}));
  assert.equal(result.metrics[0].value, "3");
  assert.equal(result.entries.find(item => item.label === "ev_missing").status, "ID 不存在");
  assert.ok(result.entries.some(item => item.text === "没有引用的结论段落"));
});

test("claim details show checked content; old aggregate-only records do not invent paragraphs", () => {
  const old = verificationDetail(check("claim_support", {claims: 4, supported: 3}));
  assert.equal(old.entries.length, 0);
  assert.match(old.emptyMessage, /没有保存逐段明细/);
  const current = verificationDetail(check("claim_support", {claims: 1, supported: 0, claim_details: [{text: "待补引用的段落", passed: false, evidence_ids: []}]}));
  assert.equal(current.entries[0].text, "待补引用的段落");
  assert.equal(current.entries[0].status, "缺少有效引用");
});

test("each deterministic check exposes its recorded conditions and observations", () => {
  assert.deepEqual(verificationDetail(check("min_evidence", 2, 3)).metrics.map(item => item.value), ["3", "2"]);
  const mode = verificationDetail(check("source_match", ["web", "graphrag"], "web"));
  assert.equal(mode.metrics[2].value, "1");
  assert.equal(mode.entries[1].status, "来源不匹配");
  const sections = verificationDetail(check("required_section", {headings: ["比较"], missing: ["验证方案"]}, ["验证方案"]));
  assert.equal(sections.entries[1].text, "验证方案");
  assert.equal(sections.entries[1].status, "未满足");
  assert.equal(verificationDetail(check("source_diversity", {sources: ["example.org"], limitation_disclosed: true})).metrics[0].value, "1");
  assert.match(verificationDetail(check("report_consistency", false)).metrics[0].value, /未通过/);
  const coverage = verificationDetail(check("evidence_card_coverage", {ledger_count: 2, card_count: 1, missing_by_stage: {cards: ["ev_b"]}}));
  assert.deepEqual(coverage.entries[0].evidenceIds, ["ev_b"]);
  assert.equal(coverage.entries[0].label, "证据卡缺失");
});

test("unknown and incomplete historical records keep unknown values and status", () => {
  for (const kind of ["min_evidence", "claim_support", "citation_integrity", "delivery_coverage", "source_diversity", "source_match"]) {
    const result = verificationDetail({kind, evidence: {}});
    assert.ok(result.metrics.every(metric => metric.value === "未记录"), kind);
    assert.equal(result.entries.length, 0);
  }
  assert.match(verificationStatus({passed: null, required: true}), /待判定/);
  assert.match(verificationStatus({passed: false, required: false}), /未通过.*非阻塞/);
  assert.match(verificationDetail(check("future_kind", {some: "raw"})).emptyMessage, /原始记录/);
});

test("stale coverage record is available only as historical raw data", () => {
  const result = verificationDetail({kind: "delivery_coverage", passed: null, evidence: {report_binding_status: "unverified", observed: {
    items: [{requirement_id: "old", status: "covered", report_quote: "old report"}]}}});
  assert.equal(result.entries.length, 0);
  assert.match(result.explanation, /未能与当前报告版本匹配/);
});

test("checker errors are distinct from inconsistent content and expose the checked revision", () => {
  const item = check("report_consistency", null, true, false);
  item.evidence.observed.details = {status: "error", report_hash: "abc123", method: "llm-assisted", error: "JSON 协议错误"};
  assert.match(verificationStatus(item), /检查异常/);
  const detail = verificationDetail(item);
  assert.equal(detail.metrics[0].value, "检查器执行异常");
  assert.equal(detail.metrics[2].value, "abc123");
  assert.equal(detail.entries[0].text, "JSON 协议错误");
  item.evidence.observed.details = {status: "failed", issues: [{location: "p2", description: "正文与来源矛盾"}]};
  assert.match(verificationStatus(item), /未通过/);
  assert.equal(verificationDetail(item).entries[0].text, "正文与来源矛盾");
});

test("semantic warnings stay visible without becoming blocking issues", () => {
  const item = check("report_consistency", true, true);
  item.evidence.observed.details = {status: "passed", policy_version: "semantic-review-v2", issues: [], warnings: [
    {kind: "editorial", report_quote: "原报告中的建议段落", description: "建议精简重复表达"},
    {kind: "insufficient_context", report_quote: "尚不能充分核验的结论", description: "当前来源片段被截断"}]};
  assert.match(verificationStatus(item), /通过（有提示）/);
  const detail = verificationDetail(item);
  assert.equal(detail.entries.length, 2);
  assert.equal(detail.entries[0].status, "提示 · 不自动重写");
  assert.equal(detail.entries[1].label, "当前材料不足以核验");
  assert.deepEqual(detail.entries[1].quotes, ["尚不能充分核验的结论"]);
  assert.match(detail.boundary, /不等于.*已经核实正确/);
});
