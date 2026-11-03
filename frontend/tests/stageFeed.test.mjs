import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import ts from "typescript";

// Test the existing pure TypeScript reducer without adding a test dependency.
const source = readFileSync(new URL("../src/components/stageFeed.ts", import.meta.url), "utf8");
const { outputText } = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 } });
const { deriveStageFeed } = await import(`data:text/javascript;base64,${Buffer.from(outputText).toString("base64")}`);
const event = (event_type, payload) => ({ run_id: "run1", event_type, ...payload });

const displaySource = readFileSync(new URL("../src/pages/evolutionDisplay.ts", import.meta.url), "utf8");
const displayModule = ts.transpileModule(displaySource, { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 } });
const { criticPresentation } = await import(`data:text/javascript;base64,${Buffer.from(displayModule.outputText).toString("base64")}`);

test("critic distinguishes skipped, pending, missing records and completed review", () => {
  assert.equal(criticPresentation({critic_state: "passed"}).stage, "done");
  assert.equal(criticPresentation({critic_state: "skipped"}).label, "无需审查");
  assert.equal(criticPresentation({critic_state: "not_started"}).label, "尚未开始");
  assert.equal(criticPresentation({critic_state: "unavailable"}).label, "审查记录缺失");
  assert.equal(criticPresentation({critic_state: "reviewed_then_ignored"}).stage, "done");
  assert.equal(criticPresentation({critic: {decision: "pass"}}).stage, "done");
});

test("merge a richer Worker completion after a partial tool event", () => {
  const feed = deriveStageFeed([
    event("tool.completed", { tool_call_id: "c1", result_count: 5 }),
    event("tool.completed", { tool_call_id: "c1", tool_name: "web_search", query: "RAG", output_preview: "result" }),
  ]);
  assert.equal(feed.tools.length, 1);
  assert.equal(feed.tools[0].tool_name, "web_search");
  assert.equal(feed.tools[0].result_count, 5);
  assert.equal(feed.tools[0].query, "RAG");
});

test("keep actual tool identity and query across start and completion", () => {
  const feed = deriveStageFeed([
    event("tool.started", { tool_call_id: "c1", tool_name: "tavily_search", query: "RAG" }),
    event("tool.completed", { tool_call_id: "c1", result_count: 5 }),
    event("tool.completed", { tool_call_id: "c1", tool_name: "web_search" }),
  ]);
  assert.equal(feed.tools.length, 1);
  assert.equal(feed.tools[0].tool_name, "tavily_search");
  assert.equal(feed.tools[0].query, "RAG");
  assert.equal(feed.tools[0].running, false);
});

test("show running and failed calls with terminal status", () => {
  const started = event("tool.started", { tool_call_id: "c1", tool_name: "tavily_search" });
  assert.equal(deriveStageFeed([started]).tools[0].running, true);
  const feed = deriveStageFeed([started, event("tool.failed", { tool_call_id: "c1" })]);
  assert.equal(feed.tools[0].failed, true);
  assert.equal(feed.tools[0].running, false);
});

test("do not guess the tool name when historical metadata is missing", () => {
  const feed = deriveStageFeed([event("tool.completed", { tool_call_id: "c1" })]);
  assert.equal(feed.tools[0].tool_name, "工具名称未记录");
});

test("normalize historical one-based rounds and preserve new zero-based rounds", () => {
  for (const base of [0, 1]) {
    const feed = deriveStageFeed([
      event("iteration.completed", { kind: "iteration", iteration_index: base, queries: [], total_results: 5 }),
      event("iteration.completed", { kind: "iteration", iteration_index: base + 1, queries: [], total_results: 5 }),
    ]);
    assert.deepEqual(feed.iterations.map((row) => row.index), [0, 1]);
  }
});
