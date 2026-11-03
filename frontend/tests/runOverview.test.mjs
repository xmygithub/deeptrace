import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import ts from "typescript";

// Render the real TSX components without a browser or a new test dependency.
const require = createRequire(import.meta.url);
const modules = new Map();
function loadSource(path) {
  if (modules.has(path)) return modules.get(path).exports;
  const module = {exports: {}};
  modules.set(path, module);
  const {outputText} = ts.transpileModule(readFileSync(path, "utf8"), {
    compilerOptions: {module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, jsx: ts.JsxEmit.ReactJSX},
    fileName: path,
  });
  const localRequire = name => name.startsWith(".")
    ? loadSource(resolve(dirname(path), `${name}.ts`)) : require(name);
  new Function("require", "module", "exports", outputText)(localRequire, module, module.exports);
  return module.exports;
}
const {RunOverview} = loadSource(fileURLToPath(new URL("../src/components/RunOverview.tsx", import.meta.url)));
const {ContextInspectorView} = loadSource(fileURLToPath(new URL("../src/components/ContextInspectorView.tsx", import.meta.url)));
const {deriveStageFeed} = loadSource(fileURLToPath(new URL("../src/components/stageFeed.ts", import.meta.url)));
const {runIsSettled, recoveryNotice} = loadSource(fileURLToPath(new URL("../src/hooks/runStatus.ts", import.meta.url)));
const run = {run_id: "new-run", status: "context_building", source_mode: "web", workflow_mode: "deep_research"};
function overview(context, selectedRun = run) {
  return renderToStaticMarkup(createElement(RunOverview, {
    run: selectedRun, context, feed: deriveStageFeed([]), stageState: () => "pending", onStage: () => {},
  }));
}
const readyContext = {
  run_id: "new-run", ready: true, status: "planning", total_input_tokens: 1200,
  curated_memory: [{memory_id: "m1"}], selected_skill: {name: "example-skill", version: "1.0"},
};

test("new Run with no checkpoint renders the API's minimal not-ready response", () => {
  // This is exactly the response returned before the first checkpoint exists.
  const pending = {run_id: "new-run", ready: false, status: "context_building"};
  const html = overview(pending);
  assert.match(html, /上下文正在构建/);
  assert.match(html, /class="context-entry" disabled=""/);
  assert.doesNotMatch(html, /条精选记忆/);
  assert.match(renderToStaticMarkup(createElement(ContextInspectorView, {context: pending})), /上下文快照正在构建/);
});

test("missing response, pending snapshot and ready snapshot render in sequence", () => {
  assert.match(overview(undefined, null), /开始研究后可查看/);
  assert.match(overview(undefined), /上下文正在构建/);
  assert.match(overview({run_id: "new-run", ready: false, status: "queued"}), /上下文正在构建/);
  const html = overview(readyContext);
  assert.match(html, /1,200 tokens · 1 条精选记忆/);
  assert.match(html, /已加载技能 · example-skill/);
  assert.doesNotMatch(html, /class="context-entry" disabled/);
});

test("not-ready snapshot never presents partial fields as confirmed metrics or skills", () => {
  const html = overview({...readyContext, ready: false});
  assert.doesNotMatch(html, /1,200 tokens|条精选记忆|example-skill/);
});

test("paused Run with a pending resume keeps polling until the lease wait ends", () => {
  assert.equal(runIsSettled({status: "paused", resume_requested: true}), false);
  assert.equal(runIsSettled({status: "paused", resume_requested: false}), true);
  assert.equal(runIsSettled({status: "interrupted"}), false);
  assert.equal(runIsSettled({status: "executing"}), false);
  assert.equal(runIsSettled({status: "completed", resume_requested: true}), true);
  assert.equal(runIsSettled({status: "cancelled", resume_requested: true}), true);
});

test("recovery notice follows the latest wait, resume, timeout or cancel event", () => {
  const waiting = [{event_type: "run.recovery_waiting"}];
  assert.equal(recoveryNotice(waiting).waiting, true);
  assert.equal(recoveryNotice([...waiting, {event_type: "run.resumed"}]), null);
  assert.equal(recoveryNotice([...waiting, {event_type: "run.cancelled"}]), null);
  assert.equal(recoveryNotice([...waiting, {event_type: "run.recovery_deferred"}]).waiting, false);
  assert.equal(recoveryNotice([]), null);
});
