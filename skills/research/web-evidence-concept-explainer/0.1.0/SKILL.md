---
name: web-evidence-concept-explainer
description: 在 plan_execute_report/web 模式下，针对“为什么需要 X / 请结合例子说明 / 并引用来源”类概念解释请求，按原因维度、具体案例维度、引用格式三类需求组织检索与写作，并用当前
  Run 的 ev_ ID 逐段就近引用。
version: 0.1.0
status: candidate
source_modes:
- web
created_from_runs:
- run_ee3f655f99ee41bdbdfa923d3f5a7207
---

## 触发条件

1. 用户问题包含“为什么需要/为什么”类原因解释诉求
2. 用户问题包含“请结合例子说明”类案例诉求
3. 用户问题包含“引用来源”类引用格式诉求
4. source_mode 为 web 且 workflow_mode 为 plan_execute_report

## 输入要求

1. user_goal 原文（用于拆解原因/案例/引用三类需求）
2. source_mode=web 的检索工具（如 tavily_search）
3. 当前 Run 的 Evidence Ledger（ev_ ID 与来源）

## 步骤

1. 先把 user_goal 拆成可核对的需求清单：原因维度、案例维度、引用格式维度，并让每个维度在最终报告中都有对应段落。
2. 按需求维度分别发起定向检索：一次覆盖原因/机制，一次覆盖具体实现案例（如检查点、断点恢复），一次覆盖配套机制细节（如幂等/去重），避免单次宽泛检索。
3. 正文每条实质性主张就近引用当前 Run 的 ev_ ID，引用 ID 必须逐字来自 Evidence Ledger，不得手写或改写。
4. 案例段落必须落到具体机制或场景（如节点后持久化并从最近检查点恢复、基于状态的去重执行、超时重试导致重复副作用），而不是泛泛而谈。
5. 报告至少包含一个非空 Markdown 标题，并保证来源多样性：至少两个独立来源，或明确披露单一来源局限。

## 允许工具

1. tavily_search

## 失败回退

1. 若某需求维度检索不到足够证据，在报告中明确披露该维度证据不足，不虚构来源或引用 ID；若来源单一，按 source_diversity 要求披露局限。

## 反模式

1. 只做一次宽泛检索就写作，导致原因与案例维度覆盖不足
2. 正文出现无 ev_ ID 的实质性主张
3. 手写或改写引用 ID，而非逐字复制 Evidence Ledger 中的 ev_ ID
4. 案例段落只给抽象结论，不落到具体机制或场景
5. 把 web 来源经验类推到本地文件或未出现的平台

## 停止与降级条件

1. 三类需求（原因/案例/引用）均已在报告中找到对应段落且引用完整时停止补充检索
2. 达到 Run 预算上限（max_tavily_calls / max_tool_calls）时停止并披露覆盖局限
3. 连续检索无法新增与需求维度相关的证据时停止并说明局限

## 验证方式

1. 核对 delivery_coverage：原因、案例、引用三类需求均标记 covered
2. 核对 citation_integrity：citations 全部存在于当前 Run Evidence Ledger，missing 与 unresolved_citations 为空
3. 核对 claim_support：实质性主张均有 ev_ ID 引用
4. 核对 required_section：至少一个非空 Markdown 标题
5. 核对 source_diversity：至少两个独立来源或已披露单一来源局限

## 已知限制

1. 仅适用于 web 来源与 plan_execute_report 模式
2. 不验证来源在语义上是否支持主张，仅保证引用覆盖与 ID 完整性
3. 不覆盖本地文件、数据库或未在轨迹中出现的平台
4. 检索次数受 Run 预算约束

## Machine Policy

```yaml
max_tavily_calls: 20
max_tool_calls: 30
max_plan_tasks: 8
max_replans: 2
max_task_retries: 2
tool_timeout_seconds: 60
wall_time_seconds: 1800
require_citation_per_claim: true
require_required_section: true
min_independent_sources: 2
citation_id_source: current_run_evidence_ledger_only
```
