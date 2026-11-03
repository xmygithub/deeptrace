---
name: workflow-vs-agent-selection
description: 基于 web 搜索证据，系统比较 AI Agent 中工作流（Workflow）与自主规划（Autonomous Planning）两种方式的区别，评估适用场景、可靠性和成本，并结合公开案例给出选型建议。
version: 0.1.0
status: candidate
source_modes:
- web
created_from_runs:
- run_479a78bce3e24e38a7de8da95a51e585
---

## 触发条件

1. 用户询问 AI Agent 中工作流与自主规划的区别
2. 用户要求比较适用场景、可靠性、成本
3. 用户要求结合公开案例说明如何选择

## 输入要求

1. user_goal: 调研 AI Agent 中工作流与自主规划两种方式的区别，比较它们的适用场景、可靠性和成本，并结合公开案例说明如何选择。
2. source_mode: web
3. required_evidence_count: 3

## 步骤

1. rule-1: 必须使用 web 搜索获取至少 3 条独立证据，覆盖中英文来源，且每条证据需记录 source_id 和 evidence_id。
2. rule-2: 在比较工作流与自主规划时，必须从控制流归属（预定义 vs 运行时推理）角度分析，并引用至少一个来源明确说明这一核心区别。
3. rule-3: 在评估可靠性和成本时，必须引用至少一个来源说明工作流的可预测性优势，以及自主规划的灵活性和潜在成本，并区分资料结论与个人建议。
4. rule-4: 在给出选型建议时，必须结合至少一个公开案例，并明确标注案例来源，避免无来源的泛化建议。

## 允许工具

1. tavily_search

## 失败回退

1. 若搜索无法获取足够证据，则明确披露单一来源局限，并基于现有证据给出保守结论

## 反模式

1. 仅依赖单一来源或单一语言来源，未进行多源交叉验证
2. 将工作流与自主规划的区别简单归结为'有无大模型'，忽略控制流归属
3. 在未引用案例来源的情况下给出选型建议
4. 混淆资料结论与个人建议，未明确区分

## 停止与降级条件

1. 已收集至少 3 条独立证据且覆盖中英文来源
2. 已明确工作流与自主规划的核心区别（控制流归属）
3. 已比较适用场景、可靠性和成本，并引用至少一个案例
4. 已区分资料结论与个人建议

## 验证方式

1. method: 检查证据数量是否≥3，来源是否独立，是否覆盖中英文，是否引用案例，是否区分资料结论与建议
2. evidence_required: true
3. citation_required: true

## 已知限制

1. 仅适用于 web 搜索来源，不适用于其他平台
2. 不保证证据的语义支持，仅保证引用覆盖

## Machine Policy

```yaml
max_tool_calls: 3
max_evidence: 14
max_llm_tokens: 200000
timeout_seconds: 60
required_source_mode: web
required_workflow_mode: deep_research
```
