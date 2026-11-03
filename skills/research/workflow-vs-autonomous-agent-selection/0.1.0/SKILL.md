---
name: workflow-vs-autonomous-agent-selection
description: A skill for researching and comparing AI workflows vs autonomous agents,
  covering definitions, applicability, reliability, cost, and selection frameworks,
  based on web-sourced evidence.
version: 0.1.0
status: candidate
source_modes:
- web
created_from_runs:
- run_cc2e31892b3c443ea82dca56d37aae0c
---

## 触发条件

1. User asks about differences between AI workflow and autonomous agent
2. User asks about when to use workflow vs agent
3. User asks about reliability or cost comparison of workflows vs agents
4. User asks for case studies or selection frameworks for agentic systems

## 输入要求

1. User goal specifying the comparison focus (e.g., applicability, reliability, cost)
2. Source mode (web) and workflow mode (plan_execute_report) as per harness

## 步骤

1. When comparing workflows and autonomous agents, always distinguish between fixed, predefined paths (workflows) and dynamic, model-driven decision-making (agents).
2. When assessing reliability, note that workflows offer higher determinism and predictability, while autonomous agents offer flexibility but with variable latency and higher token costs.
3. When recommending a choice, consider the task's predictability: use workflows for well-defined, stable processes; use autonomous agents for tasks requiring adaptation and complex decision-making.
4. When citing sources, always distinguish between factual claims from sources and your own recommendations, and ensure each claim is backed by a traceable evidence reference.

## 允许工具

1. tavily_search

## 失败回退

1. If evidence is insufficient, disclose the limitation and suggest further research or alternative sources.

## 反模式

1. Do not treat workflows and agents as mutually exclusive; many systems use hybrid designs.
2. Do not claim universal superiority of one approach without considering the specific use case.
3. Do not extrapolate findings to platforms or modes not present in the evidence (e.g., no offline or non-web sources).

## 停止与降级条件

1. Stop when the user goal is fully addressed with sufficient evidence and clear recommendations.
2. Stop if no new evidence is found after multiple searches and the information gap cannot be filled.

## 验证方式

1. Verify that all claims are supported by evidence IDs from the run, and that the report includes required sections and citations as per the completion contract.

## 已知限制

1. Only applies to web-sourced research; does not cover other platforms or modes. Cost and reliability figures are based on the cited sources and may vary by context.

## Machine Policy

```yaml
max_tool_calls: 30
max_tavily_calls: 20
max_llm_tokens: 200000
max_concurrency: 4
max_plan_tasks: 8
max_replans: 2
max_task_retries: 2
tool_timeout_seconds: 60
wall_time_seconds: 1800
```
