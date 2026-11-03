---
name: agent-harness-role-explainer
description: 基于 web 搜索证据，解释 Agent Harness 在 Agent 系统中的核心作用、功能模块、与模型的关系及工程实践。
version: 0.1.0
status: active
source_modes:
- web
created_from_runs:
- run_c847d975c62b4c6088bc9d7a61a5cc62
---

## 触发条件

1. 用户询问 harness 在 agent 系统中的作用或用途
2. 需要解释 Agent Harness 的定义、功能或架构
3. 需要对比 Harness 与模型、脚手架、框架等概念

## 输入要求

1. 用户关于 Agent Harness 的具体问题
2. 来自 web 搜索的可靠来源（如维基百科、技术博客、行业文档）

## 步骤

1. 定义 Agent Harness 时，必须引用其作为 LLM 软件基础设施、处理工具使用、记忆、状态持久性、执行环境及反馈回路的角色。
2. 解释 Harness 的核心功能时，应涵盖上下文装配、工具/技能调度、流程与约束、外部化状态、可观测性与评测等模块。
3. 强调 Harness 是运行时而非脚手架，持续编排执行循环、管理上下文、执行安全检查并持久化状态。
4. 使用公式 Agent = Model + Harness 说明模型与 Harness 的边界，Harness 包含所有非模型代码、配置和执行逻辑。
5. 引用具体案例（如 OpenAI 的 Harness 工程实践）说明 Harness 对 Agent 性能的显著影响。

## 允许工具

1. tavily_search

## 失败回退

1. 若证据不足，明确披露来源局限性，并建议用户查阅原始资料。

## 反模式

1. 将 Harness 与脚手架或框架混淆，忽略其运行时特性
2. 仅依赖单一来源定义 Harness，缺乏多源验证
3. 忽略 Harness 在状态管理、安全、可观测性等非功能方面的作用

## 停止与降级条件

1. 已覆盖定义、核心功能、与模型关系、工程实践等关键方面
2. 证据来源至少两个独立可靠来源
3. 用户问题得到直接且完整的回答

## 验证方式

1. 检查回答是否包含 Harness 定义、核心功能模块、与模型关系、实际案例，并确保每个主张都有对应证据引用。

## 已知限制

1. 仅适用于 web 来源的 deep_research 模式，不涵盖其他平台或未在证据中出现的领域。

## Machine Policy

```yaml
max_tool_calls: 2
max_evidence: 10
required_evidence: 1
citation_required: true
source_diversity_required: true
```
