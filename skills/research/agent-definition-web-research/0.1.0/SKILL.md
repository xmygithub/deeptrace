---
name: agent-definition-web-research
description: 在 deep_research 模式下，针对概念定义类问题（如“agent是什么”）进行 web 搜索时，采用分层查询策略：先宽泛查询获取整体定义，再深入查询获取分类与细节；筛选证据时优先选择包含明确定义、特征列举、分类体系的来源；报告结构应包含核心定义、特征、工作流程、分类、与传统
  AI 的区别等章节。
version: 0.1.0
status: candidate
source_modes:
- web
created_from_runs:
- run_77a6adc6ea3248d7a2496eece99123bd
---

## 触发条件

1. 用户目标为概念定义类问题（如“X是什么”）
2. source_mode 为 web
3. workflow_mode 为 deep_research
4. 需要生成包含定义、特征、分类、区别等章节的结构化报告

## 输入要求

1. 用户目标（如“agent是什么”）
2. source_mode（web）
3. workflow_mode（deep_research）

## 步骤

1. 使用分层查询策略：第一轮使用宽泛查询（如“agent是什么？”）获取整体定义和初步特征；第二轮使用更具体的查询（如“在计算机科学和人工智能领域，agent的定义和分类是什么？”）获取分类体系和细节。
2. 筛选证据时，优先选择包含明确定义、核心特征列举、分类体系或与传统 AI 对比的内容；忽略仅提供示例或缺乏结构化信息的来源。
3. 报告结构应包含：核心定义、大模型 Agent 的说明、核心特征、工作流程、技术演进、与传统 AI 的区别、总结；每个章节需引用至少一条证据。

## 允许工具

1. tavily_search

## 失败回退

1. 若证据不足，可增加一次查询（如“AI Agent 与传统 AI 的区别”），但总查询次数不超过 3 次；若仍不足，则明确披露单一来源局限。

## 反模式

1. 仅使用单一查询，不进行分层深入搜索
2. 接受缺乏定义或分类的模糊来源
3. 报告缺少核心特征或分类章节
4. 在非 web 来源或非 deep_research 模式下使用本技能

## 停止与降级条件

1. 已获取至少 5 条有效证据，覆盖定义、特征、分类等维度
2. 证据来源达到多样性要求（至少两个独立来源，或明确披露单一来源局限）
3. 报告所有必需章节已填充且引用完整

## 验证方式

1. 验证报告是否包含所有必需章节（核心定义、特征、工作流程、分类、区别等），且每条实质性主张都有对应的 evidence_id 引用；检查证据来源是否与 source_mode 匹配（web）。

## 已知限制

1. 仅适用于 web 来源的 deep_research 模式；不适用于其他来源模式（如学术、新闻）或非 deep_research 工作流；不保证覆盖所有 Agent 相关子领域（如多智能体系统）。

## Machine Policy

```yaml
max_tool_calls: 3
max_evidence: 15
required_sections:
- 核心定义
- 核心特征
- 工作流程
- 分类
- 与传统 AI 的区别
citation_required: true
source_mode_match: true
```
