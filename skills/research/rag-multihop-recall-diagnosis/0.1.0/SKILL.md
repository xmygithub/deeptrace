---
name: rag-multihop-recall-diagnosis
description: 针对 RAG 多跳问题召回率低的情况，提供从原因诊断、改进策略到验证方法的系统化流程。技能基于 web 搜索证据，覆盖查询分解、检索粒度、上下文缺失、混合检索、查询重写、迭代检索、图增强等改进方向，以及
  Hit Rate、MRR 等评估指标。
version: 0.1.0
status: candidate
source_modes:
- web
created_from_runs:
- run_7e439861ce094eb7b8ae41069286e5f8
---

## 触发条件

1. 用户询问 RAG 多跳问题召回率低的原因或改进方法
2. 用户要求提供 RAG 检索优化的验证指标或评估方法
3. 用户需要系统化诊断 RAG 检索阶段性能问题

## 输入要求

1. user_goal
2. source_mode
3. workflow_mode

## 步骤

1. rule-1
2. rule-2
3. rule-3

## 允许工具

1. tavily_search

## 失败回退

1. 若无法获取足够证据，应明确披露来源局限性，并建议用户提供更多上下文或调整搜索策略。

## 反模式

1. 仅凭单一来源或单一证据就下结论，未进行多源交叉验证
2. 将检索召回率问题简单归因于单一因素（如仅归因于嵌入模型），忽略查询分解和上下文缺失等维度
3. 提出改进策略时未附带验证方法，或仅使用定性描述而无定量指标
4. 超出 web 搜索证据范围，引入未在轨迹中出现的平台或工具（如本地数据库、特定框架）

## 停止与降级条件

1. 已覆盖原因分析、改进思路、验证方法三个核心部分，且每部分均有证据引用
2. 用户明确表示满意或要求停止
3. 达到预算限制（如最大工具调用次数或时间）

## 验证方式

1. 验证输出报告是否包含原因分析、改进思路、验证方法三个章节，且每条实质性主张均引用当前 Run 的证据 ID；检查证据来源是否至少两个独立域名；确认未使用超出 web 搜索范围的工具或平台。

## 已知限制

1. 技能仅适用于 web 搜索来源的 RAG 多跳召回率问题，不覆盖多模态 RAG 或特定框架（如 LangChain、LlamaIndex）的深度实现细节；评估指标仅涵盖检索阶段，不涉及生成质量评估。

## Machine Policy

```yaml
max_tool_calls: 3
max_evidence_cards: 15
required_sections:
- 原因分析
- 改进思路
- 验证方法
citation_required: true
source_diversity_min: 2
allowed_domains:
- '*'
disallowed_domains: []
```
