# DeepTrace 工作流评测 v1

本方案复用项目现有验证规则和 EvaluationService，评估执行完成、交付约束、引用关联、成本和恢复。内部检查通过不等于独立事实正确。本文件与 workflow-50-v1.json 是第一版协议；题集未经过独立标注，不是公开基准。

## 题集与运行条件

- 50 道中文联网研究题，机制解释、方案比较、带约束研究、综合研究、时效及证据边界各 10 道。
- 每类前 3 道为开发集，共 15 道；其余 35 道为保留集。开发集标记 smoke 的 6 道用于 12 次试跑，不额外增加题目。
- 每题 DeepResearch / PER 各执行一次，交替先后次序。只运行当前项目的两种工作流，不加入新基线或新评分器。
- 使用相同模型、报告类型 brief、配置的运行预算；保存模型、温度、输出限制、搜索深度和返回条数。工作流内部限制可能不同，比较的是当前完整工作流，不是等调用量实验。
- 每个 trial 启动独立子进程，隔离数据库、缓存、artifacts、Skill 目录。空数据库不含历史与长期记忆；关闭后台记忆提取和学习复盘。不会修改日常 data/app.db。
- 原问题中写明的时间范围优先；联网结果受外部变化影响，保留当次证据。references 空表示未提供人工事实标注，criteria 仅供以后复核，当前不会自动计为外部质量得分。

## 指标口径

| 指标 | 分子 / 分母或定义 |
|---|---|
| 完成校验通过率 | 状态 completed 且必需检查非空并全通过的数量 / 已尝试数量；基础设施异常也保留在分母 |
| 各检查项 | passed / failed / error_or_unknown / not_run 分别计数；仅覆盖已导出评测的 Run |
| 交付覆盖通过率 | passed / (passed + failed + error_or_unknown)，未执行单列；非事实准确率 |
| 引用 ID 有效率 | 有效引用出现次数总和 / 引用出现次数总和（微平均）；无引用时 null，并统计无引用报告数 |
| 来源违规率 | source_match 失败或存在来源模式不符证据的 Run / 已导出评测 Run；执行失败导致无证据不代表有来源合规证明 |
| Token | 从原始 usage_json 读取，缺字段保留 null；统计有效样本量、均值、中位数及总量 |
| 每成功交付 Token 成本 | 包含失败的所有 Token / 成功数；只在所有已尝试样本用量完整且存在成功时输出 |
| 耗时 P50/P95 | 复用 started_at 到 completed_at，线性插值；按最终状态分组，缺时间戳单独体现有效样本量；不是纯模型响应时间 |
| 工具调用 | 复用 tool_calls 表，统计均值与失败总量；内部 Provider 重试不一定是独立调用记录 |

明确带 error/not_checked 的检查详情计入异常；旧记录未结构化保存异常时，False 仍可能包含执行异常，须回看原始依据，不能直接归因为报告内容错误。批处理从最终检查表读取最后一次结果，不宣称第一次验证通过率。

不将主张引用覆盖称为语义支持率，不将工具调用指纹重复率称为真实外部重复副作用率，不从内部检查推出答案准确率。模型未返回用量时，运行时本身也可能保守记账；本评测沿用该记录，不宣称与账单精确一致。

## 文件和命令

在项目根目录、已配置的 conda 环境运行。所有输出使用新目录；只允许重新生成派生的 summary.json / REPORT.md，已有 trial 和失败记录不覆盖。

```powershell
# 只生成 12 次试跑计划，不调用外部接口
python scripts/evaluate_workflows.py --smoke --output output/evaluation/workflow-smoke-v1

# 执行上述计划，会消耗当前 LLM / Tavily 配额
python scripts/evaluate_workflows.py --smoke --output output/evaluation/workflow-smoke-v1 --execute

# 查看已完成部分；不会发请求
python scripts/evaluate_workflows.py --output output/evaluation/workflow-smoke-v1 --summarize-only

# 全部开发集：30 次；smoke 和该批次可能重复，分析时不能作为独立题合并
python scripts/evaluate_workflows.py --split dev --output output/evaluation/workflow-dev-v1 --execute

# 冻结版本后保留集：70 次
python scripts/evaluate_workflows.py --split test --output output/evaluation/workflow-test-v1 --execute
```

中断批处理后可使用完全相同命令继续：已有 attempt.json 的试次全部跳过，包括失败。若存在未写完 attempt.json 的目录，则停止并要求新目录，避免把未确定的外部调用静默重跑。源代码或题集哈希不同也拒绝继续同批。不要在批次运行期间修改模型、凭据或参数；每次运行 manifest 提供非敏感配置复核依据。

每批包含 plan.json、冻结题集 cases.json、summary.json、REPORT.md。每个 trials/<id>/ 下有 process.log、attempt.json 和 run/；后者包含隔离数据库、manifest.json、system-evaluation.json、报告、证据与检查点。原始轨迹在隔离数据库中，报告检查详情位于 api-report.json 和 system-evaluation.json。日志按本地调试数据管理，不发布凭据与供应商原始报错。

基础设施异常与业务失败分开保留。批次仍有 pending 时，只能报告阶段性结果。小样本 P95 波动大；每题一次不能证明稳定性。开发集与保留集分别汇总，修改后保留集复测标记为回归。

## 恢复评测

复用既有脚本，不混进上述研究题完成率。建议 4 个阶段各重复 3 次，共 12 次，使用固定离线 Driver，不声称覆盖两个真实 LLM 工作流或所有崩溃位置。

```powershell
python scripts/evaluate_recovery_e2e.py --repetitions 3 --output-dir output/evaluation/workflow-recovery-v1
```

脚本以独立进程 os._exit 模拟检查点写入后退出，再通过 RunService 启动恢复。通过需同时满足恢复完成、检查点哈希完整、模拟外部操作恰好一次、最终回答恰好一条及事件一致性。报告成功数/总数、Wilson 区间及恢复耗时。恢复耗时包含恢复子进程启动及续跑，不仅是调度等待。默认 1 秒租约是测试加速参数，不代表生产 90 秒租约的等待表现。

不覆盖“外部调用成功但尚未落库”窗口、旧执行者租约失效后的写入竞争或任意代码版本迁移；这些不能由本轮结果推断已解决。

## 验收

1. 题集恰好 50 道、15/35 划分、6 道 smoke；三类计划分别 12/30/70 次。
2. 不用新模型评分；从运行时已有记录导出系统指标。
3. 失败、无引用、缺用量、未检查、基础设施异常和未完成批次均有明确口径。
4. 每个结果能追溯到题号、Run ID、配置、证据和检查依据。
5. 先离线测试，再试跑，再冻结版本运行保留集；所有成绩只写实际执行结果。
