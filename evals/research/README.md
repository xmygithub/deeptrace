# DeepTrace 研究质量评测（离线第一版）

## 新一轮完整链路评测（pilot-v2）

`pilot-v2.json` 在运行前明确三道问题及逐项评分要求。它是开发集，与旧版分开保存，不能合并成优化前后受控实验。

```powershell
python scripts/run_research_pilot.py --case-id rag-multihop-v2 --workflow deep_research --output output/evaluation/research/my-new-trial
python scripts/run_research_baseline.py --case-id rag-multihop-v2 --output output/evaluation/research/my-new-baseline
```

这两个命令**会消耗真实模型和 Tavily 配额**，输出目录必须不存在。先查看 `--help`，不要为了拿到成功结果反复执行。工作流评测从真实 HTTP 入口创建 Session/Run，使用独立 SQLite、证据目录、缓存和新会话；保留现有模型及预算，关闭后台记忆提炼、Skill 学习，使用空 Skill 目录。不会改项目 `.env` 或业务数据库。

工作流脚本最多观察 900 秒；达到外部期限会请求取消，并单独记录期限命中，不能伪装成 Harness 自己的预算耗尽。后台任务异常退出但数据库状态未结束时也单独记录，不修改状态来制造一次正常失败。`needs_user_input` 不自动回答或换题。评测包仍通过原有离线导出和核查流程生成。

基线为一次 Provider 检索加一次模型总结，Provider/SDK 内部可能重试；保存原始材料与真实用量。它绕过应用入口和 Harness，不具备相同的完成校验，`verified_completion` 为 `null`。不同实时检索结果、提示词和完成门槛构成比较限制；输出非空不代表答案正确。

恢复实验另用 `scripts/evaluate_recovery_e2e.py`，须选择全新目录。它测试阶段检查点提交后硬退出的恢复，不覆盖工具已产生副作用但检查点尚未提交的窗口。

这套工具把已有 Run 导出为冻结评测包，供人工逐项核查，再自动汇总质量与运行成本。它不会调用 LLM、搜索服务、重新执行 Run，也不会改写应用数据库。没有质量标签时输出 `null`，不把引用格式检查或运行完成当成答案正确。

## 当前包含什么

- `cases.json`：3 道自建 pilot 题，包含问题范围、逐项评分要求和官方/论文参考链接。
- `scripts/evaluate_research.py`：导出已有 Run、导入外部基线、核查评分并汇总。
- `src/deepresearch_agent/evaluation/research.py`：只读导出、原始证据关联、评分验证、汇总。
- `tests/evaluation/test_research_evaluation.py`：使用假数据和临时 SQLite 验证计分边界。

题目 ID：`rag-multihop-001`、`agent-definition-001`、`graphrag-comparison-001`。这是小规模开发集，不是 DeepResearch Bench 官方评分；用它调参后不能再称其为独立测试集。参考链接和评分要求只能给评测者看，不自动注入被测 Agent。`accepted_questions` 是人工认可的变体，汇总时仍会提示配对问题是否完全一致。

## 1. 导出一个已有 Run

在项目根目录、已配置的 conda 环境中运行：

```powershell
python scripts/evaluate_research.py export --case-id rag-multihop-001 --run-id run_1e90281169c44198ada8482754d1b200 --output output/evaluation/research/rag-dr
```

默认只读 `data/app.db`，原始材料根目录是 `data/artifacts`；可以使用 `--database PATH --artifact-root PATH` 覆盖。不存在的数据库会报错，不会悄悄新建。

导出产物：

```text
rag-dr/
  bundle.json      # 冻结的题目、最终报告、证据全文/摘要、状态、配置及用量
  report.md        # 阅读副本，评分以 bundle.json 内的报告为准
  worksheet.md     # 人工核查要点和候选主张
  review.json      # 只编辑这个文件填写判定
```

目录存在时拒绝覆盖；第二次导出请换一个目录。问题不匹配默认报错；`--allow-question-mismatch` 允许探索性核查，但不纳入该题质量评分。

完整网页内容只取运行当时保存的 `raw_content`，并验证 URL、规范化内容哈希和路径属于当前 Run。Tavily 的 `content`、Evidence 的 `summary` 是摘要，不能冒充网页全文。原文缺失/哈希不符时记录为 `summary_only`，绝不联网补抓后假装是当时的材料。GraphRAG 原始资料目前没有专门适配器，也按资料不足处理。

明显的 `sk-`/`tvly-` 密钥和 Bearer 凭据会脱敏；不导出完整运行配置、Memory 或历史聊天。评测包仍可能含研究内容，仅在预期范围内使用。脱敏是导出副本处理，不改应用数据。

失败 Run 也应导出：它没有最终报告时保留失败状态、Token 和耗时，所有报告质量分留空。不能根据几段事件预览伪造完整报告。即使磁盘留有未关联的原始搜索文件，工具也只报告文件数，不擅自重建丢失的证据 ID。

## 2. 填写人工核查

所有判定默认 `pending`。先填写 `reviewer` 和带时区的 `reviewed_at`。`reviewer_kind: human` 表示实际人工核查，`assistant` 表示 AI 核查，不能把 AI 结果写成人工已确认结果。AI 复核另存 `review.assistant.json`，保留原始人工待填表；汇总按核查者类型分别计数，不混合计算人工与 AI 的平均分。

### 要点覆盖

`coverage` 中必须保留全部 criterion。每项判为 `met` / `not_met`，写 `reason`；`met` 还必须给出报告原文中确实存在的 `report_quote`。未完成全部核查时 `coverage_rate` 保持空值；全部完成后为满足项数 / 要求项数。要点覆盖不代表主张正确。

### 主张与来源

工具从正文中均匀抽取最多 12 个候选段落（`--max-claims` 可调），不含思考标签、代码块、证据附录。它不是语义主张提取模型。人工应把含多个事实的段落拆为单一 `claim_text`，可增删候选；每个 `claim_id` 唯一，`report_quote` 必须是报告中的精确片段，用来保留主张及引用的上下文。正式判定不能省略 `claim_text`。

判定：

- `supported`：引用原文支持该主张。
- `contradicted`：引用原文与该主张矛盾。
- `unsupported`：已经核查，引用没有支持该主张，或主张没有提供依据。
- `unverifiable`：原文缺失、资料不够或无法判断。
- `pending`：尚未核查。

支持/矛盾判断必须指定报告片段中实际出现的 `evidence_ids`，以及 `evidence_quotes` 中的来源原文片段。只有摘要、证据 ID 不存在、来源片段不在原文中，都不能标记为已由原文支持。

例如下面仅为填写结构示例，不能复制为真实评分：

```json
{
  "claim_id": "claim_001",
  "claim_text": "本条只核查的单一事实主张",
  "report_quote": "报告中的原文片段 [ev_example]",
  "evidence_ids": ["ev_example"],
  "verdict": "supported",
  "reason": "说明引用如何支持这个具体主张，不能只写链接存在",
  "evidence_quotes": [{"evidence_id": "ev_example", "quote": "保存来源中的精确片段"}]
}
```

主张全部完成判定后：

- `claim_support_rate_resolved` = supported / (supported + unsupported + contradicted)。这是已抽查、可核验部分的支持率。
- `claim_verifiability` = (supported + unsupported + contradicted) / 已判定主张数。
- `unverifiable` 单独报数；只要仍有 pending，以上比例保持空值。

不能把抽样支持率写成全文准确率；不能只抽容易验证的主张再宣称整篇可靠。即便精确片段存在，程序也不能证明语义蕴含，实际判定仍由核查者负责。

### 报告评分

`quality` 四维各为 1–5 分，也可先保留 `null`；每项都需要理由和报告片段。四项都填写后才给出四维等权平均，不能将其称为答案准确率或官方 RACE 分。

| 分值 | 统一锚点 |
|---|---|
| 1 | 核心要求基本未满足，存在严重缺失/错误 |
| 2 | 满足少部分要求，明显缺失影响结论 |
| 3 | 基本满足要求，有明确不足 |
| 4 | 大部分要求满足，只有少量不足 |
| 5 | 在本题范围内充分满足要求，核查理由能说明依据 |

四维分别关注：`coverage` 范围覆盖、`analysis` 原因与取舍、`instruction_following` 指令范围、`readability` 组织表达。请与题目中的逐项要求结合使用；技术正确性由主张核查共同约束，不能因为文笔好就认定可靠。

## 3. 生成结果

```powershell
python scripts/evaluate_research.py summarize --package output/evaluation/research/rag-dr --output output/evaluation/research/summary-v1
```

输出 `summary.json` 和 `summary.md`。可重复传入 `--package` 汇总多个 Run。相同 Run 不允许重复计数。运行完成率只有存在完成校验测量时才计算，外部导入的未知测量不会当成零。

AI 首轮核查结果使用单独文件：

```powershell
python scripts/evaluate_research.py summarize --package output/evaluation/research/2026-10-23-pilot/rag-deepresearch --package output/evaluation/research/2026-10-23-pilot/rag-per-budget-exhausted --package output/evaluation/research/2026-10-23-pilot/agent-per --review-file review.assistant.json --output output/evaluation/research/ai-summary-next
```

`--review-file` 只能是包内文件名。首轮 AI 结果没有独立人工校准，不能作为人工真值或简历中的质量提升指标。通过当前对话进行核查时，模型 Token 无法单独计量，评测用量记为 `null`；不能因为没有额外 API 请求就填写零 Token。

新工具使用 `latency_seconds = completed_at - started_at` 作为端到端耗时，同时保留系统预算记录的 `elapsed_seconds`，两者不可混用。LLM Token 是运行时已记录的用量，不是账单金额，也不保证底层未返回 usage 的请求都被统计。

本工具暂不自动把抽样结果写回 `evals/system/cases.json`，避免把局部核查冒充全报告语义支持率。原 `evaluate_runs.py` 和 API 行为不变。

## 4. 与基线比较

基线定义为同题“一次搜索 + 总结”。第一版不自动发请求、不伪造一条基线 Run。可以另行生成并记录基线结果，再通过 `import` 导入；生成模型、参数、真实用量和耗时都由执行者如实记录。

外部快照最小结构：

```json
{
  "run_id": "baseline_unique_experiment_id",
  "question": "与被测问题完全一致",
  "status": "completed",
  "source_mode": "web",
  "workflow_mode": "one_search_then_summarize",
  "report": "真实生成的报告",
  "model": {"llm_model": "实际模型", "temperature": 0},
  "metrics": {"verified_completion": null, "llm_tokens": null, "latency_seconds": null},
  "evidence": []
}
```

`null` 表示尚未测量，不应为了比较填零。若有证据，使用 bundle 中相同结构：`evidence_id`、`source_id`、`content_hash`、`content: {kind, text}`。raw_snapshot 的哈希按 `sha256(规范化空白后的文本)` 验证；没有全文就使用 `summary_only` 和空 text。导入的数据属于执行者提供的实验记录，不会伪装成应用数据库 Run。

```powershell
python scripts/evaluate_research.py import --input output/baseline.json --case-id rag-multihop-001 --arm baseline --output output/evaluation/research/rag-baseline
python scripts/evaluate_research.py summarize --package output/evaluation/research/rag-dr --package output/evaluation/research/rag-baseline --output output/evaluation/research/comparison-v1
```

只对同一 case 下各一份 agent/baseline 自动配对，多个 trial 请分组汇总，避免任意交叉配对。差值为 Agent 减基线；问题、模型参数、证据快照或评分不一致/缺失时列出限制，不能据此声称受控实验收益。同样的搜索材料适合隔离报告生成效果；不同检索链路的端到端比较还应固定检索环境和预算，并承认搜索结果波动，不能声称证据完全相同。

## 首次历史导出说明

2026-10-23 首次验证使用已有 Run，不重跑 Agent：RAG 的 DeepResearch 成功 Run、RAG 的 PER 预算耗尽 Run、`agent是什么` 的 PER 成功 Run。它们是历史问题复盘，不是预注册测试：评分标准在这些运行之后制定，存在选择偏差；两个工作流也不等于单次搜索基线。因此首次汇总只展示已有用量与待核查状态，不宣称质量改进。

随后按用户要求完成首轮 AI 核查，独立保存为各包的 `review.assistant.json`，结果位于 `output/evaluation/research/2026-10-23-pilot/summary-ai-v1/`。两份成功报告完成了要点与四维预评分；失败 Run 没有最终报告，不给报告分。RAG 报告的 12 条抽样主张因实际引用缺原文而全部记为无法核验；Agent 报告的 8 条抽样主张中，2 条原文支持、2 条规范引用不支持、4 条无法核验。这不等于全文准确率，也不等于事实真伪的最终判断。

首轮也暴露了题集限制：部分 criterion 把多个要求合在一起，且“agent是什么”的原始问题没有明确要求工具调用例子。严格完整达标率可能偏低，须结合部分覆盖的文字理由阅读。下一版本应在新一轮运行前拆分原子要求、明确问题语境并独立人工校准；不能看完分数再改旧标准。原始 bundle 和人工待填表均保留。

恢复评测仍使用独立故障实验；本工具不会根据一个 `run.resumed` 事件就推断“没有重复搜索”。本轮不修改恢复机制。
