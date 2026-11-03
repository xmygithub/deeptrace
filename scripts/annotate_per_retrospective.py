"""Apply explicitly authored retrospective annotations to frozen review packets.

Not a general automatic judge. Decisions below were authored for these archived
reports; exact report/source quotes are validated by the existing review module.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from deepresearch_agent.evaluation.research import summarize_packages

# report needle, atomic proposition, verdict, cited source, source needle, reasoning.
DECISIONS = {
"comparison-01--plan_execute_report--r1": [
 ("消息重试次数超过 16 次", "报告将重试超过16次作为RabbitMQ通用死信条件。", "unsupported", "ev_abeac3a07ecf66b5557ca234", "默认 16 次", "原文明确属于阿里云云消息队列RabbitMQ版；报告机制段省略产品限定，不能推广为所有RabbitMQ的固定阈值。"),
 ("重试超过 3 次", "报告将示例中重试超过3次进入失败队列的配置叙述为重试链路机制。", "unsupported", "ev_e96ddee0106ab7259608f21a", "当重试超过3次", "来源是具体实现说明；支持该做法存在，不支持RabbitMQ统一固定3次的概括，需保留配置或示例限定。"),
 ("中等吞吐，受磁盘影响", "报告引用的比较材料将RabbitMQ性能描述为中等吞吐、受磁盘影响。", "supported", "ev_341eca3ed22325c0718ba3b0", "中等", "仅核对报告转述与所引比较材料一致，不据此认定为普遍性能结论。"),
 ("延迟可控制在 100ms", "来源列出的典型实时日志案例延迟控制在100ms内。", "supported", "ev_1b373f6819d042d8af58fc7c", "延迟控制在 100ms", "来源给出了该案例；不推广为所有Redis Streams部署的性能保证。"),
 ("故障转移由仲裁队列", "报告据Raft复制介绍断言仲裁队列的故障转移保障。", "unsupported", "ev_08298e610b7e2ec123f85f59", "Raft for quorum", "引用片段说明复制与多数确认，未描述故障转移流程；报告正文另有未覆盖该机制的声明，结论范围超出本次证据。"),
],
"comparison-02--plan_execute_report--r1": [
 ("引入额外摘要调用", "增量摘要的实现会增加额外的摘要模型调用。", "supported", "ev_88c4540ad604abab6b28a4a8", "额外的 LLM 调用", "所引原文直接说明额外LLM调用用于压缩对话历史。"),
 ("结构化记忆存储的对比也显示", "摘要压缩节省Token但可能丢失细节。", "supported", "ev_a7245623c59f800146e929e6", "可能丢失细节", "原文表格直接列出该优缺点，支持此条转述。"),
 ("上下文缓存是缓解", "上下文缓存通过复用历史前缀减少重复计算开销。", "unsupported", "ev_8aa87fdf68bfa7f5ce501710", "缓存命中机制", "完整保存原文讨论语义缓存与分层存储，未说明模型历史前缀计算复用；报告把不同缓存机制混同，当前引用不足。"),
 ("选型框架的起点", "应按任务对近期原文与远期要点的需求选择策略。", "unverifiable", None, None, "属于报告的选型建议而非单一事实主张；保留为分析建议，不强行计为来源已证明的事实。"),
 ("上下文质量的退化往往", "上下文质量退化可能表现为渐进遗忘，而非硬性报错。", "supported", "ev_7537d79ba9e7bccf1d2d5572", "不是一个硬性错误", "原文明确描述渐进质量崩塌，与此条转述一致。"),
],
"mechanism-01--plan_execute_report--r1": [
 ("索引本质上是一种", "索引的作用可以类比书的目录。", "supported", "ev_37aee324f9811436a6c38238", "书的目录", "所引来源直接使用目录类比解释索引定位作用。"),
 ("当数据表记录规模突破百万级", "所引材料声称百万规模后全表扫描I/O成本指数增长。", "supported", "ev_c6990e388f8c6496a1d7edff", "指数级增长", "转述与来源一致，但来源本身这个技术论断缺乏实验及推导，本轮不把语义支持等同事实正确。"),
 ("插入操作需要在每个索引", "自增主键通常更有利于减少页分裂和随机写入。", "supported", "ev_2e60286cede56ff1020157bf", "自增主键通常更有利", "原文直接列出此机制性表述。"),
 ("上述维护动作共同构成写入放大", "多索引维护会增加写入路径的额外操作。", "unverifiable", None, None, "该段同时引用的一个来源只有摘要；复合推导未逐项完成原文核实，不能用摘要补成确认。"),
 ("应优先考虑在查询条件", "优先在高区分度且频繁查询的列建立索引。", "unverifiable", None, None, "本段所引来源只有summary_only，没有可核验的网页原文；保留未知。"),
],
"synthesis-01--plan_execute_report--r1": [
 ("本文设计一个覆盖研究任务", "幂等、重试与检查点应放置于报告提出的具体阶段。", "unverifiable", None, None, "这是报告综合多个来源提出的设计方案，不把推荐位置直接标记为已由单一来源证明。"),
 ("跨 Session 续跑一个 Task", "跨会话续跑需要装载授权任务状态而非默认共享上下文。", "supported", "ev_e46e763c718d933fd4980f5f", "Task State", "原文讨论跨Session的任务状态装载，支持该条机制描述。"),
 ("任务执行边界的幂等可借鉴", "只依靠唯一ID和数据库唯一约束即可阻止并行重复执行。", "unsupported", "ev_ed3a99a48392b9346867cac0", "主键唯一特性", "来源支持阻止重复插入记录，不保证外部执行互斥；报告虽标外推，仍需收窄保证。"),
 ("SeaTunnel 的检查点设计目标", "SeaTunnel检查点设计同时权衡开销与快速恢复。", "supported", "ev_310a980ef8667a85d2c52a2e", "最小化开销", "来源明确列出最小化开销及快速恢复目标，支持这条转述。"),
 ("重试必须设置超时与截止时间", "客户端应设置超时，避免无限等待占用资源。", "supported", "ev_8d7829054047f92a72b442f7", "浪费客户端的有限资源", "原文明确说明等待占用内存、线程和网络端口，并建议设置超时。"),
]}


SECTIONS = {
"comparison-01--plan_execute_report--r1": ["消息确认机制比较", "失败恢复机制比较", "适用条件与选型建议"],
"comparison-02--plan_execute_report--r1": ["信息保留机制对比：保留什么、丢失什么", "调用成本对比：token消耗、调用次数与计算开销", "综合权衡与场景化选型建议"],
"mechanism-01--plan_execute_report--r1": ["索引加快查询的原理", "索引增加写入成本的原理", "结论：读写权衡与索引设计"],
"synthesis-01--plan_execute_report--r1": ["端到端故障处理流程设计", "三者的协同分工与冲突点", "研究任务全流程的故障点分析"],
}


def main(root, version):
    packages = []
    for name, decisions in DECISIONS.items():
        folder = root / name
        bundle = json.loads((folder / "bundle.json").read_text(encoding="utf-8"))
        review = json.loads((folder / "review.json").read_text(encoding="utf-8"))
        report = bundle["run"]["report"]
        evidence = {e["evidence_id"]: e for e in bundle["run"]["evidence"]}
        claims = []
        for index, (needle, claim, verdict, eid, source_needle, reason) in enumerate(decisions, 1):
            candidates = [p.strip() for p in re.split(r"\n\s*\n", report) if needle in p]
            if not candidates:
                raise ValueError(f"Missing report quote: {name}: {needle}")
            quote = candidates[0]
            ids = list(dict.fromkeys(re.findall(r"\[\^?(ev_[A-Za-z0-9_-]+)\]", quote)))
            source_quotes = []
            if eid:
                source = evidence[eid]["content"]
                position = source["text"].find(source_needle)
                if eid not in ids or source["kind"] != "raw_snapshot" or position < 0:
                    raise ValueError(f"Missing cited raw source: {name}: {source_needle}")
                start = max(0, position - 80)
                source_quotes = [{"evidence_id": eid, "quote": source["text"][start:position + len(source_needle) + 220]}]
            claims.append({"claim_id": f"claim_{index:03d}", "report_quote": quote, "claim_text": claim,
                           "evidence_ids": ids, "verdict": verdict, "reason": reason, "evidence_quotes": source_quotes})
        review.update(claims=claims, reviewer_kind="assistant", reviewer="Codex authored retrospective review",
                      reviewed_at=datetime.now(timezone.utc).isoformat())
        for item, heading in zip(review["coverage"], SECTIONS[name]):
            match = re.search(r"(?m)^## " + re.escape(heading) + r"\s*\n", report)
            if not match:
                raise ValueError("Missing coverage section")
            end = re.search(r"(?m)^## ", report[match.end():])
            quote = report[match.start():match.end() + end.start() if end else len(report)].strip()
            item.update(verdict="met", report_quote=quote, reason=f"回顾性检查：该节正文实质讨论{item['criterion_id']}对应的{bundle['case']['criteria'][int(item['criterion_id'][1:])-1]['description']}；仅覆盖判断，不证明事实正确。")
        with (folder / f"review.assistant.{version}.json").open("x", encoding="utf-8") as f:
            json.dump(review, f, ensure_ascii=False, indent=2)
        packages.append(folder)
    result = summarize_packages(packages, review_filename=f"review.assistant.{version}.json")
    with (root / f"claim-review-summary.{version}.json").open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Validated {len(packages)} retrospective reviews; holistic quality remains ungraded")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--version", default="v2")
    args = parser.parse_args()
    main(args.root.resolve(), args.version)
