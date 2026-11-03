# 阶段 0 基线

`cases.json` 固化三个最小代表场景：DeepResearch、Plan–Execute–Report，以及带引用报告。真实 GraphRAG 冒烟依赖本机 Neo4j、LLM 与已构建索引，执行命令：

```powershell
python search_without_stream.py "国家奖学金和国家励志奖学金是否互斥？" --agent deep_research
python search_without_stream.py "总结材料中的核心规定并引用来源" --agent fusion
```

无外部服务的确定性回归由 `tests/smoke/` 覆盖：模块导入、PER 并行 Worker 隔离与归并。阶段 0/1 不伪造 Neo4j 或 LLM 的成功输出。
