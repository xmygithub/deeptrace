"""
引用格式化模块
"""
from typing import Iterable

from langchain_core.language_models.chat_models import BaseChatModel

from deepresearch_agent.agents.multi_agent.core.retrieval_result import RetrievalResult


class CitationFormatter:
    """
    引用格式化器，将RetrievalResult转换成符合要求的引用列表
    """

    def __init__(self, llm: BaseChatModel | None = None) -> None:
        # Kept as an injectable compatibility argument; bibliography formatting
        # is deterministic and never spends a model call.
        self._llm = llm

    def format_references(
        self,
        retrieval_results: Iterable[RetrievalResult],
        citation_style: str = "default",
    ) -> str:
        results = list(retrieval_results)
        provenance = []
        seen = set()
        for result in results:
            if result.result_id in seen:
                continue
            seen.add(result.result_id)
            label = "[Web]" if result.source_mode == "web" else "[私有库]"
            location = result.metadata.url or result.metadata.source_id
            title = result.metadata.title or result.metadata.source_type
            provenance.append(f"- {label} [{result.result_id}] {title}: {location}")
        provenance_block = "\n".join(provenance)
        return f"## 参考来源\n\n{provenance_block}" if provenance else ""
