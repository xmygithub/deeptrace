"""
章节写作模块

实现分批写作能力，支持超过LLM上下文长度的长文档生成
"""
from typing import List, Dict, Any, Optional, Iterable
import logging
import textwrap
import json
import re

from pydantic import BaseModel, Field
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from deepresearch_agent.config.prompts import (
    SECTION_WRITE_PROMPT,
    SECTION_WRITE_SYSTEM_PROMPT,
)
from deepresearch_agent.models.get_models import get_llm_model
from deepresearch_agent.agents.multi_agent.planner.llm_options import planner_json_options
from deepresearch_agent.agents.multi_agent.tools.json_parser import parse_json_text
from .citation_renderer import render_cited_paragraphs, citation_repair_messages
from deepresearch_agent.config import settings
from deepresearch_agent.agents.multi_agent.core.retrieval_result import RetrievalResult
from deepresearch_agent.agents.multi_agent.reporter.outline_builder import (
    ReportOutline,
    SectionOutline,
)

_LOGGER = logging.getLogger(__name__)


class SectionWriterConfig(BaseModel):
    """
    章节写作配置
    """
    max_evidence_per_call: int = Field(
        default=8,
        ge=1,
        description="单次写作调用可使用的最大证据数量"
    )
    max_previous_context_chars: int = Field(
        default=4000,
        ge=200,
        description="多批写作时保留的前文摘要字符数"
    )
    enable_multi_pass: bool = Field(
        default=True,
        description="是否启用多批写作以支持超长章节"
    )


class SectionDraft(BaseModel):
    """
    章节写作结果
    """
    section_id: str = Field(description="章节ID")
    content: str = Field(description="章节Markdown内容")
    used_evidence_ids: List[str] = Field(default_factory=list, description="写作中引用的证据ID")


class SectionWriter:
    """
    章节写作器
    """

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        config: Optional[SectionWriterConfig] = None,
    ) -> None:
        self._llm = llm or get_llm_model()
        self.config = config or SectionWriterConfig()
        self.model_calls = 0
        self.call_limit = None

    def _call_model(self, messages, options):
        if self.call_limit is not None and self.model_calls >= self.call_limit:
            from deepresearch_agent.harness.budgets import BudgetExceeded
            raise BudgetExceeded("report_model_calls", self.model_calls + 1, self.call_limit)
        self.model_calls += 1
        return self._llm.invoke(messages, **options)

    def write_section(
        self,
        outline: ReportOutline,
        section: SectionOutline,
        evidence_map: Dict[str, RetrievalResult],
        fallback_evidence_ids: Optional[List[str]] = None,
    ) -> SectionDraft:
        """
        根据大纲与证据撰写章节内容
        """
        evidence_ids = self._select_evidence_ids(section, evidence_map, fallback_evidence_ids)
        evidence_entries = [evidence_map[eid] for eid in evidence_ids if eid in evidence_map]
        if not evidence_entries and fallback_evidence_ids:
            # 使用备用证据，避免完全无材料
            evidence_entries = [
                evidence_map[eid] for eid in fallback_evidence_ids[: self.config.max_evidence_per_call]
                if eid in evidence_map
            ]
            evidence_ids = [result.result_id for result in evidence_entries]

        batches = self._split_into_batches(evidence_entries, self.config.max_evidence_per_call)
        contents: List[str] = []
        used_ids: List[str] = []
        outline_context = self._build_outline_snapshot(outline, section)
        outline_context_text = json.dumps(outline_context, ensure_ascii=False)

        if not evidence_entries:
            placeholder = (
                f"⚠️ 当前章节《{section.title}》未检索到可引用的证据。"
                " 请补充检索或调整查询后重试。"
            )
            _LOGGER.warning(
                "No evidence available for section_id=%s title=%s",
                section.section_id,
                section.title,
            )
            return SectionDraft(
                section_id=section.section_id,
                content=placeholder,
                used_evidence_ids=[],
            )

        for batch_index, batch in enumerate(batches, start=1):
            evidence_list_text = self._format_evidence(batch)
            context_instruction = ""

            if self.config.enable_multi_pass and len(batches) > 1:
                context_instruction = textwrap.dedent(
                    f"""
                    **写作阶段**: 第{batch_index}/{len(batches)}批，请确保与前文衔接。
                    {'**前文摘要**: ' + self._extract_previous_summary(contents) if contents else ''}
                    """
                ).strip()

            prompt = SECTION_WRITE_PROMPT.format(
                outline=outline_context_text,
                section_id=section.section_id,
                section_title=section.title,
                section_summary=section.summary,
                estimated_words=section.estimated_words,
                evidence_list=evidence_list_text + ("\n\n" + context_instruction if context_instruction else "")
            )

            allowed_ids = set(re.findall(r'"evidence_id":\s*"(ev_[A-Za-z0-9_-]+)"', evidence_list_text))
            generated = self._invoke_llm(prompt, allowed_evidence_ids=allowed_ids)
            contents.append(generated.strip())
            used_ids.extend([item.result_id for item in batch])

        final_content = "\n\n".join(contents).strip()
        final_content = self._sanitize_content(section.title, final_content)
        return SectionDraft(
            section_id=section.section_id,
            content=final_content,
            used_evidence_ids=used_ids,
        )

    def _select_evidence_ids(
        self,
        section: SectionOutline,
        evidence_map: Dict[str, RetrievalResult],
        fallback_evidence_ids: Optional[List[str]],
    ) -> List[str]:
        """按优先级选择证据ID"""
        if section.evidence_ids:
            return section.evidence_ids
        if fallback_evidence_ids:
            return fallback_evidence_ids
        return list(evidence_map.keys())

    def _split_into_batches(
        self,
        evidence_entries: List[RetrievalResult],
        batch_size: int,
    ) -> List[List[RetrievalResult]]:
        """按批次切分证据列表"""
        if not evidence_entries:
            return [[]]
        batches: List[List[RetrievalResult]] = []
        for i in range(0, len(evidence_entries), batch_size):
            batches.append(evidence_entries[i:i + batch_size])
        return batches

    def _format_evidence(self, entries: Iterable[RetrievalResult]) -> str:
        """Keep source identity next to bounded evidence text, including digests."""
        lines = []
        for item in entries:
            raw = item.evidence if isinstance(item.evidence, str) else json.dumps(item.evidence, ensure_ascii=False)
            snippet = raw[:settings.REPORT_SOURCE_MAX_CHARS]
            snippet_truncated = len(raw) > len(snippet)
            if snippet_truncated:
                snippet += "\n（后续内容已截断，不得推断未展示内容）"
            metadata = item.metadata
            if metadata.extra.get("is_evidence_digest"):
                # A digest must not inherit the first member's title/URL. Only
                # expose mappings whose IDs survived bounded summary truncation.
                visible_ids = set(re.findall(r"\[(ev_[A-Za-z0-9_-]+)\]", snippet))
                sources = [
                    source for source in metadata.extra.get("evidence_sources", [])
                    if source["evidence_id"] in visible_ids
                ]
                lines.append(
                    "多来源摘要（容器 ID 不可引用）\n"
                    + json.dumps({"sources": sources, "content": snippet,
                                  "content_kind": "evidence_digest", "truncated": snippet_truncated}, ensure_ascii=False)
                )
            else:
                lines.append(json.dumps({
                    "evidence_id": item.result_id,
                    "title": (metadata.title or "")[:240],
                    "source_id": metadata.source_id[:512],
                    "url": (metadata.url or "")[:512],
                    "content_kind": metadata.extra.get("content_kind") or ("document_text" if item.source_mode != "web" else "unknown"),
                    "authority_rank": int(metadata.extra.get("authority_rank") or 0),
                    "source_truncated": bool(metadata.extra.get("content_truncated")),
                    "card_truncated": bool(metadata.extra.get("card_truncated")),
                    "prompt_truncated": snippet_truncated or bool(metadata.extra.get("writing_source_truncated")),
                    "content": snippet,
                }, ensure_ascii=False))
        return "\n".join(lines) if lines else "（无可用证据）"

    def _extract_previous_summary(self, contents: List[str]) -> str:
        """
        从已有内容中截取摘要，用于多批写作的上下文提示
        """
        if not contents:
            return ""
        joined = "\n\n".join(contents)
        return joined[-self.config.max_previous_context_chars:]

    def _invoke_llm(self, prompt: str, *, allowed_evidence_ids=None) -> str:
        """调用LLM生成章节内容

        SystemMessage 保持全静态，变量内容全部在 HumanMessage 中，
        服务端前缀缓存（prefix cache）可以跨章节/跨轮次复用 system 前缀。
        """
        messages: List[BaseMessage] = [
            SystemMessage(content=SECTION_WRITE_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
        options = {}
        structured = bool(allowed_evidence_ids) and "deepseek" in str(getattr(self._llm, "model_name", "")).lower()
        if structured:
            options = planner_json_options(self._llm)
            messages[0] = SystemMessage(content=SECTION_WRITE_SYSTEM_PROMPT +
                '\n本次输出协议：只返回 JSON {"paragraphs":[{"text":"完整正文段落","evidence_ids":["ev_..."]}]}。'
                '按主题需要写作，最多 8 段，每段最多 500 字，不为凑长度重复材料。每段只用给定来源中真正支撑其内容的证据 ID；'
                'text 不包含标题、换行和引用标记，引用由程序根据 evidence_ids 渲染。不要生成无证据的开场白、总结或效果。')
        elif "deepseek" in str(getattr(self._llm, "model_name", "")).lower():
            options = planner_json_options(self._llm)
            options.pop("response_format", None)
        if structured:
            # A model formatting error is repairable from the SAME evidence.
            # Keep validation strict and retry only this section once.
            diagnostics = []
            for attempt in range(2):
                message = self._call_model(messages, options)
                try:
                    if getattr(message, "response_metadata", {}).get("finish_reason") == "length":
                        raise ValueError("报告章节输出被截断")
                    return render_cited_paragraphs(parse_json_text(str(message.content)).get("paragraphs"), allowed_evidence_ids)
                except ValueError as exc:
                    diagnostics.append({"attempt": attempt + 1, "error": str(exc)[:500],
                                        "response_chars": len(str(message.content)),
                                        "finish_reason": getattr(message, "response_metadata", {}).get("finish_reason")})
                    if attempt == 1:
                        from .citation_renderer import CitationBindingError
                        error = CitationBindingError(str(exc))
                        error.protocol_diagnostics = diagnostics
                        raise error from exc
                    _LOGGER.warning("报告章节协议校验失败，仅重写当前章节一次：%s", exc)
                    messages = citation_repair_messages(messages, message, exc, allowed_evidence_ids)
        message: BaseMessage = self._call_model(messages, options)  # type: ignore[assignment]
        content = getattr(message, "content", None) or str(message)
        return content.strip()

    @staticmethod
    def _normalize_heading_text(text: str) -> str:
        """
        归一化标题文本，便于比对是否与章节标题重复
        """
        normalized = re.sub(r"[#\s]+", "", text or "").strip()
        normalized = normalized.replace("：", ":").replace("，", ",").lower()
        return normalized

    def _sanitize_content(self, section_title: str, content: str) -> str:
        """
        移除与章节标题重复的标题行，避免最终报告出现双重小标题
        """
        if not content:
            return ""

        normalized_title = self._normalize_heading_text(section_title)
        cleaned_lines: List[str] = []

        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                heading_text = re.sub(r"^#+\s*", "", stripped)
                if self._normalize_heading_text(heading_text) == normalized_title:
                    # 跳过重复标题行
                    continue
            cleaned_lines.append(line)

        # 去除开头的空行，保持章节正文紧凑
        while cleaned_lines and not cleaned_lines[0].strip():
            cleaned_lines.pop(0)

        return "\n".join(cleaned_lines).strip()

    def _build_outline_snapshot(
        self,
        outline: ReportOutline,
        section: SectionOutline,
    ) -> Dict[str, Any]:
        """
        构造精简的纲要上下文，减少提示长度但保留章节位置信息。
        """
        try:
            index = next(
                idx for idx, item in enumerate(outline.sections) if item.section_id == section.section_id
            )
        except StopIteration:
            index = 0

        previous_title = outline.sections[index - 1].title if index > 0 else None
        next_title = (
            outline.sections[index + 1].title if index + 1 < len(outline.sections) else None
        )

        snapshot: Dict[str, Any] = {
            "outline_status": "待核验的写作计划，不能替代来源证据",
            "report_title": outline.title,
            "report_type": outline.report_type,
            "section_index": index + 1,
            "total_sections": len(outline.sections),
            "section_titles": [item.title for item in outline.sections],
        }
        if outline.abstract:
            snapshot["abstract"] = outline.abstract[:400]
        if previous_title:
            snapshot["previous_section"] = previous_title
        if next_title:
            snapshot["next_section"] = next_title
        return snapshot
