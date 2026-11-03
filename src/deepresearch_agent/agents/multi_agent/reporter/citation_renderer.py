"""Render explicitly selected citations without guessing source/claim matches."""
import json
import re

from langchain_core.messages import AIMessage, HumanMessage


class CitationBindingError(ValueError):
    """A report protocol error, distinct from a retrieval failure."""


def citation_repair_messages(messages, response, error, allowed_ids):
    return [*messages, AIMessage(content=str(response.content)), HumanMessage(content=
        "仅修正上一份报告的 JSON 结构与引用关联，不重新检索，也不添加新事实。"
        f"校验失败：{error}。允许的证据 ID：{json.dumps(sorted(allowed_ids))}。"
        "每个正文段落必须明确选择实际支撑它的 evidence_ids；不能为无依据的结论随便补一个 ID。"
        "无法支撑的内容应在局限中说明。保留已有实质内容，不因修复格式强制缩成短答；只输出原约定的 JSON 对象。")]


def render_cited_paragraphs(paragraphs, allowed_ids):
    if not isinstance(paragraphs, list) or not paragraphs:
        raise ValueError("研究报告缺少可引用的正文段落")
    parts = []
    for index, paragraph in enumerate(paragraphs, 1):
        if not isinstance(paragraph, dict):
            raise CitationBindingError(f"第 {index} 段必须为对象")
        raw_text = paragraph.get("text", "")
        text = " ".join(raw_text.split()) if isinstance(raw_text, str) else ""
        ids = paragraph.get("evidence_ids")
        inline = re.findall(r"\[\^?(ev_[A-Za-z0-9_-]+)\]", text)
        # Accept an explicitly cited same-paragraph ID when the model omitted
        # its duplicate JSON array. This never chooses a source for the model.
        if not ids and inline:
            ids = inline
        if not text or not isinstance(ids, list) or not ids:
            raise CitationBindingError(f"第 {index} 段正文为空或缺少 evidence_ids")
        normalized = []
        for eid in ids:
            value = eid.strip() if isinstance(eid, str) else ""
            wrapped = re.fullmatch(r"\[\^?(ev_[A-Za-z0-9_-]+)\]", value)
            normalized.append(wrapped.group(1) if wrapped else value)
        if not all(eid in allowed_ids for eid in normalized) or any(eid not in allowed_ids for eid in inline):
            raise CitationBindingError(f"第 {index} 段引用了本次提供的证据集合之外的 ID")
        suffix = " ".join(f"[{eid}]" for eid in dict.fromkeys(normalized) if eid not in inline)
        parts.append(f"{text} {suffix}".strip())
    return "\n\n".join(parts)


def render_research_report(payload, allowed_ids, *, max_sections=3, max_paragraphs=2):
    parts = []
    sections = payload.get("sections")
    if not isinstance(sections, list) or not sections:
        raise ValueError("研究报告缺少 sections")
    for section in sections[:max_sections]:
        if not isinstance(section, dict):
            raise CitationBindingError("研究报告的 section 必须为对象")
        title = " ".join(str(section.get("title", "研究发现")).split()).lstrip("# ")[:100]
        paragraphs = section.get("paragraphs")
        parts.extend([f"## {title}", render_cited_paragraphs(paragraphs[:max_paragraphs] if isinstance(paragraphs, list) else paragraphs, allowed_ids)])
    limitations = payload.get("limitations") or []
    if limitations:
        parts.extend(["## 局限", "\n".join("- " + " ".join(str(item).split()) for item in limitations[:3])])
    return "\n\n".join(parts)
