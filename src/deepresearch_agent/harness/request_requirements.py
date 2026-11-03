"""Separate research deliverables from response behavior and memory requests.

The model classifies general intent. These narrow rules also normalize explicit
language/order preferences in legacy or misclassified requirement lists; a
research question *about* language/style must remain a research deliverable.
"""
import re


_RESEARCH = re.compile(r"调研|研究|比较|对比|调查|评估|分析|解释|介绍|讨论|\b(?:research|compare|analy[sz]e|investigate)\b", re.I)
_ORDER = re.compile(r"(?:先|首先|开头).{0,10}(?:结论|答案|要点).{0,18}(?:再|然后|后).{0,10}(?:解释|分析|说明).{0,8}(?:原因|理由|依据|细节)")
_LANGUAGE = re.compile(r"(?:中文|英文|汉语|英语).{0,5}(?:回答|回复|输出|撰写)|(?:回答|回复|输出|报告).{0,5}(?:中文|英文|汉语|英语)|\b(?:answer|respond|reply|write)\s+in\s+(?:Chinese|English)\b", re.I)
_LAYOUT = re.compile(r"(?:用|采用|按照|按).{0,8}(?:表格|列表|要点|Markdown|JSON)(?:格式)?(?:回答|回复|输出|呈现)?", re.I)
_CITATIONS = re.compile(r"^(?:并|请|并请)?(?:引用|标注|附上|提供|给出)(?:公开)?(?:来源|出处)(?:链接)?[。！!\s]*$")
_MEMORY = re.compile(r"^(?:请)?(?:帮我)?(?:长期)?记住[：:，,\s]|^remember\b", re.I)


def explicit_response_constraint(quote: str) -> bool:
    """Recognize behavior instructions, never a topic mentioning those words."""
    text = quote.strip()
    if _CITATIONS.fullmatch(text):
        return True
    # '再解释原因' describes response ordering, not a research subtask.
    remainder = _ORDER.sub("", text)
    if _RESEARCH.search(remainder):
        return False
    return bool(_ORDER.search(text) or _LANGUAGE.search(text) or _LAYOUT.search(text))


def normalize_requirement(item: dict) -> dict | None:
    quote = str(item.get("request_quote", ""))
    if explicit_response_constraint(quote):
        return {**item, "kind": "format"}
    if item.get("kind") == "format" and _RESEARCH.search(quote + " " + str(item.get("text", ""))):
        # A comparative report is a substantive deliverable even if the same
        # sentence also asks for references or a particular presentation style.
        return {**item, "kind": "analysis"}
    if item.get("kind") in {"memory", "context"}:
        return None
    if _MEMORY.search(quote) and not _RESEARCH.search(quote):
        return None
    return item


def research_requirements(items):
    return [normalized for item in items if (normalized := normalize_requirement(item)) is not None
            and normalized.get("kind") not in {"format", "memory", "context"}]


def response_constraints(items):
    return [normalized for item in items if (normalized := normalize_requirement(item)) is not None
            and normalized.get("kind") == "format"]
