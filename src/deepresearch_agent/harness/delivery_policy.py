"""Cheap admission to the existing delivery reviewer, independent of verbosity.

This is an intent heuristic, not the requirement list. The latter is extracted
from the actual request and validated against verbatim quotes by DeliveryService.
"""
import re


_DELIVERABLE = re.compile(
    r"比较|对比|区别|差异|优缺点|适用场景|局限|权衡|优先级|"
    r"(?:验证|评估|实施|选型)(?:方法|方案|建议|依据|标准)|"
    r"(?:引用|附上|标注|给出|提供)(?:公开)?(?:来源|出处)|"
    r"(?:给出|提供|提出|制定|设计).{0,12}(?:建议|方案|计划|步骤)|"
    r"\b(?:compar(?:e|ison)|versus|trade-?offs?|verification plan|evaluation plan|cite sources)\b",
    re.I,
)
_MULTIPART = re.compile(r"(?:分别|逐项|同时|并且|并).{0,12}(?:分析|说明|解释|给出|提出|举例|验证|比较)")


def needs_delivery_check(question: str) -> bool:
    return bool(_DELIVERABLE.search(question) or _MULTIPART.search(question))
