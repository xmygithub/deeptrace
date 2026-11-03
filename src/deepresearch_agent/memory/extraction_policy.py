"""Conservative source-text boundaries for automatic long-term Memory writes.

These rules establish eligibility, not factual truth. Human approval remains
required; model confidence alone cannot turn a scenario into a project fact.
"""

import re


_SCENARIO = re.compile(r"假设|假如|如果|设想|想象|设定一个|模拟(?:一家|一个|场景)|有一家|某(?:家企业|个项目|公司|企业)|例如|比如|\b(?:suppose|assume|imagine|hypothetical|for example|what if)\b", re.I)
_REAL_RESET = re.compile(r"(?:实际|现实)中[，,：:\s]*(?:我们|我)|我的实际项目|我们实际的项目|\bin my actual project\b", re.I)
_TEMPORARY = re.compile(r"(?:本次|这次|本轮|这轮|当前任务|这道题|该任务|仅这次|暂时|今天).{0,20}(?:要求|请|需要|必须|只|不能|不要|限制|使用|采用|偏好)|仅用于(?:本次|这次)|\b(?:for this (?:task|question|run)|this time|temporarily)\b", re.I)
_USER_ASSERTION = re.compile(r"我(?:是|在|目前|现在|就读|毕业|从事|习惯|偏好|喜欢|不喜欢|更喜欢|更容易|通常|一直|长期|经常|希望以后|希望今后)|我的(?:专业|学校|岗位|工作|研究方向)|以后.{0,12}(?:都|始终)|今后.{0,12}(?:都|始终)|\bI (?:am|work|prefer|usually|always)|\bmy (?:preference|role)\b", re.I)
_PROJECT_ASSERTION = re.compile(r"(?:我们(?:的)?|我的|我(?:目前|现在)的|本|当前)(?:项目|系统|团队|产品|服务|知识库)|\b(?:our|my) (?:project|system|team|product|service)\b", re.I)
_REMEMBER = re.compile(r"(?:请)?记住|长期(?:约定|规则|偏好)|\bremember (?:that|my|our)\b", re.I)


def eligible_memory_evidence(message: str, quote: str, target: str) -> bool:
    """Require a grounded, non-hypothetical assertion about the user/project.

    Inspect source context too, so dropping '假设' from evidence_quote does not
    bypass the boundary. A real preference preceding a scenario is still valid.
    """
    if not quote.strip():
        return False
    for match in re.finditer(re.escape(quote), message):
        prefix = message[:match.end()]
        scenarios = list(_SCENARIO.finditer(prefix))
        resets = list(_REAL_RESET.finditer(prefix))
        if scenarios and (not resets or scenarios[-1].start() >= resets[-1].start()):
            continue
        # Keep the enclosing sentence, including a possible task-only qualifier
        # before/after the selected quote. Commas are not sentence boundaries.
        start = max(message.rfind(mark, 0, match.start()) for mark in ("。", "！", "？", "\n")) + 1
        end = min((pos for mark in ("。", "！", "？", "\n") if (pos := message.find(mark, match.end())) >= 0), default=len(message))
        sentence = message[start:end]
        # Check the enclosing clause, not only the excerpt: a model might omit
        # the question mark / 是否 from its selected quote. A factual clause
        # followed by a separate question ("我们项目用 SQLite，怎么评估？")
        # can still qualify.
        boundaries = ("，", ",", "；", ";", "。", "！", "？", "?", "\n")
        clause_start = max(message.rfind(mark, 0, match.start()) for mark in boundaries) + 1
        clause_end = min((pos for mark in boundaries if (pos := message.find(mark, match.end())) >= 0), default=len(message))
        clause = message[clause_start:clause_end + 1]
        if _TEMPORARY.search(sentence) or re.search(r"[？?]|(?:是否|能否|是不是|应不应该|要不要|为什么|吗$)", clause):
            continue
        assertion = _USER_ASSERTION if target == "user" else _PROJECT_ASSERTION
        # Ownership must appear in the quote itself. A nearby '我的项目' must
        # not license unrelated facts elsewhere in a research request.
        if assertion.search(quote) or _REMEMBER.search(sentence):
            return True
    return False


def has_memory_signal(message: str) -> bool:
    """Cheap admission only; candidates still need full source-text validation."""
    return bool(_USER_ASSERTION.search(message) or _PROJECT_ASSERTION.search(message) or _REMEMBER.search(message))
