"""
一致性校验模块
"""
from typing import Dict, Any, Optional
import logging
import hashlib
import json
import re

from pydantic import BaseModel, Field
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from deepresearch_agent.config.prompts import CONSISTENCY_CHECK_PROMPT
from deepresearch_agent.models.get_models import get_llm_model
from deepresearch_agent.agents.multi_agent.planner.llm_options import planner_json_options
from deepresearch_agent.config import settings
from .review_policy import REVIEW_POLICY_VERSION, REVIEW_SYSTEM_PROMPT, ReviewResponse, normalize_review
from deepresearch_agent.harness.report_reconciliation import report_paragraphs

_LOGGER = logging.getLogger(__name__)


class ReviewProtocolError(ValueError):
    def __init__(self, diagnostics):
        super().__init__("一致性检查输出协议错误（已重试一次），不能判为内容不一致：" + diagnostics[-1]["reason"])
        self.diagnostics = diagnostics


class ConsistencyCheckResult(BaseModel):
    """
    一致性检查结果
    """
    is_consistent: bool = Field(default=True, description="报告是否通过检查")
    issues: list[Dict[str, Any]] = Field(default_factory=list, description="问题列表")
    corrections: list[Dict[str, Any]] = Field(default_factory=list, description="修正建议")
    raw_response: Optional[str] = Field(default=None, description="LLM原始输出")
    report_hash: Optional[str] = Field(default=None, description="实际检查的报告版本")
    warnings: list[Dict[str, Any]] = Field(default_factory=list, description="不触发重写的建议与未核验项")
    policy_version: Optional[str] = None
    model_is_consistent: Optional[bool] = None


class ConsistencyChecker:
    """
    调用LLM进行事实与引用一致性校验
    """

    def __init__(self, llm: Optional[BaseChatModel] = None) -> None:
        self._llm = llm or get_llm_model()

    def check(self, report_content: str, evidence_list: str) -> ConsistencyCheckResult:
        sources = json.loads(evidence_list)
        if not isinstance(sources, list) or any(not isinstance(s, dict) or not s.get("evidence_id") for s in sources):
            raise ValueError("一致性审查必须提供带编号的来源数组")
        paragraphs, _ = report_paragraphs(report_content)
        prompt = CONSISTENCY_CHECK_PROMPT.format(
            report_content=json.dumps({pid: {"text": text, "citation_ids": re.findall(r"\[\^?(ev_[A-Za-z0-9_-]+)\]", text)} for pid, text in paragraphs.items()}, ensure_ascii=False),
            evidence_list=evidence_list,
        )
        diagnostics = []
        for attempt in range(2):
            try:
                response = self._invoke_llm(prompt, max_tokens=settings.CONSISTENCY_MAX_OUTPUT_TOKENS if attempt == 0 else settings.CONSISTENCY_RETRY_MAX_OUTPUT_TOKENS)
                parsed = self._parse_response(response)
                verdict = ReviewResponse.model_validate(parsed)
                blockers, warnings, corrections = normalize_review(verdict, report_content, sources)
                return ConsistencyCheckResult(is_consistent=not blockers, issues=blockers, warnings=warnings,
                    corrections=corrections, raw_response=response, policy_version=REVIEW_POLICY_VERSION,
                    model_is_consistent=verdict.is_consistent,
                    report_hash=hashlib.sha256(report_content.encode("utf-8")).hexdigest())
            except ValueError as exc:
                hint = str(exc)[:180] if type(exc) is ValueError else "字段类型、分类或必需字段错误"
                diagnostics.append({"attempt": attempt + 1, "error_type": type(exc).__name__, "reason": hint})
                _LOGGER.warning("ConsistencyChecker output rejected (attempt %s, %s)", attempt + 1, type(exc).__name__)
                if attempt:
                    raise ReviewProtocolError(diagnostics) from exc
                prompt += '\n上次输出未满足协议：' + hint + '。只返回完整JSON，issues只选择真实paragraph_id和evidence_ids，不输出report_quote、source_quotes或推理。'

    def _invoke_llm(self, prompt: str, *, max_tokens: int) -> str:
        message = self._llm.invoke([
            SystemMessage(content=REVIEW_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ], **planner_json_options(self._llm), max_tokens=max_tokens)
        if getattr(message, "response_metadata", {}).get("finish_reason") == "length":
            raise ValueError("一致性检查输出被截断")
        content = getattr(message, "content", None)
        if isinstance(content, list):
            content = "".join(item if isinstance(item, str) else str(item.get("text", "")) for item in content if isinstance(item, (str, dict)))
        if not isinstance(content, str) or not content.strip():
            raise ValueError("一致性检查返回空内容")
        return content.strip()

    def _parse_response(self, response: str) -> Dict[str, Any]:
        try:
            # Accept a single fenced object for compatible gateways, but never
            # scan reasoning text/examples and mistake one of them for a verdict.
            fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", response.strip(), flags=re.S | re.I)
            parsed = json.loads(fenced.group(1) if fenced else response)
            if not isinstance(parsed, dict):
                raise ValueError("一致性结果必须为JSON对象")
            return parsed
        except (ValueError, TypeError) as exc:
            raise ValueError("一致性校验结果解析失败") from exc
