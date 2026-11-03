"""
任务分解节点模块

负责将清晰的查询拆解为结构化的任务图(TaskGraph)
"""
from typing import Optional, Dict, Any, List
import logging

from pydantic import BaseModel, Field
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage

from deepresearch_agent.config.prompts import TASK_DECOMPOSE_PROMPT
from deepresearch_agent.models.get_models import get_llm_model
from deepresearch_agent.agents.multi_agent.core.plan_spec import (
    TaskGraph,
    TaskNode,
    TASK_TYPE_CHOICES,
)
from deepresearch_agent.agents.multi_agent.tools.json_parser import parse_json_text
from deepresearch_agent.agents.multi_agent.planner.llm_options import planner_json_options

_LOGGER = logging.getLogger(__name__)

_ALLOWED_TASK_TYPES = set(TASK_TYPE_CHOICES)


class TaskDecompositionResult(BaseModel):
    """
    任务分解结果数据模型
    """
    task_graph: TaskGraph = Field(description="结构化任务图")
    raw_task_graph: Dict[str, Any] = Field(description="未经清洗的任务图原始JSON")
    raw_response: str = Field(description="LLM 原始输出，便于调试")


class TaskDecomposer:
    """
    任务分解节点
    """

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        *,
        max_tasks: int = 6,
    ) -> None:
        self._llm = llm or get_llm_model()
        self._max_tasks = max_tasks

    def decompose(self, query: str, *, source_mode: str = "graphrag") -> TaskDecompositionResult:
        """
        根据查询生成TaskGraph

        参数:
            query: 已澄清的目标查询

        返回:
            TaskDecompositionResult
        """
        prompt = TASK_DECOMPOSE_PROMPT.format(
            query=query,
            max_tasks=self._max_tasks,
        )
        if source_mode == "web":
            prompt += (
                "\n\n【不可覆盖的信息源约束】本 Run 的 source_mode=web。"
                "所有检索任务只能使用 web_search、deep_research 或 deeper_research；"
                "禁止 local_search/global_search/hybrid_search/naive_search/chain_exploration。"
                "\n【任务粒度与成本】简单定义或单一技术问题通常只需 1 个 web_search；"
                "比较问题按独立研究维度拆成 2～3 个有明确边界的检索任务。"
                "只有需要依据检索结果继续补查的信息缺口才使用 deep_research；"
                "deep_research/deeper_research 现在只返回发现、证据和未解决问题，最多搜索 2 次，不生成子报告。"
                "所有任务共享全局预算，禁止多个任务重复研究同一主题或各自生成完整报告；最终由 Reporter 统一成文。"
            )
        else:
            prompt += (
                "\n\n【不可覆盖的信息源约束】本 Run 的 source_mode=graphrag。"
                "禁止生成 web_search；检索只能使用私有知识图谱工具。"
            )

        _LOGGER.debug("TaskDecomposer prompt: %s", prompt)
        response = self._invoke_llm(prompt)
        parsed = self._parse_response(response)
        task_graph = self._build_task_graph(parsed, source_mode=source_mode)
        _LOGGER.debug("TaskDecomposer graph: %s", task_graph.to_dict())
        return TaskDecompositionResult(
            task_graph=task_graph,
            raw_task_graph=parsed,
            raw_response=response,
        )

    def _invoke_llm(self, prompt: str) -> str:
        """调用LLM得到纯文本输出"""
        structured_llm = self._llm.bind(**planner_json_options(self._llm))
        message: BaseMessage = structured_llm.invoke(prompt)  # type: ignore[assignment]
        content = getattr(message, "content", None) or str(message)
        return content.strip()

    def _parse_response(self, response: str) -> Dict[str, Any]:
        try:
            return parse_json_text(response)
        except ValueError as exc:
            _LOGGER.error("TaskDecomposer JSON解析失败: %s | 原始输出: %s", exc, response)
            raise ValueError("无法解析任务分解输出为有效JSON") from exc

    def _build_task_graph(self, data: Dict[str, Any], *, source_mode: str = "graphrag") -> TaskGraph:
        """
        将原始JSON转换为TaskGraph模型

        会执行以下清洗策略：
            1. 自动补充缺失字段（status、priority、estimated_tokens等）
            2. 规范化任务类型，无法识别的映射到custom并保留原始信息
            3. 确保依赖字段为列表
        """
        nodes_data: List[Dict[str, Any]] = data.get("nodes") or []
        sanitized_nodes: List[TaskNode] = []

        for raw in nodes_data:
            node_dict = dict(raw)

            task_type = node_dict.get("task_type", "custom")
            if source_mode == "web" and task_type in {
                "local_search", "global_search", "hybrid_search", "naive_search", "chain_exploration"
            }:
                task_type = "web_search"
            elif source_mode == "graphrag" and task_type == "web_search":
                task_type = "hybrid_search"
            if task_type not in _ALLOWED_TASK_TYPES:
                original_type = task_type
                task_type = "custom"
                # 将原始类型记录在参数中，方便后续人工校正
                parameters = node_dict.setdefault("parameters", {})
                parameters["original_task_type"] = original_type
            node_dict["task_type"] = task_type
            node_dict["source_mode"] = source_mode

            # 补充必备字段
            node_dict.setdefault("priority", 2)
            node_dict.setdefault("estimated_tokens", 500)
            node_dict.setdefault("depends_on", [])
            node_dict.setdefault("entities", [])
            node_dict.setdefault("parameters", {})
            node_dict.setdefault("status", "pending")

            # 依赖字段需为列表
            depends_on = node_dict.get("depends_on")
            if isinstance(depends_on, str):
                node_dict["depends_on"] = [dep.strip() for dep in depends_on.split(",") if dep.strip()]
            elif depends_on is None:
                node_dict["depends_on"] = []

            try:
                sanitized_nodes.append(TaskNode(**node_dict))
            except Exception as exc:  # noqa: BLE001
                _LOGGER.error("TaskNode构建失败: %s | 原始节点: %s", exc, raw)
                raise

        execution_mode = data.get("execution_mode", "sequential")
        task_graph = TaskGraph(nodes=sanitized_nodes, execution_mode=execution_mode)
        task_graph.validate_dependencies()
        return task_graph
