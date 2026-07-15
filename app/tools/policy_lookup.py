"""工具2：制度条款结构化查询。

区别于 RAG 的模糊语义检索，本工具面向"有明确字段的数值型制度参数"
（如报销额度上限、住宿标准），从本地 JSON（data/policy_params.json）
中按关键词精确查出结构化条目。

权限：与文档密级 ACL 对齐——条目 source 对应文件的 access_level
决定可见性；实习生不可通过本工具绕过 internal 报销/差旅参数。
"""

import json
from functools import lru_cache

from pydantic import BaseModel, Field

from app.config import PROJECT_ROOT, get_settings
from app.rag.access_control import resolve_access_level, role_can_access
from app.tools.base import BaseTool
from app.utils.logger import get_logger

logger = get_logger(__name__)

POLICY_DATA_PATH = PROJECT_ROOT / "data" / "policy_params.json"


class PolicyLookupInput(BaseModel):
    """制度参数查询的输入参数。"""

    keyword: str = Field(
        min_length=1,
        description="查询关键词，如：办公用品、住宿标准、餐补、报销时限",
    )


@lru_cache
def _load_policy_entries() -> tuple[dict, ...]:
    """加载并缓存制度参数条目。

    Returns:
        条目字典组成的元组。

    Raises:
        RuntimeError: 数据文件缺失或 JSON 格式损坏时抛出。
    """
    try:
        with open(POLICY_DATA_PATH, encoding="utf-8") as f:
            payload = json.load(f)
        return tuple(payload["entries"])
    except FileNotFoundError as exc:
        raise RuntimeError(f"制度参数数据文件不存在: {POLICY_DATA_PATH}") from exc
    except (json.JSONDecodeError, KeyError) as exc:
        raise RuntimeError(f"制度参数数据文件格式错误: {POLICY_DATA_PATH}（{exc}）") from exc


def _entry_visible(entry: dict, user_role: str | None) -> bool:
    """判断条目对当前角色是否可见（按 source 文件密级）。"""
    if not get_settings().enable_access_control:
        return True
    source = str(entry.get("source") or "")
    level = entry.get("access_level") or resolve_access_level(source)
    return role_can_access(user_role, level)


class PolicyLookupTool(BaseTool):
    """制度参数结构化查询工具（带密级过滤）。"""

    name = "policy_lookup"
    description = (
        "按关键词查询公司制度中有明确数值的参数，如报销额度上限、住宿标准、"
        "补贴金额、报销时限等。适用于'XX的报销上限是多少''出差住宿标准是多少钱'"
        "这类需要精确数字的问题。输入：keyword（查询关键词，如'办公用品''住宿标准'）。"
        "返回：命中的制度参数条目（名称、数值、单位、出处）。"
        "注意：结果受当前用户文档权限约束，无权条目不会返回。"
    )
    input_schema = PolicyLookupInput

    def run(self, params: PolicyLookupInput) -> dict:
        """按关键词查询制度参数，并按角色过滤密级。

        Args:
            params: 校验后的输入参数（含 keyword）。
                角色从 invoke(context={"user_role": ...}) 注入，不由 LLM 传入。

        Returns:
            dict：found / matches / message / filtered_by_acl。
        """
        keyword = params.keyword.strip()
        entries = _load_policy_entries()
        user_role = getattr(self, "_invoke_context", {}).get("user_role")

        keyword_hits = [
            entry
            for entry in entries
            if keyword in entry["name"]
            or any(kw in keyword or keyword in kw for kw in entry["keywords"])
        ]
        visible = [e for e in keyword_hits if _entry_visible(e, user_role)]
        hidden = len(keyword_hits) - len(visible)

        if not visible:
            if hidden > 0:
                logger.info(
                    "制度参数命中但 ACL 全部拦截: keyword=%r role=%s hidden=%d",
                    keyword,
                    user_role,
                    hidden,
                )
                return {
                    "found": False,
                    "matches": [],
                    "filtered_by_acl": True,
                    "message": (
                        f"未找到与'{keyword}'相关且当前角色可见的制度参数"
                        f"（有 {hidden} 条因权限不可见）。如需访问请联系管理员提升权限或咨询人力资源部。"
                    ),
                }
            logger.info("制度参数查询未命中: keyword=%r", keyword)
            return {
                "found": False,
                "matches": [],
                "filtered_by_acl": False,
                "message": f"未找到与'{keyword}'相关的制度参数，可尝试换个关键词（如：办公用品、住宿标准、餐补）",
            }

        if hidden:
            logger.info(
                "制度参数查询命中 %d 条（另 ACL 过滤 %d）: keyword=%r role=%s",
                len(visible),
                hidden,
                keyword,
                user_role,
            )
        else:
            logger.info("制度参数查询命中 %d 条: keyword=%r", len(visible), keyword)

        return {
            "found": True,
            "matches": visible,
            "filtered_by_acl": hidden > 0,
            "message": "",
        }
