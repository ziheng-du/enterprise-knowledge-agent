"""工具抽象基类：所有工具的统一接口规范。

设计意图（架构硬性约束，请勿绕过）：
- 每个工具必须声明 name（工具名）、description（供 LLM 判断何时调用的
  自然语言描述）、input_schema（Pydantic 输入参数模型）并实现 run()。
- Agent 决策逻辑只通过 tools/registry.py 动态获取工具，不直接 import
  具体工具函数。
- 未来某个工具要改造成 MCP Server 暴露给外部时，协议适配层只需消费
  `name / description / input_schema / invoke()` 这四个稳定接口
  （name/description/schema 映射到 MCP 的工具声明，invoke 映射到
  工具调用请求），`run()` 内的业务逻辑零改动。

异常处理策略（模板方法模式）：
- 子类的 run() 只写业务逻辑，可以直接抛业务异常
- invoke() 作为统一入口收敛参数校验与异常捕获，永远返回结构化的
  ToolResult 而不向上抛异常 —— 阶段四的 Agent 拿到 ToolResult.error
  后自行决定重试或降级，避免一次工具调用失败导致整个服务崩溃
"""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from app.utils.logger import get_logger

logger = get_logger(__name__)


class ToolResult(BaseModel):
    """工具调用的统一结构化返回。

    success=False 表示"调用失败"（参数校验不通过或执行抛异常），
    error 中带可读的失败原因；业务层面的"查询无结果"不算失败，
    应以 success=True + data 中说明的方式返回。
    """

    success: bool = Field(description="调用是否成功")
    data: Any | None = Field(default=None, description="成功时的结构化结果")
    error: str | None = Field(default=None, description="失败时的可读错误信息")


class BaseTool(ABC):
    """所有工具的抽象基类。

    子类必须定义三个类属性并实现 run()：
    - name: 工具唯一标识（snake_case）
    - description: 自然语言描述，供 LLM 判断何时该调用此工具，
      需写清楚工具能做什么、输入是什么
    - input_schema: Pydantic BaseModel 子类，定义结构化输入参数
    """

    name: str
    description: str
    input_schema: type[BaseModel]

    @abstractmethod
    def run(self, params: BaseModel) -> Any:
        """执行工具的业务逻辑（子类实现）。

        Args:
            params: 已通过 input_schema 校验的参数模型实例。

        Returns:
            任意可 JSON 序列化的结构化结果。

        Raises:
            业务异常可直接抛出，由 invoke() 统一捕获转为 ToolResult.error。
        """

    def invoke(self, raw_args: dict, *, context: dict | None = None) -> ToolResult:
        """工具调用的统一入口：参数校验 -> 执行 -> 结构化返回。

        这是 Agent（以及未来的 MCP 适配层）应当调用的方法，
        保证任何失败都以 ToolResult 形式返回而不是异常冒泡。

        Args:
            raw_args: 原始参数字典（通常由 LLM 生成，可能不合法）。
            context: 可选运行时上下文（如 user_role），由 Agent 注入，
                不进入 LLM 可见的 input_schema，避免模型伪造权限。

        Returns:
            ToolResult：成功时 data 为 run() 的返回值；
            参数校验失败或执行异常时 success=False 且 error 带可读原因。
        """
        # 第一步：参数校验。LLM 生成的参数可能缺字段/类型错误，
        # 转成带字段位置的可读错误信息，供 Agent 决定重试或降级
        try:
            params = self.input_schema(**raw_args)
        except ValidationError as exc:
            error_detail = "; ".join(
                f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}" for err in exc.errors()
            )
            logger.warning("工具 %s 参数校验失败: %s (raw_args=%r)", self.name, error_detail, raw_args)
            return ToolResult(success=False, error=f"参数校验失败: {error_detail}")

        # 将上下文挂到实例上供 run() 读取（模板方法，不污染 input_schema）
        self._invoke_context: dict = dict(context or {})

        # 第二步：执行业务逻辑。捕获所有异常防止冒泡导致服务崩溃
        try:
            data = self.run(params)
        except Exception as exc:
            logger.exception("工具 %s 执行失败: params=%r", self.name, params)
            return ToolResult(success=False, error=f"工具执行失败: {exc}")
        finally:
            self._invoke_context = {}

        logger.info("工具 %s 调用成功", self.name)
        return ToolResult(success=True, data=data)
