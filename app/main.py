"""FastAPI 应用入口。

启动方式（在项目根目录）：
    uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

- /api/* ：业务接口（见 api/routes.py）
- /docs  ：Swagger 交互式文档
- /      ：静态前端聊天页（frontend/）
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from app.api.auth_routes import router as auth_router
from app.api.routes import router as api_router
from app.config import PROJECT_ROOT, get_settings
from app.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)

FRONTEND_DIR = PROJECT_ROOT / "frontend"


class NoCacheFrontendMiddleware(BaseHTTPMiddleware):
    """禁止缓存 HTML/JS/CSS，避免浏览器继续使用登录改造前的旧前端。

    开发/演示场景下静态页更新频繁；若 Edge 等浏览器缓存了旧的 index.html
    （无登录页、请求不带 Bearer），会出现「看不到登录页 + 聊天 401」。
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        path = request.url.path
        if path.startswith("/api") or path.startswith("/docs") or path.startswith("/openapi"):
            return response
        if (
            path == "/"
            or path.endswith(".html")
            or path.endswith(".js")
            or path.endswith(".css")
        ):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """应用生命周期：启动时初始化日志；提醒开发默认鉴权密钥。"""
    setup_logging()
    settings = get_settings()
    if settings.auth_secret_key == "dev-only-change-me-eka-auth-secret":
        logger.warning(
            "正在使用默认 AUTH_SECRET_KEY，仅适合本地开发；生产环境请在 .env 中覆盖"
        )
    logger.info("企业新员工知识助手服务启动")
    yield
    logger.info("企业新员工知识助手服务关闭")


app = FastAPI(
    title="企业新员工知识助手",
    description="融合 RAG 检索与 Agent 工具调用的企业内部知识问答服务",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(NoCacheFrontendMiddleware)
app.include_router(auth_router)
app.include_router(api_router)

# 静态前端挂载到根路径；API 路由已先注册，不会被静态文件覆盖
if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
else:
    logger.warning("前端目录不存在: %s，跳过静态文件挂载", FRONTEND_DIR)
