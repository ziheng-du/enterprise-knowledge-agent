# =============================================================================
# 企业新员工知识助手 — 应用镜像
#
# 生活比喻：这份文件是「标准厨房操作手册」。
# 任何人按同一手册做菜（build），端出来的成品（image）味道一致；
# 客人点餐时再起一份热菜（container），互不影响。
#
# 企业里为什么要 Docker：
# - 新人电脑不再「我这边能跑你那边不行」
# - CI 打出同一镜像，测试 / 预发 / 生产同一份制品
# - 部署只关心「跑哪个镜像 + 什么环境变量」，不关心本机装了啥
# =============================================================================

FROM python:3.11-slim-bookworm

# 容器内进程日志立刻刷到 docker logs，不要卡在缓冲里
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    # HuggingFace / sentence-transformers 模型缓存目录（配合 volume 持久化）
    HF_HOME=/home/appuser/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/appuser/.cache/huggingface

WORKDIR /app

# 系统依赖：部分 Python 包编译 / 证书校验需要
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ---- 依赖层：先只复制 requirements，利用 Docker 层缓存 ----
# 代码天天改，依赖很少改。分开 COPY 后，改业务代码时不必每次重装 torch。
COPY requirements.txt .

# 服务端推理一般用 CPU 版 torch，镜像更小，也避免误拉 CUDA 大包（企业常见做法）
RUN pip install --no-cache-dir \
        torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# ---- 业务代码与演示数据 ----
COPY app/ ./app/
COPY frontend/ ./frontend/
COPY scripts/ ./scripts/
COPY data/raw_docs/ ./data/raw_docs/
COPY data/doc_access.json ./data/doc_access.json
COPY data/policy_params.json ./data/policy_params.json
COPY data/users.json ./data/users.json
COPY data/eval/ ./data/eval/

# 运行期可写目录：向量库、会话库、模型缓存（compose 里挂到 data/runtime）
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/data/runtime/vector_db /home/appuser/.cache/huggingface \
    && chown -R appuser:appuser /app /home/appuser/.cache

USER appuser

EXPOSE 8000

# 健康检查：编排 / 云平台用它判断容器是否就绪（K8s / Compose 都认）
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/openapi.json >/dev/null || exit 1

# 生产容器一般不加 --reload；热重载是开发机的事
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
