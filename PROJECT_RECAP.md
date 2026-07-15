# 项目复盘笔记：企业新员工知识助手（RAG + Agent）

> 用途：自己重新看懂代码 + 面试前速查。  
> 本笔记按目录列出「每个文件干什么」。读法建议：先扫本文件建立地图，再按路径打开源码对照。  
> 生成日期：2026-07-14（随代码演进需自行增补）。

---

## 0. 项目一句话

企业内部知识问答：制度类问题走 **RAG 检索**；「按入职日期算年假」等走 **Agent 调工具**；多轮靠 **SessionStore + Query Rewrite**，不是 LangGraph Checkpointer。

请求主链路：`前端/Swagger → /api/chat → run_agent() → LangGraph 状态图 → 写回会话 → 返回答案`

---

## 1. 根目录配置与说明文档

| 文件 | 功能 |
|------|------|
| `README.md` | 项目对外说明书：能力列表、架构图、快速开始、API、已知局限 |
| `PROJECT_SPEC.md` | 最初开发规格/提示词：技术栈、目录、阶段要求（「该做成什么样」） |
| `.cursorrules` | Cursor / AI 协作约束：架构硬性要求、禁止事项、工作方式 |
| `requirements.txt` | pip 依赖清单（LangGraph、LangChain、FastAPI、Chroma 等） |
| `environment.yml` | conda 环境定义（环境名 `enterprise-knowledge-agent`） |
| `.env.example` | 环境变量模板（API Key、检索模式、预算等）；不含真实密钥 |
| `.env` | 本地真实配置（**含密钥，勿提交/勿在笔记里抄 Key**） |
| `.gitignore` | Git 忽略规则（`.env`、缓存、向量库等） |
| `PROJECT_RECAP.md` | 本文件：复盘笔记与文件功能速查 |

---

## 2. `app/` —— 应用核心代码

### 2.1 入口与基建

| 文件 | 功能 |
|------|------|
| `app/__init__.py` | 将 `app` 声明为 Python 包 |
| `app/main.py` | FastAPI 应用入口：生命周期、挂载 `/api`、静态服务 `frontend/` |
| `app/config.py` | 全项目配置入口：从 `.env` 读 LLM、检索、工具轮次、上下文预算等 |
| `app/llm.py` | 共享 Chat 模型工厂（OpenAI 兼容客户端，如 DeepSeek） |

### 2.2 `app/api/` —— HTTP 接口层

| 文件 | 功能 |
|------|------|
| `app/api/__init__.py` | API 包标识 |
| `app/api/routes.py` | 路由：`/api/health`、`/api/chat`、`/api/chat/stream`（伪流式）、`/api/ingest`；Pydantic 请求/响应；业务交给 `run_agent` / `ingest` |

### 2.3 `app/agent/` —— Agent 状态图（面试最核心）

| 文件 | 功能 |
|------|------|
| `app/agent/__init__.py` | Agent 包标识 |
| `app/agent/state.py` | `AgentState` TypedDict：问题、路由、检索结果、工具消息、轮次、耗时、最终答案等 |
| `app/agent/prompts.py` | 全部 Prompt 模板（路由 / 工具决策 / 最终回答 / 拒答约束） |
| `app/agent/graph.py` | LangGraph 图：`rewrite → router → retrieve/agent ↔ execute_tools → generate`；统一入口 `run_agent()` |

`graph.py` 内关键符号（便于对照）：

- `rewrite_node`：多轮指代消解，写 `search_query`
- `route_node`：分诊 `rag` / `tool` / `both`
- `retrieve_node`：调用独立 Retriever
- `agent_node`：LLM 绑定工具并决策
- `execute_tools_node`：经 registry 执行工具并回馈
- `generate_node`：综合生成 `final_answer`
- `build_agent_graph()`：编译状态图
- `run_agent()`：加载会话 → invoke 图 → 持久化 → 返回 `AgentResult`

### 2.4 `app/tools/` —— 工具层（为 MCP 预留抽象）

| 文件 | 功能 |
|------|------|
| `app/tools/__init__.py` | 工具包包说明 |
| `app/tools/base.py` | `BaseTool` / `ToolResult`：`name/description/input_schema/run`；`invoke()` 校验并收敛异常 |
| `app/tools/registry.py` | 显式注册表：`get_tool` / `get_all_tools`；**Agent 只应依赖本文件取工具** |
| `app/tools/leave_calculator.py` | 工具1：按入职日期计算年假天数（模拟规则） |
| `app/tools/policy_lookup.py` | 工具2：查 `policy_params.json` 结构化制度参数；按角色过滤密级 |

### 2.5 `app/rag/` —— 检索与入库（与 Agent 解耦）

| 文件 | 功能 |
|------|------|
| `app/rag/__init__.py` | RAG 包标识 |
| `app/rag/document_loader.py` | 从 `raw_docs` 加载 md/txt/pdf/docx → LangChain Document |
| `app/rag/chunking.py` | 文本切分：`fixed` / `recursive`，可配 size/overlap，保留元数据 |
| `app/rag/embedding.py` | Embedding 模型封装（如本地 bge 中文） |
| `app/rag/vector_store.py` | Chroma 本地向量库：增删查、持久化路径管理 |
| `app/rag/bm25_index.py` | BM25 关键词索引 + RRF 融合（Hybrid 检索） |
| `app/rag/retriever.py` | 统一检索入口：`vector` / `hybrid` + 角色 ACL；返回 `RetrievalResult` |
| `app/rag/access_control.py` | 模拟密级：角色 `intern/employee/admin` ↔ 文档 `public/internal/confidential` |
| `app/rag/ingest_service.py` | 入库编排：load → split → 向量写入 → 刷新 BM25 |
| `app/rag/qa_chain.py` | 纯 RAG 问答链（检索+生成，不走 Agent 工具循环）；含 `format_context` |

### 2.6 `app/memory/` —— 多轮记忆与上下文管控

| 文件 | 功能 |
|------|------|
| `app/memory/__init__.py` | 记忆包说明 |
| `app/memory/session_store.py` | SQLite：会话历史、轻量用户画像、摘要；跨轮记忆核心 |
| `app/memory/query_rewrite.py` | Query Rewrite：把「那报销上限呢」改成可独立检索的 query |
| `app/memory/context_budget.py` | 字符预算：截断历史 / 检索片段 / 工具结果，防 Prompt 过长 |
| `app/memory/history_summary.py` | 会话过长时对更早轮次做 LLM 摘要压缩 |

### 2.7 `app/utils/` —— 工具函数

| 文件 | 功能 |
|------|------|
| `app/utils/__init__.py` | utils 包标识 |
| `app/utils/logger.py` | 标准库 logging 的初始化与 `get_logger` |

---

## 3. `frontend/` —— 演示前端

| 文件 | 功能 |
|------|------|
| `frontend/index.html` | 聊天页结构：输入框、角色选择、消息区 |
| `frontend/style.css` | 页面样式 |
| `frontend/script.js` | 调 `POST /api/chat`；localStorage 持久化 `session_id` 与角色；展示来源/工具标签 |

---

## 4. `data/` —— 业务数据与评测集

| 文件 | 功能 |
|------|------|
| `data/raw_docs/员工手册.md` | 模拟制度：入职/出勤等规章，供 RAG 检索 |
| `data/raw_docs/报销制度.md` | 模拟报销规则文档 |
| `data/raw_docs/差旅政策.md` | 模拟差旅规则文档 |
| `data/raw_docs/请假与年假政策.md` | 模拟请假/年假政策（与计算器规则对齐） |
| `data/raw_docs/高管薪酬指引.md` | confidential 样例，测密级 ACL |
| `data/raw_docs/IT设备领用须知.docx` | docx 格式入库演示样本 |
| `data/raw_docs/消防与安全须知.pdf` | pdf 格式入库演示样本（可抽取文本） |
| `data/doc_access.json` | 文件名 → 密级映射（入库时写入 metadata） |
| `data/policy_params.json` | 结构化制度参数，供 `policy_lookup` 查询（非向量库） |
| `data/eval/golden_set.jsonl` | 评测黄金集：固定问答/路由/拒答等多类型用例 |

> 说明：运行入库后还会有 `data/vector_db/` 等 Chroma 持久化目录；属生成物，一般不入手工编辑。会话 SQLite 路径由配置决定，同样多为运行时生成。

---

## 5. `scripts/` —— 命令行脚本

| 文件 | 功能 |
|------|------|
| `scripts/ingest.py` | CLI：把 `raw_docs` 入库（可 `--rebuild` / 选切分策略） |
| `scripts/search_demo.py` | CLI：只测检索，打印命中片段与分数 |
| `scripts/agent_demo.py` | CLI：跑一轮（或多轮带 `--session-id`）完整 Agent |
| `scripts/run_eval.py` | 对黄金集评测；可 `--offline` 或写出 `docs/eval_report.md` |
| `scripts/generate_sample_docs.py` | 生成演示用 docx/pdf 样本文档 |

---

## 6. `tests/` —— 自动化测试

| 文件 | 功能 |
|------|------|
| `tests/__init__.py` | 测试包标识 |
| `tests/test_config.py` | 配置加载与默认值相关测试 |
| `tests/test_chunking.py` | 切分策略与元数据相关测试 |
| `tests/test_tools.py` | 工具层（计算器 / 制度查询）单元测试 |
| `tests/test_bm25_hybrid.py` | BM25 / Hybrid / RRF 相关测试 |
| `tests/test_access_control.py` | 角色与文档密级可见性测试 |
| `tests/test_agent_tools_node.py` | Agent 工具节点行为相关测试 |
| `tests/test_api.py` | FastAPI 接口层测试 |
| `tests/test_session_store.py` | 会话存储读写测试 |
| `tests/test_query_rewrite.py` | Query Rewrite 相关测试 |
| `tests/test_context_budget.py` | 上下文预算截断测试 |
| `tests/test_history_summary.py` | 历史摘要相关测试 |
| `tests/test_eval_offline.py` | 评测黄金集结构（离线）校验 |
| `tests/test_eval_report.py` | 评测报告生成相关测试 |

---

## 7. `docs/` —— 专题文档

| 文件 | 功能 |
|------|------|
| `docs/retrieval.md` | Hybrid（向量+BM25+RRF）原理与配置说明 |
| `docs/chunking_comparison.md` | fixed vs recursive 切分对比实验记录（选型依据） |
| `docs/eval_report.md` | 黄金集评测报告（可由 `run_eval.py` 刷新） |
| `docs/interview_talk.md` | 8–12 分钟面试讲解提纲（架构/记忆/Hybrid/MCP/兜底） |

---

## 8. 建议阅读顺序（复盘用）

1. 本笔记 §1–§2 扫一遍，知道「文件落在哪一层」  
2. `app/agent/state.py` → 知道状态图上有哪些「数据格子」  
3. `app/agent/graph.py` 的 `build_agent_graph` / `run_agent` → 串起整条链路  
4. `app/tools/base.py` + `registry.py` → 工具抽象（面试必问）  
5. `app/rag/retriever.py` + `docs/retrieval.md` → Hybrid  
6. `app/memory/session_store.py` + `query_rewrite.py` → 多轮为何能接上  
7. `app/api/routes.py` + `frontend/script.js` → 对外表现  
8. 用 `docs/interview_talk.md` 对着讲 8 分钟做验收  

---

## 9. 后续待补（复盘笔记预留章节）

- [ ] 状态图三路径（rag / tool / both）逐步推演  
- [ ] SessionStore vs Checkpointer 对比话术  
- [ ] 工具抽象与 MCP「做到哪、没做到哪」  
- [ ] 兜底策略清单 + 常见面试拷问与标准答法  

（本章节可在下次复盘时继续填充。）
