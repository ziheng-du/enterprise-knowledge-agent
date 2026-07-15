# 面试讲解稿：企业新员工知识助手（RAG + Agent）

面向 Agent / RAG 实习面试的 8–12 分钟讲解提纲。路径均对应本仓库当前代码。

---

## 1. 一分钟项目介绍

这是一个**企业内部知识助手**：员工用自然语言查制度，遇到「按入职日期算年假」这类问题会**自主调工具**，而不是只靠检索猜数字。

技术栈：LangGraph 状态图 + LangChain RAG + Chroma + FastAPI。  
工程上刻意做了：工具抽象（为 MCP 预留）、Hybrid 检索、多轮 SessionStore、上下文预算、模拟文档密级、黄金集评测。

---

## 2. 架构走读

```text
用户 → FastAPI /api/chat
     → SessionStore 加载历史/画像
     → LangGraph:
          rewrite（Query Rewrite）
          → router（rag / tool / both）
          → retrieve（Hybrid + ACL）和/或 agent ↔ execute_tools
          → generate
     → 写回 SessionStore → 返回 answer + sources + tool_calls + timings
```

核心文件：

- 图：[`app/agent/graph.py`](../app/agent/graph.py)
- 状态：[`app/agent/state.py`](../app/agent/state.py)
- 检索：[`app/rag/retriever.py`](../app/rag/retriever.py)
- 记忆：[`app/memory/`](../app/memory/)
- 工具：[`app/tools/base.py`](../app/tools/base.py) + [`registry.py`](../app/tools/registry.py)

---

## 3. 为何 SessionStore，不用 LangGraph Checkpointer

| | SessionStore（本项目） | Checkpointer |
|--|------------------------|--------------|
| 存什么 | 聊过什么、用户画像、摘要 | 图执行到哪一步的状态快照 |
| 适合 | 每轮完整跑完一张图的多轮问答 | HITL、断点续跑、长任务中断恢复 |

本项目每轮是 `START → … → END` 的完整 `invoke`，下一轮带着历史再跑一遍。需要的是**产品层对话记忆**，不是图内暂停。  
若以后做「敏感工具人工确认」，会在图内加 interrupt，那时再引入 Checkpointer，与 SessionStore **分工并存**。

---

## 4. Hybrid / RRF 解决什么问题

制度文本有大量专名和数字（「报销」「30天」）。纯向量偶发漏召回字面强匹配条款。  
做法：向量一路 + BM25 一路，用 **RRF** 融合（见 [`docs/retrieval.md`](retrieval.md)）。  
踩坑可讲：`rank_bm25` 分数可能为负，若用 `score<=0` 过滤会导致 BM25 路空召回。

---

## 5. 工具抽象与 MCP 预留（诚实边界）

- 所有工具继承 `BaseTool`（`name` / `description` / `input_schema` / `run`），经 `invoke()` 做校验与异常收敛。
- Agent **只通过 registry 取工具**，不 import 具体工具模块。
- **尚未**实现真实 MCP Server；面试时强调「接口稳定，协议适配可外包一层」，不要说成已经上线 MCP。

---

## 6. 兜底与评测：如何证明不瞎编

兜底：

- 工具参数失败 → `ToolMessage` 回喂重试，受 `MAX_TOOL_ROUNDS` 限制  
- 检索/工具皆空 → 固定「未找到」话术 / Prompt 禁编造  
- 路由/生成异常 → 降级或道歉，不把 500 抛给用户  

评测：

- 黄金集：[`data/eval/golden_set.jsonl`](../data/eval/golden_set.jsonl)  
- 脚本：`python scripts/run_eval.py --output docs/eval_report.md`  
- 报告：[`docs/eval_report.md`](eval_report.md)（按 rag/tool/refuse/multi_turn 分组）  
面试话术：用固定题集回归路由、工具选择与拒答，而不是「演示时碰巧答对」。

---

## 7. ACL 设计意图与局限

- 映射表：[`data/doc_access.json`](../data/doc_access.json)（`public` / `internal` / `confidential`）  
- 角色：`intern` ≤ public；`employee` ≤ internal；`admin` 全可见  
- 入库写 `access_level` 到 metadata；检索返回前按 `user_role` 过滤（[`app/rag/access_control.py`](../app/rag/access_control.py)）  

鉴权与局限（主动说）：

- Web：**工号+密码登录** → HMAC Bearer token；`/api/chat` 从 token 取角色，**不信任请求体 `role`**（[`app/auth/`](../app/auth/)）  
- 不是企业 SSO / LDAP；演示账号在 [`data/users.json`](../data/users.json)  
- 检索与 `policy_lookup` 按同一密级过滤（角色由 Agent `execute_tools` 注入 context，不由 LLM 传参伪造）  
- CLI 联调仍可用 `agent_demo.py --role`；改密级后需重新 `ingest --rebuild`；扫描件 PDF 无 OCR  

演示句：用实习生账号登录后问报销时限 / 办公用品上限 → 检索与工具均不可见 internal；管理员问高管薪酬 → 可命中 confidential。多格式：`generate_sample_docs.py` 生成 docx/pdf 后入库。
---

## 8. 三个可讲的踩坑

1. **BM25 负分**：过滤 `<=0` 导致 hybrid 退化；改为按排序取 top_k。  
2. **伪流式**：`/api/chat/stream` 是 Agent 跑完再按块推 SSE，不是 generate 真 token 流；面试别说成「已实现流式生成」。  
3. **评测非确定性**：路由 LLM 有波动，期望 route 用集合（如 `rag|both`）；报告解读时说明容差。

---

## 简历一行（可直接用）

> 企业知识库 Agent：LangGraph 分诊 + Hybrid RAG + 可插拔工具；SessionStore 多轮记忆、上下文预算、模拟文档密级与黄金集评测。
