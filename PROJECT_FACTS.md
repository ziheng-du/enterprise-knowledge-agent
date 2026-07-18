# 项目事实提取报告

> 用途：简历/面试前核实用。只陈述代码与文档中真实存在的内容，不写简历文案。  
> 生成日期：2026-07-15

---

## 一、评估方法论澄清

### 1. `scripts/run_eval.py` 的评分机制

**纯规则匹配，不是 LLM-as-Judge。**

- 会调用 `run_agent`（内部会调 LLM 做路由/生成），但**判定对错不经过 LLM 打分**。
- 核心逻辑在 `eval_single` / `eval_multi_turn` / `_check_keywords`（`scripts/run_eval.py`）：
  - `expect_route`：`result.route` 是否落在允许列表
  - `expect_tool`：工具名是否出现在 `tool_calls`
  - `expect_refuse`：答案是否包含字符串 `"未找到"`
  - `expect_keywords`：答案是否**全部包含**列出的关键词（`all(k in answer)`）
  - `expect_sources_any`：来源拼接串是否命中任一子串
- 文件头文档字符串写明：「量化路由 / 工具 / 拒答 / 关键词命中」

### 2. 是否有 LLM-as-Judge 实现

- **无。** 全仓库未找到 `judge` / `LLM-as-Judge` 等实现。
- 无对应文件，无跑出过的 Judge 数据。

### 3. 明确结论

**评测为规则匹配式功能测试，不涉及 LLM 打分。**  
（简历勿写「LLM 评判 / LLM-as-Judge / 相关性打分」。）

---

## 二、评测详细数据

### 1. 23 条用例按类型分布

文件：`data/eval/golden_set.jsonl`

在线评测 23 条（另有 1 条 `offline` 仅结构校验，不计入 23）：

| type | 条数 |
|------|------|
| rag | 10 |
| tool | 5 |
| refuse | 4 |
| both | 2 |
| multi_turn | 2 |

### 2. 失败用例

来源：`docs/eval_report.md`（2026-07-16 扩充语料后全量）

- `refuse_weather` / `refuse_lottery`：约定拒答关键词「未找到」未出现（route 曾落到 tool）
- `rag_supplementary_medical`：当时 expect_route 未含 tool，且要求来源；题面已改为允许 tool（后续重跑可消除该类失败）

### 3. `docs/eval_report.md` 完整内容

见仓库文件 [`docs/eval_report.md`](docs/eval_report.md)（生成时间：2026-07-16 03:23 UTC）。

要点摘要：

- 通过：**34/37**（91.9%）
- both：2/2；multi_turn：3/3；rag：18/19；refuse：3/5；tool：8/8
- 黄金集条目约 **38** 行（含 1 条 offline）；在线评测 **37** 条

### 4. 是否覆盖文档密级 ACL

- **黄金集未覆盖。** `golden_set.jsonl` 无角色/密级字段，无「实习生看不到内部文档」类用例。
- ACL 由单元测试覆盖：
  - `tests/test_access_control.py`（mock 检索结果按角色过滤）
  - `tests/test_tools.py::test_intern_cannot_bypass_acl_via_tool`
- 黄金集端到端实测 ACL：**暂无实测数据**

---

## 三、Hybrid 检索（RRF）相关数据

### 1. `docs/chunking_comparison.md` 要点

文件：[`docs/chunking_comparison.md`](docs/chunking_comparison.md)

- 时间：2026-07-07
- 对比：`fixed` vs `recursive`（`chunk_size=500, overlap=50`，均 8 块）
- Embedding：`BAAI/bge-small-zh-v1.5`
- top-1 分数：

| 查询 | fixed | recursive |
|------|-------|-----------|
| 报销多久内要提交 | 0.6898 报销制度.md | 0.6962 报销制度.md |
| 入职3年年假几天 | 0.7284 请假与年假政策.md | 0.7311 请假与年假政策.md |
| 迟到超过30分钟怎么处理 | 0.6160 员工手册.md | 0.5679 员工手册.md |

- 结论：默认 `CHUNKING_STRATEGY=recursive`

### 2. Hybrid vs 纯向量对比

- 代码可切换（`RETRIEVAL_MODE=hybrid|vector`，见 `app/config.py`、`docs/retrieval.md`）
- 对照实验：[`docs/retrieval_comparison.md`](docs/retrieval_comparison.md)（`scripts/compare_retrieval_modes.py`）
- 扩充语料后一次实测（**21** 条查询，top_k=4，阈值 0.3，语料 **44** 块）：
  - Top-1 来源正确 / Hit@4：**两侧均为 21/21（100%）**
  - Top-1 关键词命中：**两侧均为 18/19（94.7%）**（失败同为「团建花絮噪声」题：top-1 命中非正式文稿、正文无「30」）
  - Top-4 平均独特来源数：纯向量 **2.95**，Hybrid **3.05**
- 解读：来源级仍易触顶；差异更多体现在 Top-K 来源多样性与噪声题的关键词落点；勿写「Hybrid 提升 XX%」

### 3. RRF 参数

- `HYBRID_RRF_K` 默认 **60**（`app/config.py`）
- `HYBRID_BM25_TOP_K` 默认 **8**
- 融合公式：`score += 1.0 / (k + rank)`（`app/rag/bm25_index.py` → `reciprocal_rank_fusion()`）
- **无单独 BM25/向量权重系数**（两路等权按排名融合）

---

## 四、查询改写（Query Rewrite）细节

### 1. 如何判断「依赖上文的追问」

- **没有单独的「是否追问」分类器/规则。**
- `app/memory/query_rewrite.py` → `rewrite_query()`：
  - 无历史且无摘要 → 直接返回原问题
  - 有历史 → **一律调 LLM 改写**（Prompt 约定：已完整则可原样输出）
- 节点：`app/agent/graph.py` → `rewrite_node`
- Prompt：`app/agent/prompts.py` → `QUERY_REWRITE_*`

### 2. 改写准确率/成功率

- 仅有 mock 单元测试（`tests/test_query_rewrite.py`）
- **无批量真实改写准确率数据 → 暂无实测数据**

### 3. 改写失败时的表现

- 异常 / 空输出 / 长度 >500 → **回退原问题**（代码明确）
- 回退后检索效果对比：**暂无实测数据**

---

## 五、鉴权与文档密级 ACL 细节

### 1. 密码存储

- **哈希存储**（非明文）
- 算法：`PBKDF2-HMAC-SHA256`，迭代 **100_000**
- 文件：`app/auth/users.py`；盐+哈希在 `data/users.json`

### 2. token HMAC 实现

- **未用 PyJWT**
- 标准库：`hmac` + `hashlib.sha256`
- 格式：`payload_b64.signature_b64`
- 文件：`app/auth/tokens.py`

### 3. 文档密级过滤生效时机

- **检索完成后再过滤结果**（非检索前过滤候选库）
- `Retriever.retrieve()` 先 vector/hybrid，再 `_filter_by_access()`
- 文件：`app/rag/retriever.py`
- 另：`policy_lookup` 工具结果也会按角色过滤（`app/tools/policy_lookup.py`）

### 4. 实现耗时 / 是否原规划

- `PROJECT_SPEC.md` 原六阶段**未要求**登录鉴权；局限示例写「未做权限控制」
- 文档密级 ACL（按角色过滤）先有；**工号密码登录是追加**（2026-07-15：从「前端下拉传 role」改为 token）
- **精确工时：代码/文档未记录 → 无法给出天数数字**

---

## 六、SessionStore 记忆机制细节

### 1. 底层存储与表结构

- **SQLite**（`app/memory/session_store.py`）
- `sessions`：`session_id, summary, profile_json, created_at, updated_at`
- `messages`：`id, session_id, role, content, created_at`

### 2. 为何不用 LangGraph Checkpointer

文档真实表述（`docs/interview_talk.md`）：

- SessionStore：产品层「聊过什么 / 画像 / 摘要」
- Checkpointer：图执行断点、HITL、中断恢复
- 本项目每轮是完整 `START→END` invoke，需要的是对话记忆而非图内暂停
- 若以后做敏感工具人工确认，再引入 Checkpointer，与 SessionStore **分工并存**

### 3. 用户画像（入职日期）如何提取

- **非**从对话自由文本做 LLM 抽取
- 来源：
  1. 登录用户花名册 `hire_date` → 写入会话画像（`app/api/routes.py` `_prepare_session_for_user`）
  2. 成功调用 `leave_calculator` 时，从**工具参数** `hire_date` 写入（`app/agent/graph.py`）——规则提取工具 args

---

## 七、代码规模与工程规范

### 1. Python 代码行数

- 排除 `.venv` / `venv` / `__pycache__` / `.git` 等后：**59 个 `.py` 文件，5719 行**（2026-07-15 本机实测）

### 2. pytest

环境：`enterprise-knowledge-agent`

- `pytest --collect-only`：**77 tests collected**
- `pytest -q`：**77 passed**（约 11.51s）

### 3. Docker

- 仓库无 `Dockerfile` / `docker-compose` → **未做 Docker 化**

### 4. `/api/chat/stream` 伪流式实现

文件：`app/api/routes.py`

1. 先 `await asyncio.to_thread(run_agent, ...)` **整段跑完**
2. SSE `event=meta` 推元数据
3. 再把 `answer` 按 **每 8 个字符**切块，以 `event=token` 推送，块间 `asyncio.sleep(0.02)`
4. 最后 `event=done`

→ 非 generate 真 token 流

---

## 八、开发时间线

### 1. 总天数

- Git 仅约 2026-07-15 两次提交，**无法从 git 还原分阶段天数**
- 可核实时间锚点：
  - `docs/chunking_comparison.md`：2026-07-07
  - `docs/eval_report.md`：2026-07-14
  - 登录鉴权追加：2026-07-15
  - 相关 Cursor 会话最早痕迹约 **2026-07-04**
- **「总共花了多少天」无工时日志 → 暂无权威实测数字**

### 2. 功能：原计划还是追加

| 功能 | 相对 `PROJECT_SPEC` 六阶段 | 说明 |
|------|---------------------------|------|
| Hybrid+RRF | **追加** | 原规格未列 BM25/Hybrid |
| Query Rewrite / SessionStore | **追加** | 原规格局限示例写「未做多轮对话记忆」 |
| 文档密级 ACL | **追加**（相对六阶段） | 原规格写「未做权限控制」；先于登录存在 |
| 工号密码鉴权 | **再追加** | 2026-07-15：把「请求体 role」换成 token |

各追加功能**精确花费天数：暂无记录**。
