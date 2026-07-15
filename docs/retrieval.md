# 检索策略说明

本项目检索与 Agent 解耦，统一入口为 `app/rag/retriever.py`。

## 模式

| 模式 | 配置 `RETRIEVAL_MODE` | 行为 |
|------|----------------------|------|
| 纯向量 | `vector` | Chroma 余弦相似度 + `RETRIEVAL_SCORE_THRESHOLD` 过滤 |
| Hybrid（默认） | `hybrid` | 向量路 + BM25 关键词路，经 **RRF**（`HYBRID_RRF_K`）融合后取 `top_k` |

## 为何加 BM25

制度文本含大量专名与数字（如「报销」「30天」）。纯语义向量偶发漏召回字面匹配强的条款；BM25 补充关键词信号。

## 索引刷新

`ingest`（CLI / `POST /api/ingest`）在写入 Chroma 后调用 `refresh_bm25_from_vector_store`，保证 hybrid 语料与向量库一致。

## 分词

未引入 jieba：英文/数字按词，中文按字 + bigram（见 `app/rag/bm25_index.py`），依赖面更小，适合作品集本地复现。
