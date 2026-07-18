# 黄金集评测报告

生成时间：2026-07-16 03:23 UTC

## 如何复现

```bash
python scripts/run_eval.py --offline          # 无 Key：仅结构校验
python scripts/run_eval.py --output docs/eval_report.md  # 有 Key：全量并写报告
```

## 总体结果

- 通过：**34/37**（91.9%）

## 按题型分组

| type | 通过 | 总数 | 通过率 |
|------|------|------|--------|
| both | 2 | 2 | 100.0% |
| multi_turn | 3 | 3 | 100.0% |
| rag | 18 | 19 | 94.7% |
| refuse | 3 | 5 | 60.0% |
| tool | 8 | 8 | 100.0% |

## 失败用例

- `refuse_weather`（refuse）：expected refuse phrase 未找到; missing keywords ['未找到']
- `refuse_lottery`（refuse）：expected refuse phrase 未找到; missing keywords ['未找到']
- `rag_supplementary_medical`（rag）：route=tool not in ['rag', 'both']; sources [] miss any of ['福利', '保险']

## 逐条结果

| id | type | 结果 | route |
|----|------|------|-------|
| rag_expense_deadline | rag | PASS | rag |
| rag_travel | rag | PASS | rag |
| tool_leave | tool | PASS | tool |
| tool_policy_office | tool | PASS | tool |
| both_leave_policy | both | PASS | both |
| refuse_weather | refuse | FAIL | tool |
| refuse_stock | refuse | PASS | rag |
| rag_attendance | rag | PASS | rag |
| rag_leave_apply | rag | PASS | rag |
| tool_leave_2022 | tool | PASS | tool |
| multi_turn_followup | multi_turn | PASS | rag |
| rag_reimburse_flow | rag | PASS | rag |
| rag_travel_transport | rag | PASS | rag |
| rag_expense_invoice | rag | PASS | rag |
| tool_leave_missing_date | tool | PASS | rag |
| tool_policy_meal | tool | PASS | tool |
| refuse_lottery | refuse | FAIL | tool |
| rag_handbook_onboarding | rag | PASS | rag |
| multi_turn_leave_then_policy | multi_turn | PASS | rag |
| rag_leave_carryover | rag | PASS | rag |
| both_expense_and_calc | both | PASS | tool |
| rag_hybrid_30days | rag | PASS | rag |
| refuse_crypto | refuse | PASS | rag |
| rag_obsolete_45_days | rag | PASS | rag |
| rag_comp_leave_deadline | rag | PASS | rag |
| rag_supplementary_medical | rag | FAIL | tool |
| rag_late_30_minutes | rag | PASS | rag |
| rag_travel_reimburse_joint | rag | PASS | rag |
| rag_noise_team_building | rag | PASS | rag |
| rag_remote_days | rag | PASS | rag |
| rag_password_length | rag | PASS | rag |
| rag_breach_report_hours | rag | PASS | rag |
| tool_policy_comp_leave | tool | PASS | tool |
| tool_policy_remote | tool | PASS | tool |
| tool_policy_medical_wait | tool | PASS | tool |
| multi_turn_30_disambiguate | multi_turn | PASS | rag |
| refuse_horoscope | refuse | PASS | rag |

## 指标解读（面试可用）

- **rag**：制度问答是否命中正确文档/关键词。
- **tool**：是否调用了预期工具（如年假计算）。
- **refuse**：知识库外问题是否如实「未找到」而非编造。
- **multi_turn**：同 session 追问是否仍答对关键信息。
- 路由允许一定容差（如 rag/both），因分诊 LLM 非确定性。
