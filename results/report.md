# PII Shield evaluation report

Detection run: 2026-10-05T08:54:12+00:00.

## Detection (gold set)

202 documents, 330 entities, 50 documents without PII. Thresholds and allow-list: policy `support-chat`.

| Configuration | Precision | Recall | F1 strict | F1 partial | Any-type recall | Clean docs flagged | Latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| Patterns + validators | 0.98 / 0.99 | 0.42 / 0.43 | 0.59 | 0.60 | 42.7% | 2/50 | 0 / 1 ms |
| Patterns + NER | 0.93 / 0.94 | 0.97 / 0.98 | 0.95 | 0.96 | 98.2% | 7/50 | 67 / 80 ms |

Precision and recall cells: strict / partial. Latency: mean / p95 per document.

Held-out split only (36 documents written after the detectors were frozen):

| Configuration | Precision | Recall | F1 partial | F1 strict | Any-type recall |
|---|---:|---:|---:|---:|---:|
| Patterns + validators | 1.00 | 0.39 | 0.56 | 0.56 | 39.1% |
| Patterns + NER | 0.97 | 0.96 | 0.96 | 0.95 | 97.1% |

| Entity | Gold | Patterns + validators P / R / F1 | Patterns + NER P / R / F1 |
|---|---:|---:|---:|
| PERSON | 139 | – / 0.00 / – | 0.98 / 0.99 / 0.98 |
| PHONE | 32 | 0.94 / 1.00 / 0.97 | 0.94 / 1.00 / 0.97 |
| EMAIL | 31 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| ORGANIZATION | 24 | – / 0.00 / – | 0.67 / 1.00 / 0.80 |
| ADDRESS | 23 | – / 0.00 / – | 0.91 / 0.91 / 0.91 |
| SECRET | 16 | 1.00 / 0.88 / 0.93 | 1.00 / 0.88 / 0.93 |
| DATE_OF_BIRTH | 12 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| IP_ADDRESS | 12 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| CREDIT_CARD | 10 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| IBAN | 9 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| URL | 8 | 1.00 / 0.88 / 0.93 | 1.00 / 0.88 / 0.93 |
| NATIONAL_ID | 8 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| US_SSN | 6 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |

Per-type cells use partial matching (same type, overlapping span).

### With the LLM detector: every 3rd document (68 documents, 126 entities), all configurations on the same subset

| Configuration | Precision | Recall | F1 partial | F1 strict | Any-type recall | Clean docs flagged | Latency | Detector errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Patterns + validators | 0.98 | 0.33 | 0.49 | 0.49 | 32.5% | 1/15 | 0 / 2 ms | 0 |
| Patterns + NER | 0.97 | 0.97 | 0.97 | 0.95 | 97.6% | 2/15 | 70 / 84 ms | 0 |
| Patterns + NER + LLM | 0.96 | 0.99 | 0.98 | 0.96 | 100.0% | 2/15 | 9045 / 13606 ms | 1 |

### NER model choice (patterns + model, whole gold set)

| Model | PERSON F1 | ADDRESS F1 | ORGANIZATION F1 | Overall F1 | NER ms / doc | Load s |
|---|---:|---:|---:|---:|---:|---:|
| `knowledgator/gliner-pii-base-v1.0` | 0.98 | 0.91 | 0.80 | 0.96 | 67 | 3.0 |
| `knowledgator/gliner-pii-edge-v1.0` | 0.95 | 0.96 | 0.61 | 0.93 | 16 | 2.82 |
| `urchade/gliner_multi_pii-v1` | 0.96 | 0.94 | 0.70 | 0.94 | 110 | 5.27 |

## Leak test (gold set through the gateway, recording upstream)

| Policy | Detectors | Channel | Protected values | Leaked | Leak rate | Partial name leaks | Blocked requests |
|---|---|---|---:|---:|---:|---:|---:|
| `support-chat` | Patterns + validators | user | 330 | 189 | **57.27%** | 0 | 0/202 |
| `support-chat` | Patterns + validators | tool_result | 330 | 189 | **57.27%** | 0 | 0/202 |
| `strict-finance` | Patterns + validators | user | 330 | 181 | **54.85%** | 0 | 28/202 |
| `strict-finance` | Patterns + validators | tool_result | 330 | 181 | **54.85%** | 0 | 28/202 |
| `natural-text` | Patterns + validators | user | 330 | 189 | **57.27%** | 0 | 0/202 |
| `natural-text` | Patterns + validators | tool_result | 330 | 189 | **57.27%** | 0 | 0/202 |
| `analytics-irreversible` | Patterns + validators | user | 306 | 165 | **53.92%** | 0 | 0/202 |
| `analytics-irreversible` | Patterns + validators | tool_result | 306 | 165 | **53.92%** | 0 | 0/202 |
| `support-chat` | Patterns + NER | user | 330 | 6 | **1.82%** | 5 | 0/202 |
| `support-chat` | Patterns + NER | tool_result | 330 | 6 | **1.82%** | 5 | 0/202 |
| `strict-finance` | Patterns + NER | user | 330 | 3 | **0.91%** | 5 | 28/202 |
| `strict-finance` | Patterns + NER | tool_result | 330 | 3 | **0.91%** | 5 | 28/202 |
| `natural-text` | Patterns + NER | user | 330 | 6 | **1.82%** | 6 | 0/202 |
| `natural-text` | Patterns + NER | tool_result | 330 | 6 | **1.82%** | 5 | 0/202 |
| `analytics-irreversible` | Patterns + NER | user | 306 | 12 | **3.92%** | 16 | 0/202 |
| `analytics-irreversible` | Patterns + NER | tool_result | 306 | 12 | **3.92%** | 16 | 0/202 |

## Utility (real models)

Task `reply`, policy `support-chat`, 20 documents. Model `nvidia/nemotron-3-super-120b-a12b:free`, judge `dots-studio/dots-3-note-preview:free`.

| Redacted vs original | Count |
|---|---:|
| better | 7 |
| equivalent | 6 |
| worse | 7 |
| error | 0 |

- Equivalent or better: **65.0%**
- Restore accuracy: **100.0%** (39 placeholders in raw answers, 0 unknown, 0 left after restore)
- Name fidelity: **95.0%** of 20 answers where the baseline used the customer's name
- Gold PII values sent upstream during the utility run: 1
- Noise floor (two samples of the original-text answer, same judge): equivalent or better **70.0%** (better 5, equivalent 9, worse 6, errors 0)

## Real API calls

275 requests; every model id ends in `:free`: True.
By tag: {'smoke': 4, 'utility_model': 119, 'utility_judge': 82, 'llm_detector': 68, 'playground': 2}. By status: {'ok': 264, 'retryable_error': 5, 'error': 6}. Served models: {'nvidia/nemotron-3-super-120b-a12b:free': 185, 'dots-studio/dots-3-note-preview:free': 79}.
