# PII Shield evaluation report

Detection run: 2026-10-12T09:06:19+00:00.

## Detection (gold set)

202 documents, 330 entities, 50 documents without PII. Thresholds and allow-list: policy `support-chat`.

| Configuration | Precision | Recall | F1 strict | F1 partial | Any-type recall | Clean docs flagged | Latency |
|---|---:|---:|---:|---:|---:|---:|---:|

Precision and recall cells: strict / partial. Latency: mean / p95 per document.

| Entity | Gold |
|---|---:|
| PERSON | 139 |
| PHONE | 32 |
| EMAIL | 31 |
| ORGANIZATION | 24 |
| ADDRESS | 23 |
| SECRET | 16 |
| DATE_OF_BIRTH | 12 |
| IP_ADDRESS | 12 |
| CREDIT_CARD | 10 |
| IBAN | 9 |
| URL | 8 |
| NATIONAL_ID | 8 |
| US_SSN | 6 |

Per-type cells use partial matching (same type, overlapping span).

### With the LLM detector: every 3rd document (68 documents, 126 entities), all configurations on the same subset

| Configuration | Precision | Recall | F1 partial | F1 strict | Any-type recall | Clean docs flagged | Latency | Detector errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Patterns + validators | 0.98 | 0.33 | 0.49 | 0.49 | 32.5% | 1/15 | 0 / 2 ms | 0 |
| Patterns + NER | 0.97 | 0.97 | 0.97 | 0.95 | 97.6% | 2/15 | 76 / 102 ms | 0 |
| Patterns + NER + LLM | 0.96 | 0.99 | 0.98 | 0.96 | 100.0% | 2/15 | 112 / 94 ms | 0 |

## Utility (real models)

Task `reply`, policy `support-chat`, 20 documents. Model `claude-opus-5-5`, judge `claude-opus-5-5`.

| Redacted vs original | Count |
|---|---:|
| better | 8 |
| equivalent | 6 |
| worse | 6 |
| error | 0 |

- Equivalent or better: **70.0%**
- Restore accuracy: **100.0%** (55 placeholders in raw answers, 0 unknown, 0 left after restore)
- Name fidelity: **100.0%** of 20 answers where the baseline used the customer's name
- Gold PII values sent upstream during the utility run: 1
- Noise floor (two samples of the original-text answer, same judge): equivalent or better **80.0%** (better 11, equivalent 5, worse 4, errors 0)

## Real API calls

171 requests; every model id ends in `:free`: False.
By tag: {'llm_detector': 69, 'utility_model': 62, 'utility_judge': 40}. By status: {'error': 3, 'ok': 168}. Served models: {'claude-opus-5-5': 168}.
