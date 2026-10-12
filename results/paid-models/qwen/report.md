# PII Shield evaluation report


## Utility (real models)

Task `reply`, policy `support-chat`, 5 documents. Model `qwen3.7-plus`, judge `claude-opus-5-5`.

| Redacted vs original | Count |
|---|---:|
| better | 2 |
| equivalent | 2 |
| worse | 1 |
| error | 0 |

- Equivalent or better: **80.0%**
- Restore accuracy: **100.0%** (9 placeholders in raw answers, 0 unknown, 0 left after restore)
- Name fidelity: **100.0%** of 5 answers where the baseline used the customer's name
- Gold PII values sent upstream during the utility run: 1

## Real API calls

15 requests; every model id ends in `:free`: False.
By tag: {'utility_model': 10, 'utility_judge': 5}. By status: {'ok': 15}. Served models: {'qwen3.7-plus': 10, 'claude-opus-5-5': 5}.
