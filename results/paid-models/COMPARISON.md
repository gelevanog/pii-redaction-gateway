# Paid models vs the free baseline

Run on 2026-10-12, branch `eval/paid-models`. The free baseline is the committed run in [`results/`](..)
(2026-10-05, OpenRouter `:free` models only). Everything here used the same gold set, the same deterministic
document selection, prompts, policy (`support-chat`) and the same local GLiNER NER model; only the LLMs differ.
All paid calls went through an OpenAI-compatible budget proxy (`provider: openai` with a per-role `base_url`),
enabled with both `require_free_models: false` in [`configs/paid-models/*.yaml`](../../configs/paid-models)
and `PII_SHIELD_ALLOW_PAID_MODELS=1`.

What was run:

| Run | Answer / detector model | Judge | Scope |
|---|---|---|---|
| Free baseline (committed) | `nvidia/nemotron-3-super-120b-a12b:free` | `dots-studio/dots-3-note-preview:free` | LLM detector on the 68-doc subset; utility on 20 docs + noise floor |
| Free answers, re-judged | (the committed free answers, not regenerated) | `claude-opus-5-5` | 20 docs ([`rejudge_free_baseline.py`](rejudge_free_baseline.py)) |
| Haiku | `claude-haiku-5-5` | `claude-opus-5-5` | full: LLM detector on the 68-doc subset; utility on 20 docs + noise floor |
| Opus | `claude-opus-5-5` | `claude-opus-5-5` (self-judging, see notes) | full: same as Haiku |
| Qwen | `qwen3.7-plus` | `claude-opus-5-5` | full: same as Haiku (an earlier 5-doc smoke run is kept in [`qwen/history/`](qwen/history)) |

The judge changed from the free run (a free model is not reachable through the proxy), so the utility rows
should be compared against the **re-judged** free answers, not only the original free-judge numbers.

## End-to-end answer quality through the gateway (redact → model → restore)

"As good or better" = the judge rated the reply produced through the gateway at least as good as the reply the
same model wrote from the original, unredacted text. The noise floor is the same comparison between two samples
of the *unredacted* reply: the rate you get with no redaction at all.

| Metric | Free (free judge) | Free answers, Opus judge | Haiku | Opus | Qwen |
|---|---:|---:|---:|---:|---:|
| Documents | 20 | 20 | 20 | 20 | 20 |
| Redacted reply as good or better | 65% (13/20) | 55% (11/20) | **70% (14/20)** | **70% (14/20)** | **70% (14/20)** |
| better / equivalent / worse | 7 / 6 / 7 | 8 / 3 / 9 | 10 / 4 / 6 | 8 / 6 / 6 | 5 / 9 / 6 |
| Noise floor (two unredacted samples) | 70% (14/20) | – | 45% (9/20) | 80% (16/20) | 85% (17/20) |
| Placeholders in raw answers restored to the right value | 39/39 | – | 45/45 | 55/55 | 42/42 |
| Placeholders left in restored answers | 0 | – | 0 | 0 | 0 |
| Customer's real name in restored reply (when baseline used it) | 19/20 | – | 20/20 | 20/20 | 20/20 |
| Gold values sent upstream (same NER pipeline) | 1 | – | 1 | 1 | 1 |
| Unredacted replies with a `[Your Name]`-style slot | 4/20 | – | 9/20 | 5/20 | 0/20 |
| Answer-model latency per call, mean | 5.1 s | – | 3.4 s | 7.0 s | 3.6 s |
| Judge latency per call, mean | 15.7 s | 4.8 s | 5.2 s | 5.4 s | 4.5 s |
| Mean reply length, unredacted / redacted (chars) | 304 / 253 | – | 597 / 510 | 758 / 700 | 397 / 377 |

Agreement between the free judge and the Opus judge on the same 20 free answer pairs: **45% (9/20)**.

## LLM-assisted detection (68-document subset, 126 gold values, every 3rd document by id)

The LLM layer only sees text after patterns + NER masked what they found. Patterns + NER alone on this subset:
P/R/F1 (partial) 0.968 / 0.968 / 0.968, 70 ms per document.

| Patterns + NER + LLM | Free (nemotron) | Haiku | Opus | Qwen |
|---|---:|---:|---:|---:|
| Precision / recall / F1, partial match | 0.962 / 0.992 / 0.977 | **0.969** / 0.992 / **0.980** | 0.962 / 0.992 / 0.977 | 0.954 / 0.992 / 0.973 |
| Precision / recall / F1, strict match | 0.946 / 0.976 / 0.961 | 0.953 / 0.976 / 0.965 | 0.946 / 0.976 / 0.961 | 0.939 / 0.976 / 0.957 |
| Gold values redacted at all (any type) | 126/126 | 126/126 | 126/126 | 126/126 |
| Clean documents with a finding | 2/15 | 2/15 | 2/15 | 2/15 |
| Latency per document, mean / p95 | 9.0 s / 13.6 s | **2.3 s / 3.2 s** | 3.1 s / 5.5 s | 2.3 s / 3.0 s |
| Documents where the LLM call failed (final) | 1 | 0 | 0 | 0 |

Latency for Haiku and Opus is from the first pass ([`haiku/detection-first-pass.json`](haiku/detection-first-pass.json),
[`opus/detection-first-pass.json`](opus/detection-first-pass.json)); the final `detection.json` files re-ran from the
disk cache to fill documents whose call failed, so their latency columns are cache replays, not model latency.
Qwen had no failed calls, so its `detection.json` latency is from a single live pass.

## Cost and calls

| Model | Calls (ok / total) | Tokens in / out | Cost |
|---|---:|---:|---:|
| Free baseline (all roles) | 264 / 275 | 69k / 150k | $0 |
| `claude-haiku-5-5` (detector + answers, incl. a discarded first attempt) | 196 / 216 | 149k / 33.6k | ≈ $0.03 |
| `claude-opus-5-5` (judge in every run, detector, answers in the Opus run) | 273 / 276 | 217k / 66.9k | ≈ $2.21 |
| `qwen3.7-plus` (detector + answers; incl. the 5-doc smoke run) | 138 / 138 | 32.3k / 7.1k | 39,401 of 150,000 tokens |

Costs per model are computed from the call ledgers (`*/calls.jsonl`) at $0.10 / $0.50 (Haiku) and $4 / $20 (Opus)
per 1M tokens. The proxy's own counter for this project reads **$2.46** in total (Anthropic); the ~$0.22 gap is not attributed
in the ledgers (the proxy may bill requests it rejected with 403, which carry no usage in the ledger).

## Observations

- **Placeholders are not the bottleneck for any model.** Restore accuracy was 100% for every model (181 placeholders
  in total), no placeholder was left in a restored reply, and the customer's name came back in every Haiku, Opus and
  Qwen reply. The free model was already fine here (39/39, 19/20).
- **Haiku and Opus look slightly better than free (70% vs 55% under the same Opus judge), but the effect is small and
  partly an artifact.** The gateway's placeholder hint says "do not invent new placeholders", so redacted replies
  almost never contain `[Your Name]`-style signature slots, while unredacted replies often do (Haiku 9/20, Opus 5/20,
  free 4/20). The Opus judge penalizes those slots, and they are cited in 6 of the 10 Haiku "better" verdicts.
  With 20 documents a 3-document difference is not statistically meaningful.
- **Qwen ties Haiku and Opus at 70%, without that artifact.** Qwen never wrote a `[Your Name]` slot (0/20), so its
  score is not inflated by it, but its own noise floor is the highest (85%): its two unredacted samples are very alike,
  and the redacted pipeline is 3 documents below that floor (6 "worse" vs 3). Of its 6 "worse" verdicts, the judge's
  reasons point to who-is-who mistakes (`eml-015`, `mix-002`), a dropped or less precise detail (`ru-002`, `sup-010`,
  `eml-009`) and the German salutation (`de-001`), the same kinds of cases the other models lose.
- **The judge matters as much as the answer model.** Opus and the free judge agreed on only 9 of 20 identical free
  answer pairs, and the noise floor under the same Opus judge swung from 45% (Haiku samples) to 80% (Opus samples).
  Read every "as good or better" number against its own noise floor; a 20-document set cannot separate these models.
- **The failure that redaction causes is model-independent.** `de-001` was judged worse for free, Haiku, Opus
  and Qwen alike: `<PERSON_1>` hides the customer's gender, so the German reply says "Sehr geehrte/r" instead of "Herr
  Hoffmann". A stronger model does not fix this; gender-aware placeholders or the `natural-text` policy would.
- **For the LLM detector, free was already enough on quality; paid models only buy speed.** All four reached 126/126
  values redacted and 0.99 recall; the residual errors are the same documents (a "Ford" false positive, a LinkedIn
  URL typed as a person, a phone-like number), and Opus was no more precise than the free model. Qwen added one
  false positive of its own (a public GitHub repository URL flagged as personal), so its precision is the lowest
  (0.954), at the same 2.3 s per document as Haiku. Haiku was ~4x faster
  (2.3 s vs 9.0 s per document) at about $0.0001 per document, which matters for the batch use the README suggests.

## Caveats and incidents

- The Opus run is judged by Opus itself. Both sides of each comparison are Opus replies, so self-preference should
  largely cancel, but it is not a fully independent judge.
- The proxy intermittently answered `403 Request not allowed` (7 of 546 requests in the final runs, none for Qwen; different documents each time).
  The first Haiku detector attempt ran 3 requests in parallel and got 15 such errors; it was discarded (ledger kept in
  [`haiku/history/`](haiku/history)), concurrency was set to 1, and the remaining failures were filled by re-running
  from the disk cache, so every reported number covers all documents.
- Qwen: the first run was a 5-document smoke test with `max_tokens` 600 and no noise floor (a 3,000-token budget);
  after the budget was raised it was re-run on the full set with the same settings as Haiku and Opus
  (`max_tokens` 3000); the smoke ledger and results are in [`qwen/history/`](qwen/history) and counted in the totals.
- `seed` is sent for the noise-floor sample as in the baseline; Anthropic models ignore it, so the second sample is
  simply another sample at default temperature.
