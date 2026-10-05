# Gold set: hand-labeled PII spans

`gold.jsonl` is compiled from the markup in [`source/`](source) by `pii-shield gold build` (CI checks that it is
up to date). Each document is written and labeled individually as `[[value|TYPE]]` against the guidelines
below; the compiler only strips the markup and computes character offsets. **The set was not produced by a
generation pipeline: no LLM API was called to write or label it, and no detector output was copied into the
labels.** Every name, address, number and key in it is invented; card numbers and IBANs were computed to pass
their checksums, and secrets are deliberately malformed so they cannot be mistaken for live credentials.

| | |
|---|---|
| Documents | 202 (166 development + 36 held-out), 50 of them without any PII |
| Entities | 330 across 13 types |
| Domains | support tickets, emails, CRM notes, chat logs, HR and medical-style notes, application logs |
| Languages | English, plus German, Spanish, Russian, Dutch and mixed-language texts |

**Held-out split.** `source/10-holdout.txt` (tag `holdout`) was written after the pattern recognizers and NER
post-processing were frozen and was never used to tune them. The README reports it separately: the development
split was used while building the recognizers, so its pattern scores are optimistic.

## Labeling guidelines

- **PERSON**: names of people, including single first names, surnames and nicknames. Titles (Mr., Dr., Frau)
  and a possessive `'s` are not part of the span. Role words (customer, agent, the CFO) are not names.
- **ADDRESS**: a street address with house number (plus flat, postcode and city when they follow directly, as
  one span). A city, country or postcode on its own is not labeled.
- **ORGANIZATION**: organizations a person is tied to: employer, the customer's company, school, clinic.
  Product names, the operator's own brand (Brightloop) and companies mentioned only as brands are not labeled.
- **EMAIL** (including obfuscated "name at domain dot com"), **PHONE** (with country code and parentheses, without
  "ext."), **CREDIT_CARD** and **IBAN** (only real ones: a 16-digit tracking number is not a card),
  **IP_ADDRESS** (except loopback), **US_SSN**, **NATIONAL_ID** (Spanish DNI/NIE, Dutch BSN).
- **URL**: only links that point at a person (profiles, account pages, links carrying personal identifiers).
- **DATE_OF_BIRTH**: only dates that are someone's birth date; other dates are not labeled.
- **SECRET**: passwords, API keys and tokens; the span is the value only, not the variable name.

## Corrections

While reviewing model output, one annotation omission was found in the development split (the second "Diane"
in `crm-005` was not labeled). It was fixed before any of the reported runs.
