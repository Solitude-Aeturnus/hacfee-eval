# HAC-FEE Evaluator

Evaluation script for **HAC-FEE: A Human–AI Collaborative Framework for Document-Level Chinese Financial Event Extraction** (ChineseCSCW 2026, Paper 127).

This repository contains the matching-based evaluator used for the domain experiments: it compares model predictions against gold annotations and reports per-type precision/recall/F1, a pooled micro score, and the macro F1 over the 29-type schema. Evaluation only — inference, training code, and annotated data are not included (the source announcements are used under their original access conditions).

## Requirements

Python 3.7+, standard library only. No installation needed.

## Input format

Gold and prediction files are JSONL; each line is one document:

```json
{"doc_id": "doc-001", "events": [{"event_type": "股权质押", "质押方": "控股集团有限公司", "接收方": "工商银行北京分行", "质押金额": "1.2亿元", "质押开始日期": "2023年6月15日", "质押结束日期": ""}]}
```

- `event_type` takes one of the 29 types in `schema_29_types.json`;
- every other key is an argument role defined for that type; the value is the argument string;
- arguments not present in the text are empty strings `""` — they are dropped from both sides before scoring;
- an optional `event_id` field, if present, is ignored;
- documents without events have `"events": []`;
- event types outside the schema are reported to stderr and ignored.

## Scoring conventions

1. Within each document, gold and predicted instances of the same event type are aligned one-to-one: each gold instance is paired with the predicted instance sharing the most `role:value` pairs, and the paired prediction is consumed.
2. A hit is an exact `role:value` match inside a matched pair.
3. Unmatched predicted pairs count as false positives; unmatched gold pairs count as false negatives.
4. Duplicates are matched at most once; remaining duplicates count as false positives.
5. Types with zero gold and zero predicted pairs score 0.

The per-type F1 scores follow `2TP/(2TP+FP+FN)` on argument-level counts within each type; `micro overall` pools all pairs; `macro F1` is the unweighted mean of the per-type F1 scores over the schema.

## Usage

```bash
python evaluate.py --gold data/example_gold.jsonl --pred data/example_pred.jsonl --schema schema_29_types.json
```

`--schema` is optional; without it a built-in list of the 29 types is used. Expected output on the bundled example:

```
micro overall  : P 0.7333  R 0.9167  F1 0.8148
macro F1       : 0.0948
```

## Files

- `evaluate.py` — the evaluator.
- `schema_29_types.json` — the 29 event types and their legal argument roles (170 type–role pairs).
- `data/example_gold.jsonl`, `data/example_pred.jsonl` — a small runnable example.
