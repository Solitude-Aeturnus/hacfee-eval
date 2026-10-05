# -*- coding: utf-8 -*-
"""HAC-FEE event and argument evaluator.

Scoring conventions (one-to-one matching):
  * Empty-string arguments and the "event_id" field are dropped from both the
    gold annotations and the predictions before scoring.
  * Within each document, gold and predicted event instances are matched
    one-to-one per event type: each gold instance is paired with the predicted
    instance of the same type that shares the most role:value pairs; the paired
    prediction is then consumed and cannot be matched again.
  * A hit is an exact role:value match ("role:value" string) inside a matched
    instance pair.
  * A predicted instance left without a gold match counts its role:value pairs
    as false positives; a gold instance left without a prediction counts its
    pairs as false negatives.
  * Event types with zero gold and zero predictions score 0 under the
    evaluator convention.

Metrics reported:
  * per-type precision / recall / F1 (argument-level counts within each type),
  * micro overall P/R/F1 (all pairs pooled),
  * macro F1 (unweighted mean of the per-type F1 scores over the schema).
"""

import argparse
import copy
import json
import sys


def load_schema(path):
    with open(path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    return [t["type"] for t in schema["event_types"]]


def load_jsonl(path):
    docid2events = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            docid2events[record["doc_id"]] = record["events"]
    return docid2events


def strip_events(events):
    """Drop event_id and empty-string arguments, as done before scoring."""
    return [{k: v for k, v in ev.items() if k != "event_id" and v != ""}
            for ev in events]


def convert_to_dict(events):
    """Group instances by event type; each instance becomes a set of
    "role:value" strings."""
    new_events = {}
    for event in events:
        event_type = event["event_type"]
        roles_list = [role + ":" + str(value)
                      for role, value in event.items() if role != "event_type"]
        new_events.setdefault(event_type, []).append(roles_list)
    return new_events


def find_the_most_similar(roles_list_label, roles):
    most_similar_roles = None
    biggest_num_hit = 0
    for roles_label in roles_list_label:
        num_hit = len(set(roles_label).intersection(set(roles)))
        if num_hit > biggest_num_hit:
            biggest_num_hit = num_hit
            most_similar_roles = roles_label
    return biggest_num_hit, most_similar_roles


def evaluate_single_sample(label, pred):
    label = convert_to_dict(label)
    pred = convert_to_dict(pred)
    num_total = {ev: sum(map(len, rl)) for ev, rl in label.items()}
    num_pred = {ev: sum(map(len, rl)) for ev, rl in pred.items()}
    num_hit = {ev: 0 for ev in label}

    for event, roles_list in label.items():
        if event not in pred:
            continue
        for roles in roles_list:
            hit, most_similar_roles = find_the_most_similar(pred[event], roles)
            if most_similar_roles:
                pred[event].remove(most_similar_roles)
            num_hit[event] += hit

    return num_total, num_pred, num_hit


def evaluate(labels, preds, all_events):
    n_gt = {ev: 0 for ev in all_events}
    n_pred = {ev: 0 for ev in all_events}
    n_hit = {ev: 0 for ev in all_events}
    known = set(all_events)
    labels = {k: [ev for ev in v if ev["event_type"] in known]
              for k, v in labels.items()}
    preds = {k: [ev for ev in v if ev["event_type"] in known]
             for k, v in preds.items()}

    for docid, label in labels.items():
        pred = preds.get(docid, [])
        gt_counts, pred_counts, hit_counts = evaluate_single_sample(label, pred)
        for ev, c in gt_counts.items():
            n_gt[ev] += c
        for ev, c in pred_counts.items():
            n_pred[ev] += c
        for ev, c in hit_counts.items():
            n_hit[ev] += c

    per_type = {}
    for ev in all_events:
        p = n_hit[ev] / n_pred[ev] if n_pred[ev] else 0.0
        r = n_hit[ev] / n_gt[ev] if n_gt[ev] else 0.0
        f1 = (2 * p * r) / (r + p) if (r + p) else 0.0
        per_type[ev] = {"P": p, "R": r, "F1": f1,
                        "gold_pairs": n_gt[ev], "pred_pairs": n_pred[ev], "hits": n_hit[ev]}

    total_pred = sum(n_pred.values())
    total_hit = sum(n_hit.values())
    total_gt = sum(n_gt.values())
    micro_p = total_hit / total_pred if total_pred else 0.0
    micro_r = total_hit / total_gt if total_gt else 0.0
    micro_f1 = (2 * micro_p * micro_r) / (micro_r + micro_p) if (micro_r + micro_p) else 0.0
    macro_f1 = sum(v["F1"] for v in per_type.values()) / len(all_events) if all_events else 0.0

    return per_type, {"P": micro_p, "R": micro_r, "F1": micro_f1}, macro_f1


def main():
    ap = argparse.ArgumentParser(description="HAC-FEE evaluator")
    ap.add_argument("--gold", required=True, help="gold JSONL (doc_id, events)")
    ap.add_argument("--pred", required=True, help="prediction JSONL (doc_id, events)")
    ap.add_argument("--schema", default=None, help="schema JSON with event_types")
    args = ap.parse_args()

    if args.schema:
        all_events = load_schema(args.schema)
    else:
        all_events = [
            "股东减持", "股东增持", "股权质押", "股权冻结", "解除质押", "股份回购",
            "高管变动", "高层死亡", "重大资产损失", "重大对外赔付", "重大安全事故",
            "亏损", "盈利", "中标", "公司上市", "并购重组", "企业融资", "企业破产",
            "被约谈", "分红派息", "股权激励", "债券发行", "债券违约", "股权转让",
            "关联交易", "投资", "股东大会决议", "资产减值", "异常波动公告"]

    labels = {k: strip_events(v) for k, v in load_jsonl(args.gold).items()}
    preds = {k: strip_events(v) for k, v in load_jsonl(args.pred).items()}

    known = set(all_events)
    unknown = {ev["event_type"] for doc in list(labels.values()) + list(preds.values())
               for ev in doc if ev["event_type"] not in known}
    if unknown:
        print("WARNING: event types outside the schema are ignored:", unknown,
              file=sys.stderr)

    per_type, micro, macro = evaluate(labels, preds, all_events)

    print("%-20s %8s %8s %8s %8s" % ("event_type", "P", "R", "F1", "gold/pred"))
    for ev in all_events:
        d = per_type[ev]
        print("%-20s %8.4f %8.4f %8.4f %6d/%d" %
              (ev, d["P"], d["R"], d["F1"], d["gold_pairs"], d["pred_pairs"]))
    print("-" * 60)
    print("micro overall  : P %.4f  R %.4f  F1 %.4f" % (micro["P"], micro["R"], micro["F1"]))
    print("macro F1       : %.4f" % macro)


if __name__ == "__main__":
    main()
