"""Evaluate a trained model by exact match on generated answers.

  # Held-out questions in the training world
  python -m hemispheres.evaluate --run runs/lookup-a --data data/world-a
  # World swap: a world the model never saw
  python -m hemispheres.evaluate --run runs/lookup-a --data data/world-b --sets all
  # Counterfactual edits: direct / ripple / locality questions after 100 edits
  python -m hemispheres.evaluate --run runs/lookup-a --data data/world-a --edits 100
  # Leakage: world-A questions with the store's values hidden, or with world C's store
  python -m hemispheres.evaluate --run runs/latent-multi --data data/world-a --store none
  python -m hemispheres.evaluate --run runs/latent-multi --data data/world-a --store data/world-c

An answer is correct only if the generated text, after the last store reply in
the lookup format, is exactly the answer followed by [EOS]. For the lookup arm
the store is the evaluated world (with edits applied). The context arm gets that
world's facts in its prompt. The dense arm gets nothing, so it must have been
updated (see train.py --init) to know a new world or edits.

`--store` replaces the store for leakage tests: "none" hides every value (the
latent arm's retrieved entries contribute nothing; the lookup arm's lookups get
no reply), and a world directory serves that world's facts instead. A model
that holds no facts itself should fall to chance on the evaluated world.
"""

import argparse
import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx

from . import checkpoint, provenance
from .data import ARMS, WorldData
from .generate import generate
from .store import Store
from .synth import render
from .synth.render import Tokenizer
from .synth.schema import END, EOS, LOOKUP, PAD, RESULT
from .synth.world import World

SPLIT_SETS = ("test_id", "test_ood", "test_1hop_ood", "train", "all")
EDIT_SETS = ("direct", "ripple", "locality")


@dataclass
class Question:
    set: str
    subject: str
    path: tuple[str, ...]
    answer: str
    meta: dict = field(default_factory=dict)


def split_questions(data: WorldData, split: str, n_per_hop: int, seed: int = 0) -> list[Question]:
    """Up to `n_per_hop` questions per hop count from a split ("all" = every split)."""
    qs = [q for qs in data.splits.values() for q in qs] if split == "all" else data.splits[split]
    by_hops = defaultdict(list)
    for subject, path in qs:
        by_hops[len(path)].append((subject, path))
    rng = random.Random(f"eval:{split}:{seed}")
    out = []
    for hops in sorted(by_hops):
        for subject, path in rng.sample(by_hops[hops], min(n_per_hop, len(by_hops[hops]))):
            out.append(Question(split, subject, path, data.world.answer(subject, path)))
    return out


def edit_questions(data: WorldData, kind: str, n: int, seed: int = 0) -> list[Question]:
    items = data.edit_set[kind]
    items = random.Random(f"eval:{kind}:{seed}").sample(items, min(n, len(items)))
    out = []
    for q in items:
        meta = {k: q[k] for k in ("changed", "edit_hop", "before") if k in q}
        out.append(Question(kind, q["subject"], tuple(q["path"].split("/")), q["answer"], meta))
    return out


def prompt_tokens(data: WorldData, q: Question, style: str, seed: int = 0) -> list[str]:
    rng = random.Random(f"prompt:{q.subject}:{'/'.join(q.path)}:{seed}")
    ex = render.qa(data.world, q.subject, q.path, rng, style)
    return ex.tokens[:ex.tokens.index("A:") + 1]


def parse_output(output: list[str], style: str) -> tuple[list[str] | None, list[tuple]]:
    """(final answer tokens or None if generation never ended, lookup queries made)."""
    queries, i = [], 0
    while LOOKUP in output[i:]:
        i = output.index(LOOKUP, i) + 1
        if RESULT not in output[i:]:
            break
        j = output.index(RESULT, i)
        queries.append((tuple(output[i:j - 1]), output[j - 1].lstrip("@")))
        i = j
    if EOS not in output:
        return None, queries
    body = output[:output.index(EOS)]
    if style == "lookup" and END in body:
        body = body[len(body) - body[::-1].index(END):]
    return body, queries


def inserted_tokens(output: list[str]) -> int:
    """Tokens the store appended to a lookup-arm output: each `value [END]` after a `[RESULT]`."""
    n, inside = 0, False
    for t in output:
        if inside:
            n += 1
            inside = t != END
        elif t == RESULT:
            inside = True
    return n


def score(model, tok: Tokenizer, data: WorldData, questions: list[Question], style: str, seed: int = 0,
          batch_size: int = 256, lookup_store: World | None | str = "same") -> list[dict]:
    """`model` maps token ids (B, T) to logits; latent-store models are bound to a store first.

    `lookup_store` answers the lookup arm's queries: "same" is the evaluated world, None no store.
    """
    prompts = [prompt_tokens(data, q, style, seed) for q in questions]
    max_new = 24 + 16 * max((len(q.path) for q in questions), default=1) if style == "lookup" else 24
    store = (data.world if lookup_store == "same" else lookup_store) if style == "lookup" else None
    outputs = generate(model, tok, prompts, max_new=max_new, store=store, batch_size=batch_size)
    records = []
    for q, prompt, out in zip(questions, prompts, outputs):
        final, queries = parse_output(out, style)
        gold = list(data.world.surface(q.answer))
        rec = {"set": q.set, "subject": q.subject, "path": "/".join(q.path), "hops": len(q.path),
               "answer": render.detokenize(gold), "output": render.detokenize(out),
               "correct": final == gold, "prompt_tokens": len(prompt), **q.meta}
        inserted = inserted_tokens(out) if style == "lookup" else 0
        rec["output_tokens"] = len(out) - inserted  # tokens the model generated, [EOS] included
        if style == "lookup":
            rec["inserted_tokens"] = inserted
        if style == "lookup":
            chain = data.world.chain(q.subject, q.path)
            rec["trace_ok"] = queries == [(data.world.surface(s), r) for s, r, _ in chain]
        records.append(rec)
    return records


def retrieval_diagnostics(model, store: Store, tok: Tokenizer, data: WorldData, questions: list[Question],
                          prompts: list[list[str]], batch_size: int = 256) -> list[dict]:
    """For a latent-store model: at the last prompt position (where the answer starts),
    did read layer i retrieve hop i's gold fact, as the top-1 entry and within the top-k?"""
    import mlx.core as mx

    pad, out = tok.token_id(PAD), []
    for c in range(0, len(prompts), batch_size):
        enc = [tok.encode(p) for p in prompts[c:c + batch_size]]
        width = max(map(len, enc))
        ids = mx.array([e + [pad] * (width - len(e)) for e in enc], dtype=mx.int32)
        _, queries, keys = model.forward(ids, store.arrays)
        rows, last = mx.arange(len(enc)), mx.array([len(e) - 1 for e in enc])
        layers = []
        for read, q in zip(model.reads, queries):
            q = q[rows, last]                                            # (b, dk)
            bias = store.arrays["bias"]
            layers.append((mx.argmax(q @ keys.T + bias, axis=-1).tolist(),
                           read.retrieve(q[:, None, :], keys, bias)[:, 0].tolist()))
        for j, question in enumerate(questions[c:c + batch_size]):
            gold = [store.index[(s, r)] for s, r, _ in data.world.chain(question.subject, question.path)]
            gold = gold[:len(layers)]
            out.append({"ret_top1": [layers[i][0][j] == g for i, g in enumerate(gold)],
                        "ret_topk": [g in layers[i][1][j] for i, g in enumerate(gold)]})
    return out


def summarize(records: list[dict]) -> dict:
    groups = defaultdict(list)
    for r in records:
        groups[(r["set"], f"{r['hops']}hop")].append(r)
        if r["set"] == "ripple":
            groups[("ripple", "changed" if r.get("changed") else "unchanged")].append(r)
            groups[("ripple", f"edit_hop{r.get('edit_hop')}")].append(r)
        groups[(r["set"], "all")].append(r)
    out: dict = defaultdict(dict)
    for (s, key), rs in sorted(groups.items()):
        out[s][key] = {"acc": sum(r["correct"] for r in rs) / len(rs), "n": len(rs)}
        if "trace_ok" in rs[0]:
            out[s][key]["trace_ok"] = sum(r["trace_ok"] for r in rs) / len(rs)
        if "ret_top1" in rs[0]:
            ret = {}
            for i in range(max(len(r["ret_top1"]) for r in rs)):
                hop = [r for r in rs if len(r["ret_top1"]) > i]
                ret[f"h{i}"] = {"top1": sum(r["ret_top1"][i] for r in hop) / len(hop),
                                "topk": sum(r["ret_topk"][i] for r in hop) / len(hop)}
            out[s][key]["retrieval"] = ret
    return dict(out)


def format_summary(summary: dict) -> str:
    lines = []
    for s, groups in summary.items():
        cells = []
        for key, v in groups.items():
            extra = f", trace {v['trace_ok']:.0%}" if "trace_ok" in v else ""
            if "retrieval" in v:
                extra += ", ret " + "/".join(f"{h['top1']:.0%}" for h in v["retrieval"].values())
            cells.append(f"{key} {v['acc']:.1%} (n={v['n']}{extra})")
        lines.append(f"{s:>14}: " + " · ".join(cells))
    return "\n".join(lines)


def evaluate(model, tok: Tokenizer, data: WorldData, arm: str, sets: list[str], n: int, seed: int = 0,
             store_override: World | str | None = None) -> tuple:
    """`store_override`: None uses the evaluated world's store, "none" hides every value,
    a World serves that world's facts instead (see the module docstring)."""
    questions = []
    for s in sets:
        questions += edit_questions(data, s, n, seed) if s in EDIT_SETS else split_questions(data, s, n, seed)
    store_world = store_override if isinstance(store_override, World) else data.world
    forward, store = model, None
    if ARMS[arm].latent_store:
        store = Store(store_world, tok)
        arrays = store.arrays
        if store_override == "none":
            arrays = {**arrays, "val_mask": mx.zeros_like(arrays["val_mask"])}

        def forward(x):
            return model(x, arrays)
    lookup_store = None if store_override == "none" else store_world
    records = score(forward, tok, data, questions, ARMS[arm].qa_style, seed, lookup_store=lookup_store)
    # Retrieval diagnostics need the evaluated world's facts to be in the store.
    if store is not None and store_world is data.world:
        prompts = [prompt_tokens(data, q, ARMS[arm].qa_style, seed) for q in questions]
        for rec, diag in zip(records, retrieval_diagnostics(model, store, tok, data, questions, prompts)):
            rec.update(diag)
    return summarize(records), records


def write(out: Path, name: str, summary: dict, records: list[dict], **meta) -> Path:
    """<out>/<name>.json (summary and `meta`) and <out>/<name>.jsonl (one record per question)."""
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps({**meta, "summary": summary}, indent=2))
    with open(out / f"{name}.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return out / name


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--checkpoint", default="final")
    p.add_argument("--data", required=True, help="built world directory")
    p.add_argument("--edits", type=int, default=0, help="evaluate with this many edits applied")
    p.add_argument("--sets", default=None,
                   help=f"comma list from {SPLIT_SETS + EDIT_SETS} (default: test splits, or edit sets with --edits)")
    p.add_argument("--n", type=int, default=500, help="questions per hop count (splits) or per set (edits)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default=None, help="output name under <run>/evals/")
    p.add_argument("--store", default=None,
                   help='leakage tests: "none" hides the store\'s values; a world directory serves that world\'s facts')
    args = p.parse_args()

    model, config = checkpoint.load_model(args.run, args.checkpoint)
    data = WorldData(args.data, args.edits)
    tok = Tokenizer(data.vocab)
    sets = (args.sets.split(",") if args.sets else
            list(EDIT_SETS) if args.edits else ["test_id", "test_ood", "test_1hop_ood"])
    store_override = None
    if args.store == "none":
        store_override = "none"
    elif args.store:
        store_override = WorldData(args.store).world
    summary, records = evaluate(model, tok, data, config["arm"], sets, args.n, args.seed, store_override)
    print(f"{config['arm']} · {args.run} [{args.checkpoint}] on {data.path.name}"
          + (f" with {args.edits} edits" if args.edits else "")
          + (f", store: {args.store}" if args.store else ""))
    print(format_summary(summary))

    store_tag = f"-store-{Path(args.store).name}" if args.store else ""
    name = args.name or f"{data.path.name}{f'-k{args.edits}' if args.edits else ''}{store_tag}-{args.checkpoint}"
    weights = Path(args.run) / "checkpoints" / args.checkpoint / "model.safetensors"
    prov = {**provenance.record([args.data] + ([args.store] if store_override not in (None, "none") else [])),
            "checkpoint_sha256": provenance.sha256(weights)}
    path = write(Path(args.run) / "evals", name, summary, records, args=vars(args), provenance=prov)
    print(f"wrote {path}.json")


if __name__ == "__main__":
    main()
