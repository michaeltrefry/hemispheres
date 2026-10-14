"""Build the step-1 comparison report from exported run records.

  # Regenerate results/step1/REPORT.md and results/step1/reproduce.sh from results/step1/runs/
  python -m hemispheres.report
  # Fail if they are stale, or if any eval summary disagrees with its per-question records
  python -m hemispheres.report --check
  # The same report from your own reproduction (run directories under runs/), to diff against ours
  python -m hemispheres.report --runs runs --out /tmp/my-step1

Every accuracy is recounted from per-question records (`evals/*.jsonl`) where they exist.
Cells marked † come from the summary logged during training (`metrics.jsonl`), which
has no per-question records. Intervals are 95% Wilson score intervals.

Standard library only: the claims can be checked without MLX or Apple hardware.
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

from .synth.schema import ENTITY_TYPES, RELATIONS

RECORDS = Path("results/step1")
HELD_OUT = ("test_id", "test_1hop_ood")  # the sets the leakage evaluations use
# Flags that changed default after a run was trained, with the value the run used.
LEGACY_TRAIN_FLAGS = {"latent": {"supervise": "first"}}  # --supervise was added in d98359c; before it, "first"
DERIVED_CONFIG_KEYS = {"model", "model_type", "params", "vocab", "provenance"}


@dataclass
class Cell:
    correct: int
    n: int
    source: str
    logged: bool = False  # from a logged summary, not per-question records

    @property
    def acc(self) -> float:
        return self.correct / self.n

    def wilson(self, z: float = 1.96) -> tuple[float, float]:
        p, n = self.acc, self.n
        centre = (p + z * z / (2 * n)) / (1 + z * z / n)
        half = z / (1 + z * z / n) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
        return max(0.0, centre - half), min(1.0, centre + half)


def pct(x: float) -> str:
    v = 100 * x
    return f"{v:.0f}" if abs(v - round(v)) < 0.05 else f"{v:.1f}"


class Run:
    def __init__(self, path: Path):
        self.path, self.name = path, path.name
        self.config = json.loads((path / "config.json").read_text())
        self.metrics = [json.loads(line) for line in (path / "metrics.jsonl").read_text().splitlines() if line]
        prov = path / "provenance.json"
        self.provenance = json.loads(prov.read_text()) if prov.exists() else {}
        self.evals = {}  # name -> (json, records or None)
        for f in sorted((path / "evals").glob("*.json")):
            jl = f.with_suffix(".jsonl")
            records = [json.loads(line) for line in jl.read_text().splitlines() if line] if jl.exists() else None
            self.evals[f.stem] = (json.loads(f.read_text()), records)

    @property
    def world(self) -> str:
        return Path(self.config["data"].split(",")[0]).name

    @property
    def last_step(self) -> int:
        return max(m["step"] for m in self.metrics)

    def cell(self, world: str, sets: tuple[str, ...], hops: int | None = None, edits: int = 0,
             store: str | None = None, step: int | None = None) -> Cell | None:
        """Accuracy on `sets` (one hop count, or all) of `world` with `edits` applied, served by
        `store` (None: the world's own; "none": values hidden; a world name: that world's)."""
        if step is None:
            for name, (ev, records) in self.evals.items():
                a = ev["args"]
                if (records is None or Path(a["data"]).name != world or a.get("edits", 0) != edits
                        or (Path(a["store"]).name if a.get("store") not in (None, "none") else a.get("store")) != store):
                    continue
                rs = [r for r in records if r["set"] in sets and (hops is None or r["hops"] == hops)]
                if rs:
                    return Cell(sum(r["correct"] for r in rs), len(rs), f"evals/{name}.jsonl")
        if world != self.world or edits != self.config.get("edits", 0) or store is not None:
            return None
        logged = [m for m in self.metrics if "eval" in m and (step is None or m["step"] == step)]
        if not logged:
            return None
        m = logged[-1]
        key = f"{hops}hop" if hops else "all"
        parts = [m["eval"][s][key] for s in sets if key in m["eval"].get(s, {})]
        if not parts:
            return None
        return Cell(round(sum(p["acc"] * p["n"] for p in parts)), sum(p["n"] for p in parts),
                    f"metrics.jsonl, step {m['step']}", logged=True)

    def metric_at(self, step: int, key: str):
        return next((m[key] for m in self.metrics if m["step"] == step and key in m), None)


def load_runs(runs_dir: Path) -> dict[str, Run]:
    return {p.name: Run(p) for p in sorted(runs_dir.iterdir()) if (p / "config.json").exists()}


# ---------------------------------------------------------------- tables

MAIN_COLUMNS = [
    ("lookup-a", "lookup"), ("dense-a", "dense"), ("dense-a-k100", "dense + FT on 100 edits"),
    ("dense-a-to-b", "dense + trained on world B"), ("context-a", "context-a (QA only, world A)"),
    ("context-multi-bios", "context-multi (10k)"), ("context-multi-bios-20k", "**context-multi (20k)**"),
    ("context-multi-bios-20k-s1", "context-multi (20k, seed 1)"),
    ("latent-a", "latent-a"), ("latent-a-all", "latent-a-all"), ("latent-multi", "**latent-multi**"),
]
MAIN_ROWS = [
    ("World A, 1 hop, held-out people", dict(world="world-a", sets=("test_1hop_ood",), hops=1)),
    ("World A, 2 hop, held-out people", dict(world="world-a", sets=("test_ood",), hops=2)),
    ("World A, 3 hop, held-out people", dict(world="world-a", sets=("test_ood",), hops=3)),
    ("World A, 2 / 3 hop, people seen in multi-hop training", dict(world="world-a", sets=("test_id",))),
    ("100 edits: the edited fact", dict(world="world-a", sets=("direct",), edits=100)),
    ("100 edits: ripple (multi-hop through an edit)", dict(world="world-a", sets=("ripple",), edits=100)),
    ("100 edits: locality (untouched questions)", dict(world="world-a", sets=("locality",), edits=100)),
    ("100 edits: locality, 1 hop only", dict(world="world-a", sets=("locality",), hops=1, edits=100)),
    ("1,000 edits: the edited fact", dict(world="world-a", sets=("direct",), edits=1000)),
    ("1,000 edits: ripple", dict(world="world-a", sets=("ripple",), edits=1000)),
    ("1,000 edits: locality", dict(world="world-a", sets=("locality",), edits=1000)),
    ("Unseen world B, all hops", dict(world="world-b", sets=("all",))),
    ("Unseen world C, all hops", dict(world="world-c", sets=("all",))),
    ("World A, store values hidden (lower is better)", dict(world="world-a", sets=HELD_OUT, store="none")),
    ("World A, world C's store (lower is better)", dict(world="world-a", sets=HELD_OUT, store="world-c")),
]


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in rows]


def show(c: Cell | None) -> str:
    return "—" if c is None else pct(c.acc) + ("†" if c.logged else "")


def main_table(runs: dict[str, Run]) -> tuple[list[str], list[tuple]]:
    cols = [(r, label) for r, label in MAIN_COLUMNS if r in runs]
    rows, cells = [], []
    for label, q in MAIN_ROWS:
        row = [label]
        for r, _ in cols:
            c = runs[r].cell(**q)
            row.append(show(c))
            if c is not None:
                cells.append((label, r, c))
        rows.append(row)
    return table([""] + [label for _, label in cols], rows), cells


def answer_type(path: str) -> str:
    obj = RELATIONS[path.split("/")[-1]].object
    return obj if obj in ENTITY_TYPES or obj == "currency" else "value"


def answer_type_table(runs: dict[str, Run], world: str = "world-b") -> list[str]:
    names = [r for r in ("lookup-a", "latent-a", "latent-a-all", "latent-multi") if r in runs]
    groups = [("people", "person"), ("companies", "company"), ("universities", "university"), ("cities", "city"),
              ("countries", "country"), ("currencies", "currency"),
              ("single-token values (years, majors, industries)", "value")]
    rows = []
    for label, t in groups:
        row = [label]
        for r in names:
            recs = next((rs for ev, rs in runs[r].evals.values()
                         if rs and Path(ev["args"]["data"]).name == world and not ev["args"].get("store")), None)
            rs = [x for x in recs or [] if answer_type(x["path"]) == t]
            row.append(f"{pct(sum(x['correct'] for x in rs) / len(rs))} (n={len(rs)})" if rs else "—")
        rows.append(row)
    return table([f"{world} accuracy by answer type"] + names, rows)


def leakage_table(runs: dict[str, Run]) -> list[str]:
    def with_range(run: Run, store: str | None) -> str:
        pooled = run.cell("world-a", HELD_OUT, store=store)
        by_hop = [c for h in (1, 2, 3) if (c := run.cell("world-a", HELD_OUT, hops=h, store=store))]
        if pooled is None:
            return "—"
        lo, hi = min(c.acc for c in by_hop), max(c.acc for c in by_hop)
        return f"{show(pooled)} ({pct(lo)}–{pct(hi)} by hop)"
    rows = [[r] + [with_range(runs[r], s) for s in (None, "none", "world-c")]
            for r in ("lookup-a", "latent-a", "latent-a-all", "latent-multi") if r in runs]
    return table(["world-A test_id + test_1hop_ood", "own store", "values hidden", "world C's store"], rows)


def hop_supervision_table(runs: dict[str, Run]) -> list[str]:
    a, b = runs.get("latent-multi"), runs.get("latent-multi-nohop")
    if not a or not b:
        return []

    def ret(run: Run, step: int) -> str:
        v = run.metric_at(step, "retrieval_top1")
        return " / ".join(f"{100 * x:.3g}" for x in v) if v else "—"

    def num(run: Run, step: int, key: str) -> str:
        v = run.metric_at(step, key)
        return "—" if v is None else f"{v:.3g}"

    def evals(run: Run, step: int) -> str:
        cs = [run.cell("world-a", ("test_1hop_ood",), 1, step=step)] + \
             [run.cell("world-a", ("test_ood",), h, step=step) for h in (2, 3)]
        return " / ".join(show(c) for c in cs)

    rows = [[f"Retrieval top-1 % during training, hops 1 / 2 / 3, step {s}", ret(a, s), ret(b, s)] for s in (500, 3000)]
    rows += [[f"Training loss, step {s}", num(a, s, "loss"), num(b, s, "loss")] for s in (2000, 3000)]
    rows += [[f"Retrieval (InfoNCE) loss, step {s}", num(a, s, "hop_loss"), num(b, s, "hop_loss")] for s in (100, 3000)]
    rows.append(["World A held-out at step 2,000: 1 / 2 / 3 hop", evals(a, 2000), evals(b, 2000)])
    return table(["", f"latent-multi (hop weight {a.config['hop_weight']})",
                  f"latent-multi-nohop (hop weight {b.config['hop_weight']})"], rows)


def runs_table(runs: dict[str, Run]) -> list[str]:
    rows = []
    for r in runs.values():
        c = r.config
        worlds = c["data"].split(",")
        flags = [f"mix {','.join(f'{k}={v:g}' for k, v in c['mix'])}"]
        if c.get("init"):
            flags.append(f"init {Path(c['init']).name}")
        if c.get("edits"):
            flags.append(f"edits {c['edits']}")
        if c.get("model_type") == "latent":
            flags.append(f"hop weight {c['hop_weight']:g}")
            flags.append(f"supervise {c.get('supervise', LEGACY_TRAIN_FLAGS['latent']['supervise'])}")
            if c.get("hop_until"):
                flags.append(f"hop until {c['hop_until']}")
        steps = f"{r.last_step:,} of {c['steps']:,}" if r.last_step < c["steps"] else f"{c['steps']:,}"
        code = r.provenance.get("code") or c.get("provenance", {}).get("code") or {}
        commit = (code.get("commit") or "")[:7]
        commit = f"{commit}{'*' if code.get('dirty') else ''}{' (inferred)' if 'note' in code else ''}" or "—"
        ckpt = r.provenance.get("checkpoints", {})
        final = ckpt.get("final") or ckpt.get("latest")
        rows.append([r.name, c["arm"], f"{c['params'] / 1e6:.1f}M",
                     worlds[0].split("/")[-1] + (f" + {len(worlds) - 1} pool" if len(worlds) > 1 else ""),
                     steps, str(c["seed"]), "; ".join(flags), commit,
                     f"`{final['sha256'][:12]}`" if final else "—"])
    return table(["run", "arm", "params", "worlds", "steps", "seed", "settings", "code", "final weights sha256"], rows)


def cell_table(cells: list[tuple]) -> list[str]:
    rows = []
    for label, r, c in cells:
        lo, hi = c.wilson()
        rows.append([label, r, f"{c.correct}/{c.n}", pct(c.acc) + ("†" if c.logged else ""),
                     f"{pct(lo)}–{pct(hi)}", f"`{c.source}`"])
    return table(["row", "run", "correct / n", "%", "95% CI", "source"], rows)


def integrity(runs: dict[str, Run]) -> tuple[int, list[str]]:
    """Recount every eval summary from its per-question records: (files checked, mismatches)."""
    checked, bad = 0, []
    for r in runs.values():
        for name, (ev, records) in r.evals.items():
            if records is None:
                bad.append(f"{r.name}/evals/{name}.json has no per-question records")
                continue
            checked += 1
            for s, groups in ev["summary"].items():
                for key, v in groups.items():
                    if key == "all":
                        rs = [x for x in records if x["set"] == s]
                    elif key.endswith("hop"):
                        rs = [x for x in records if x["set"] == s and x["hops"] == int(key[:-3])]
                    else:
                        continue  # ripple breakdowns by edit position
                    if len(rs) != v["n"] or abs(sum(x["correct"] for x in rs) / len(rs) - v["acc"]) > 1e-9:
                        bad.append(f"{r.name}/evals/{name}: {s} {key} summary {v['acc']:.4f} (n={v['n']}), "
                                   f"records {sum(x['correct'] for x in rs)}/{len(rs)}")
    return checked, bad


def reevaluations(runs: dict[str, Run]) -> tuple[list[str], int, list[str]]:
    """Evaluations logged during training, compared with later reruns of the same checkpoint on the
    same questions: (runs compared, cells that agree, disagreements)."""
    compared, agree, differ = [], 0, []
    for r in runs.values():
        c = r.config
        latest = c["steps"] if r.last_step >= c["steps"] else r.last_step // c["save_every"] * c["save_every"]
        tags = {"final": c["steps"] if r.last_step >= c["steps"] else None, "latest": latest}
        for name, (ev, _) in r.evals.items():
            a = ev["args"]
            if ("step" in a or Path(a["data"]).name != r.world or a.get("edits", 0) != c.get("edits", 0)
                    or a.get("store") is not None or a.get("seed", 0) != c["seed"]):
                continue
            logged = next((m for m in r.metrics if "eval" in m and m["step"] == tags.get(a.get("checkpoint"))), None)
            if logged is None:
                continue
            for s, groups in logged["eval"].items():
                for key, v in groups.items():
                    rerun = ev["summary"].get(s, {}).get(key)
                    if rerun is None or rerun["n"] != v["n"]:
                        continue  # a different question sample
                    if r.name not in compared:
                        compared.append(r.name)
                    if abs(rerun["acc"] - v["acc"]) < 1e-9:
                        agree += 1
                    else:
                        differ.append(f"{r.name} {s} {key}: logged {pct(v['acc'])} at step {logged['step']}, "
                                      f"rerun {pct(rerun['acc'])} (`evals/{name}.json`, n={v['n']})")
    return compared, agree, differ


# ---------------------------------------------------------------- reproduction commands

def flag(k: str, v) -> str:
    return f"--{k.replace('_', '-')} {v}"


def build_command(path: str, build: dict) -> str:
    out_dir = str(Path(path).parent)
    parts = [flag(k, v) for k, v in build.items() if k not in ("out_dir",) and v not in ("", None)]
    if out_dir != "data":
        parts.append(flag("out_dir", out_dir))
    return "python -m hemispheres.synth.build " + " ".join(parts)


def train_command(run: Run) -> str:
    c = run.config
    args = {k: v for k, v in c.items() if k not in DERIVED_CONFIG_KEYS and v is not None and v is not False}
    args["mix"] = ",".join(f"{k}={v:g}" for k, v in c["mix"])
    for k, v in LEGACY_TRAIN_FLAGS.get(c.get("model_type", "gpt"), {}).items():
        args.setdefault(k, v)
    if c.get("model_type") != "latent":
        for k in ("n_reads", "top_k", "hop_weight", "hop_until", "max_supervision", "supervise"):
            args.pop(k, None)
    args["out"] = f"runs/{run.name}"
    if args.get("init"):
        args["init"] = f"runs/{Path(args['init']).name}"
    return "python -m hemispheres.train " + " ".join(flag(k, f'"{v}"' if "," in str(v) else v) for k, v in args.items())


def eval_command(run: Run, args: dict) -> str | None:
    if "step" in args:  # written by the training loop
        return None
    parts = [flag("run", f"runs/{run.name}")]
    for k in ("checkpoint", "data", "edits", "sets", "n", "seed", "store"):
        v = args.get(k)
        if v not in (None, 0, "") or k == "seed":
            parts.append(flag(k, v))
    return "python -m hemispheres.evaluate " + " ".join(parts)


def reproduce_script(runs: dict[str, Run], worlds: dict, weights: dict | None = None) -> str:
    lines = ["#!/bin/sh",
             "# Rebuild every world, retrain every run and rerun every evaluation behind REPORT.md.",
             "# Generated by `python -m hemispheres.report`; do not edit.",
             "# Runs one GPU job at a time. The small models take ~40-100 min each on an M5 Max.",
             "set -e", 'PYTHON="${PYTHON:-.venv/bin/python}"', "",
             "# 1. Worlds. Deterministic: verify-data checks them byte-for-byte against the ones we used."]
    lines += [build_command(p, w["build"]) for p, w in worlds.items()]
    lines += ["python -m hemispheres.records verify-data " + " ".join(worlds), "", "# 2. Training"]
    if weights:
        lines += ["# To evaluate our weights instead of retraining, skip this step and fetch them (sha256-checked):",
                  "#   python -m hemispheres.records fetch " + " ".join(weights["runs"])]
    ordered = sorted(runs.values(), key=lambda r: (bool(r.config.get("init")), r.name))
    for r in ordered:
        if r.last_step < r.config["steps"]:
            lines.append(f"# {r.name} was stopped by hand at step {r.last_step:,} of {r.config['steps']:,}")
        lines.append(train_command(r))
    lines += ["", "# 3. Evaluations"]
    for r in ordered:
        lines += [cmd for ev, _ in r.evals.values() if (cmd := eval_command(r, ev["args"]))]
    lines += ["", "# 4. Your report, to diff against results/step1/REPORT.md",
              "python -m hemispheres.report --runs runs --out repro-step1",
              "diff results/step1/REPORT.md repro-step1/REPORT.md || true", ""]
    return "\n".join('"$PYTHON"' + l[len("python"):] if l.startswith("python -m") else l for l in lines)


# ---------------------------------------------------------------- report

def reevaluation_lines(runs: dict[str, Run]) -> list[str]:
    compared, agree, differ = reevaluations(runs)
    if not compared:
        return ["- No evaluation logged during training has been rerun yet."]
    head = (f"- Evaluations logged during training, rerun from the saved checkpoint on the same questions "
            f"({', '.join(compared)}): {agree} of {agree + len(differ)} cells identical")
    return [head + ("." if not differ else ". These differ:"), *[f"  - {d}" for d in differ]]


def report(runs: dict[str, Run], worlds: dict, runs_dir: Path, weights: dict | None = None) -> str:
    main, cells = main_table(runs)
    checked, bad = integrity(runs)
    logged = sorted({(label, r) for label, r, c in cells if c.logged})
    out = [
        "# Step 1 comparison report",
        "",
        f"*Generated by `python -m hemispheres.report` from the records in `{runs_dir}/`. Do not edit by hand.*",
        "",
        "Exact-match accuracy in %: the whole generated answer must match. Each cell's question count, "
        "95% interval and source file are in [every cell](#every-cell). "
        "† = from the summary logged during training (`metrics.jsonl`), which has no per-question records. "
        "— = not run. Setup, interpretation and caveats: [`research/results-step1.md`](../../research/results-step1.md).",
        "",
        "## Main comparison",
        "",
        *main,
        "",
        "- `dense + FT on 100 edits` is `dense-a` fine-tuned for 200 steps on the 100 edited facts; "
        "`dense + trained on world B` is `dense-a` trained for 2,000 more steps on world B's bios.",
        "- The latent variants change one thing at a time: `latent-a-all` supervises retrieval at every name "
        "token, `latent-multi` also trains across 16 worlds.",
        "- `context-a` is the in-context oracle arm trained on world-A questions only (no bios): it never learns "
        "to copy from its prompt. The `context-multi` runs use the other arms' mix (bios + QA) across the same 16 "
        "worlds as `latent-multi`; the `20k` runs train for twice the others' step budget, which the arm needs "
        "because its copy circuit forms late.",
        "",
        "## World-swap errors by answer type",
        "",
        *answer_type_table(runs),
        "",
        "## Leakage: accuracy without the right store",
        "",
        "Pooled over world-A `test_id` and `test_1hop_ood` (lower is better for the last two columns); "
        "in brackets, the range over hop counts.",
        "",
        *leakage_table(runs),
    ]
    hop = hop_supervision_table(runs)
    if hop:
        out += ["", "## Hop supervision", "",
                "From `metrics.jsonl`: retrieval accuracy on training batches, logged every 100 steps. "
                "Chance is 1 in 62.6k.", "", *hop]
    out += [
        "", "## Runs", "",
        "Every flag, seed and model shape is in `runs/<run>/config.json`; checkpoint hashes and the "
        "fingerprint of every world read are in `runs/<run>/provenance.json`. \"(inferred)\" code versions "
        "were not recorded at training time: they are the last commit before the run started, and the working "
        "tree may have had uncommitted changes. `*` = uncommitted changes.",
        "",
        *runs_table(runs),
        "", "## Integrity", "",
        f"- {checked} evaluation files: every summary recounted from its per-question records "
        + ("matches." if not bad else f"has {len(bad)} problem(s):"),
        *[f"  - {b}" for b in bad],
        *reevaluation_lines(runs),
        f"- {len(logged)} cells come from logged summaries only (†)" + (
            ": rerun those evaluations with `python -m hemispheres.evaluate` to get per-question records."
            if logged else "."),
        "- Worlds: `worlds.json` holds each world's build config and file hashes. "
        "`python -m hemispheres.records verify-data data/<world>` checks a rebuilt world against them.",
        "", "## Reproducing", "",
        "`reproduce.sh` has every command, generated from the records: world builds, training runs (with every "
        "recorded flag) and evaluations. Retraining at a different code version or MLX build may not be "
        "bit-identical. Compare at the level of this report.",
        *(["", f"The weights are on the Hugging Face Hub at [{weights['repo']}](https://huggingface.co/{weights['repo']}), "
               f"commit `{weights['revision'][:7]}`. `python -m hemispheres.records fetch <run>` downloads a run into "
               "`runs/<run>/` and checks every file against the sha256 above, so the evaluations can be rerun "
               "without retraining."] if weights else []),
        "", "## Every cell", "",
        *cell_table(cells), "",
    ]
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", default=str(RECORDS / "runs"), help="directory of run record directories")
    p.add_argument("--worlds", default=str(RECORDS / "worlds.json"))
    p.add_argument("--out", default=str(RECORDS), help="where REPORT.md and reproduce.sh go")
    p.add_argument("--check", action="store_true", help="fail if the outputs are stale or the records inconsistent")
    args = p.parse_args()

    runs = load_runs(Path(args.runs))
    worlds = json.loads(Path(args.worlds).read_text()) if Path(args.worlds).exists() else {}
    weights_path = Path(args.worlds).parent / "weights.json"
    weights = json.loads(weights_path.read_text()) if weights_path.exists() else None
    outputs = {"REPORT.md": report(runs, worlds, Path(args.runs), weights),
               "reproduce.sh": reproduce_script(runs, worlds, weights)}
    out = Path(args.out)
    if args.check:
        _, bad = integrity(runs)
        stale = [n for n, text in outputs.items() if not (out / n).exists() or (out / n).read_text() != text]
        for b in bad:
            print(f"inconsistent: {b}")
        for n in stale:
            print(f"stale: {out / n} (regenerate with python -m hemispheres.report)")
        sys.exit(1 if bad or stale else 0)
    out.mkdir(parents=True, exist_ok=True)
    for n, text in outputs.items():
        (out / n).write_text(text)
    (out / "reproduce.sh").chmod(0o755)
    print(f"wrote {out / 'REPORT.md'} and {out / 'reproduce.sh'}")


if __name__ == "__main__":
    main()
