"""Publish a run's records to the repo, and check data or weights against them.

  # Copy runs' configs, metrics, logs and per-question eval records into results/step1/runs/
  python -m hemispheres.records export runs/lookup-a runs/latent-multi --out results/step1
  # Check that rebuilt worlds are byte-identical to the ones the runs used
  python -m hemispheres.records verify-data data/world-a data/world-b --out results/step1
  # Check that a run's checkpoints are the ones the records came from
  python -m hemispheres.records verify-run runs/latent-multi --out results/step1
  # Publish runs' weights to the Hugging Face Hub, pinning the upload in <out>/weights.json
  python -m hemispheres.records upload runs/lookup-a runs/latent-multi --repo SceneWorks/hemispheres-step1
  # Download them into runs/<run>/ (checked against the recorded sha256), ready for evaluate.py
  python -m hemispheres.records fetch latent-multi lookup-a

`runs/` and `data/` stay git-ignored: checkpoints are ~100 MB each and worlds rebuild
deterministically from their recorded build config in seconds. Everything a result
depends on is exported: the training config (every flag, seed and model shape), the
metrics log, stdout, each evaluation's arguments, summary and per-question outputs,
sha256 of every checkpoint, and a fingerprint of every world a run read
(<out>/worlds.json). `python -m hemispheres.report` builds the comparison from these.

Standard library only, except `upload` and `fetch`, which need `pip install -e '.[hub]'`.
"""

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import provenance

RECORD_FILES = ("config.json", "metrics.jsonl", "stdout.log")


def data_dirs(run: Path) -> list[str]:
    """Every world directory a run's training and evaluations read."""
    config = json.loads((run / "config.json").read_text())
    dirs = config["data"].split(",")
    for f in sorted((run / "evals").glob("*.json")):
        args = json.loads(f.read_text()).get("args", {})
        dirs.append(args["data"])
        if args.get("store") not in (None, "none"):
            dirs.append(args["store"])
    return list(dict.fromkeys(dirs))


def code_at_start(run: Path, config: dict) -> dict:
    """The recorded code version, or for runs from before it was recorded, the last commit
    before the run started (config.json is written at the start)."""
    if "provenance" in config:
        return config["provenance"]["code"]
    started = datetime.fromtimestamp((run / "config.json").stat().st_mtime, timezone.utc)
    return {"commit": provenance.git("log", "-1", f"--before={started.isoformat()}", "--format=%H"),
            "dirty": None, "started": started.isoformat(timespec="seconds"),
            "note": "not recorded at training time: the last commit before the run started; "
                    "the working tree may have had uncommitted changes"}


def export(run: Path, out: Path) -> Path:
    dest = out / "runs" / run.name
    if dest.exists():
        shutil.rmtree(dest)
    (dest / "evals").mkdir(parents=True)
    for name in RECORD_FILES:
        if (run / name).exists():
            shutil.copy2(run / name, dest / name)
    for f in sorted((run / "evals").glob("*.json*")):
        shutil.copy2(f, dest / "evals" / f.name)
    config = json.loads((run / "config.json").read_text())
    checkpoints = {d.name: {"sha256": provenance.sha256(d / "model.safetensors"),
                            "bytes": (d / "model.safetensors").stat().st_size}
                   for d in sorted((run / "checkpoints").iterdir()) if (d / "model.safetensors").exists()}
    meta = {"code": code_at_start(run, config), "checkpoints": checkpoints,
            "data": {d: provenance.world_fingerprint(d) for d in data_dirs(run)}}
    (dest / "provenance.json").write_text(json.dumps(meta, indent=2) + "\n")

    worlds_path = out / "worlds.json"
    worlds = json.loads(worlds_path.read_text()) if worlds_path.exists() else {}
    for d in meta["data"]:
        stats = json.loads((Path(d) / "stats.json").read_text())
        worlds[d] = {"build": stats["config"], "fingerprint": meta["data"][d], "files": provenance.world_files(d)}
    worlds_path.write_text(json.dumps(dict(sorted(worlds.items())), indent=2) + "\n")
    return dest


def verify_data(dirs: list[str], out: Path) -> bool:
    worlds, ok = json.loads((out / "worlds.json").read_text()), True
    for d in dirs:
        rec = worlds.get(d) or next((w for k, w in worlds.items() if Path(k).name == Path(d).name), None)
        if rec is None:
            print(f"{d}: not in {out / 'worlds.json'}")
            ok = False
            continue
        files = provenance.world_files(d)
        bad = sorted(k for k in rec["files"].keys() | files.keys() if rec["files"].get(k) != files.get(k))
        print(f"{d}: {'OK' if not bad else 'MISMATCH in ' + ', '.join(bad)}")
        ok &= not bad
    return ok


def verify_run(run: Path, out: Path) -> bool:
    rec, ok = json.loads((out / "runs" / run.name / "provenance.json").read_text()), True
    for tag, c in rec["checkpoints"].items():
        path = run / "checkpoints" / tag / "model.safetensors"
        match = path.exists() and provenance.sha256(path) == c["sha256"]
        print(f"{path}: {'OK' if match else 'missing' if not path.exists() else 'MISMATCH'}")
        ok &= match
    return ok


def upload(runs: list[Path], out: Path, repo: str, private: bool = True) -> dict:
    """Upload each run's distinct checkpoints and its config in one commit to a Hub model repo,
    and record the repo, the commit and each checkpoint's path in <out>/weights.json."""
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()
    api.create_repo(repo, repo_type="model", private=private, exist_ok=True)
    weights_path = out / "weights.json"
    weights = json.loads(weights_path.read_text()) if weights_path.exists() else {"repo": repo, "runs": {}}
    if weights["repo"] != repo:
        raise SystemExit(f"{weights_path} points at {weights['repo']}, not {repo}")
    ops = []
    for run in runs:
        meta = json.loads((out / "runs" / run.name / "provenance.json").read_text())
        files, by_sha = {}, {}
        for tag, c in meta["checkpoints"].items():
            if c["sha256"] not in by_sha:  # "latest" is usually the same weights as "final"
                by_sha[c["sha256"]] = f"{run.name}/checkpoints/{tag}/model.safetensors"
                ops.append(CommitOperationAdd(by_sha[c["sha256"]], str(run / "checkpoints" / tag / "model.safetensors")))
            files[tag] = {"path": by_sha[c["sha256"]], "sha256": c["sha256"]}
        ops.append(CommitOperationAdd(f"{run.name}/config.json", str(run / "config.json")))
        weights["runs"][run.name] = files
    ops.append(CommitOperationAdd("README.md", model_card(weights, out).encode()))
    commit = api.create_commit(repo, operations=ops, repo_type="model",
                               commit_message=f"Upload {', '.join(r.name for r in runs)}")
    weights["revision"] = commit.oid
    weights["runs"] = dict(sorted(weights["runs"].items()))
    weights_path.write_text(json.dumps(weights, indent=2) + "\n")
    return weights


def model_card(weights: dict, out: Path) -> str:
    rows = []
    for name, files in sorted(weights["runs"].items()):
        config = json.loads((out / "runs" / name / "config.json").read_text())
        tags = ", ".join(f"`{t}`" for t in files)
        sha = next(iter(files.values()))["sha256"][:12]
        rows.append(f"| `{name}` | {config['arm']} | {config['params'] / 1e6:.1f}M | {tags} | `{sha}` |")
    return "\n".join([
        "---", "license: mit", "library_name: mlx", "tags: [mlx, knowledge-editing, retrieval, synthetic-data]", "---", "",
        "# Hemispheres: step-1 checkpoints", "",
        "Weights for the step-1 synthetic-world runs of Hemispheres, a language model that keeps its knowledge in a "
        "separate, editable store and reads it inside the forward pass. Each run directory holds `config.json` "
        "(every training flag, seed and model shape) and MLX safetensors checkpoints.", "",
        "The run records (metrics, per-question evaluation outputs, data fingerprints), the comparison report and "
        "the code to load and evaluate these weights are in the project repository, under `results/step1/`.", "",
        "| run | arm | params | checkpoints | sha256 |", "|---|---|---|---|---|", *rows, "",
        "```sh",
        "pip install -e '.[hub]'",
        "python -m hemispheres.records fetch latent-multi          # into runs/latent-multi/, sha256-checked",
        "python -m hemispheres.synth.build --name world-b --index 1 --edits 1,100,1000",
        "python -m hemispheres.evaluate --run runs/latent-multi --data data/world-b --sets all --n 300",
        "```", "",
        "Models are 25.5M (dense, lookup, context) or 29.2M (latent) parameters, fp32, trained from scratch on "
        "generated worlds. They are research artifacts for the synthetic task, not general-purpose language models.",
        "", "Released under the MIT license (see `LICENSE`).", ""])


def fetch(names: list[str], out: Path, dest: Path) -> None:
    """Download runs' checkpoints from the pinned Hub commit into <dest>/<run>/, verifying sha256."""
    from huggingface_hub import hf_hub_download

    weights = json.loads((out / "weights.json").read_text())
    for name in names:
        if name not in weights["runs"]:
            raise SystemExit(f"{name} was not uploaded; see {out / 'weights.json'}")
        run = dest / name
        run.mkdir(parents=True, exist_ok=True)
        if not (run / "config.json").exists():
            shutil.copy2(out / "runs" / name / "config.json", run / "config.json")
        for tag, f in weights["runs"][name].items():
            target = run / "checkpoints" / tag / "model.safetensors"
            if target.exists() and provenance.sha256(target) == f["sha256"]:
                print(f"{target}: already present")
                continue
            cached = Path(hf_hub_download(weights["repo"], f["path"], revision=weights["revision"]))
            if provenance.sha256(cached) != f["sha256"]:
                raise SystemExit(f"{weights['repo']}/{f['path']}: sha256 does not match the records")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(cached, target)
            print(f"{target}: OK")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("export", "verify-data", "verify-run", "upload", "fetch"))
    p.add_argument("paths", nargs="+", help="run directories (export, verify-run, upload), world directories "
                                            "(verify-data) or run names (fetch)")
    p.add_argument("--out", default="results/step1", help="records directory")
    p.add_argument("--repo", default="SceneWorks/hemispheres-step1", help="Hub model repo (upload)")
    p.add_argument("--public", action="store_true", help="create the Hub repo public (upload; default private)")
    p.add_argument("--dest", default="runs", help="where fetched runs go (fetch)")
    args = p.parse_args()
    out = Path(args.out)
    if args.command == "export":
        for run in args.paths:
            print(f"exported {export(Path(run), out)}")
    elif args.command == "verify-data":
        sys.exit(0 if verify_data(args.paths, out) else 1)
    elif args.command == "verify-run":
        sys.exit(0 if all([verify_run(Path(r), out) for r in args.paths]) else 1)
    elif args.command == "upload":
        w = upload([Path(r) for r in args.paths], out, args.repo, private=not args.public)
        print(f"uploaded to https://huggingface.co/{w['repo']} at {w['revision']}; wrote {out / 'weights.json'}")
    else:
        fetch(args.paths, out, Path(args.dest))


if __name__ == "__main__":
    main()
