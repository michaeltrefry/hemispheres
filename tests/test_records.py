import json
from pathlib import Path

import pytest

from hemispheres import records, report
from hemispheres.synth import WorldSizes, generate_world, vocabulary
from hemispheres.synth.build import make_splits

REPO = Path(__file__).resolve().parent.parent


def _world_dir(root: Path, name: str, index: int) -> Path:
    d = root / "data" / name
    d.mkdir(parents=True)
    w = generate_world(name, index, sizes=WorldSizes(persons=150, companies=15, universities=4, cities=10, countries=3))
    (d / "world.json").write_text(json.dumps(w.to_json()))
    (d / "vocab.json").write_text(json.dumps(vocabulary()))
    (d / "splits.json").write_text(json.dumps({s: [list(q) for q in qs] for s, qs in make_splits(w).items()}))
    (d / "stats.json").write_text(json.dumps({"config": {"name": name, "index": index, "seed": 0},
                                              "facts": len(w.facts)}))
    return d


def test_committed_report_matches_its_records():
    """results/step1/REPORT.md and reproduce.sh are regenerated from the committed records,
    and every eval summary agrees with its per-question records."""
    runs = report.load_runs(REPO / "results/step1/runs")
    worlds = json.loads((REPO / "results/step1/worlds.json").read_text())
    weights_path = REPO / "results/step1/weights.json"
    weights = json.loads(weights_path.read_text()) if weights_path.exists() else None
    _, bad = report.integrity(runs)
    assert not bad
    assert (REPO / "results/step1/REPORT.md").read_text() == report.report(runs, worlds, Path("results/step1/runs"),
                                                                          weights)
    assert (REPO / "results/step1/reproduce.sh").read_text() == report.reproduce_script(runs, worlds, weights)


def test_wilson_interval():
    lo, hi = report.Cell(200, 200, "").wilson()
    assert hi == 1.0 and lo == pytest.approx(0.9812, abs=1e-4)
    lo, hi = report.Cell(0, 900, "").wilson()
    assert lo == 0.0 and hi == pytest.approx(0.00425, abs=1e-4)


def test_train_export_verify_and_report(tmp_path, monkeypatch):
    from hemispheres import evaluate, train
    monkeypatch.chdir(tmp_path)
    a, b = _world_dir(tmp_path, "world-a", 0), _world_dir(tmp_path, "world-b", 1)
    run = Path("runs/latent-t")
    monkeypatch.setattr("sys.argv", ["train", "--arm", "latent", "--data", "data/world-a", "--out", str(run),
                                     "--size", "tiny", "--steps", "2", "--batch-size", "2", "--seq-len", "128",
                                     "--log-every", "1", "--eval-every", "2", "--eval-n", "2", "--save-every", "0",
                                     "--warmup", "1"])
    train.main()
    config = json.loads((run / "config.json").read_text())
    assert set(config["provenance"]) >= {"code", "env", "data", "time"}
    assert config["provenance"]["data"]["data/world-a"] == records.provenance.world_fingerprint(a)
    # The training loop's final evaluation keeps its per-question records.
    step_eval = json.loads((run / "evals" / "world-a-step2.json").read_text())
    assert step_eval["args"]["step"] == 2
    assert len((run / "evals" / "world-a-step2.jsonl").read_text().splitlines()) == step_eval["summary"]["test_ood"]["all"]["n"] + \
        step_eval["summary"]["test_id"]["all"]["n"] + step_eval["summary"]["test_1hop_ood"]["all"]["n"]

    monkeypatch.setattr("sys.argv", ["evaluate", "--run", str(run), "--data", "data/world-b", "--sets", "all",
                                     "--n", "2"])
    evaluate.main()
    ev = json.loads((run / "evals" / "world-b-final.json").read_text())
    assert ev["provenance"]["data"] == {"data/world-b": records.provenance.world_fingerprint(b)}
    assert ev["provenance"]["checkpoint_sha256"] == records.provenance.sha256(
        run / "checkpoints" / "final" / "model.safetensors")

    # Rerunning the evaluation logged at the last step reproduces it exactly.
    monkeypatch.setattr("sys.argv", ["evaluate", "--run", str(run), "--data", "data/world-a", "--n", "2"])
    evaluate.main()

    out = Path("results/step1")
    records.export(run, out)
    meta = json.loads((out / "runs" / run.name / "provenance.json").read_text())
    assert set(meta["data"]) == {"data/world-a", "data/world-b"}
    assert meta["code"] == config["provenance"]["code"]
    assert records.verify_run(run, out)
    assert records.verify_data(["data/world-a", "data/world-b"], out)
    (a / "splits.json").write_text("[]")
    assert not records.verify_data(["data/world-a"], out)

    runs = report.load_runs(out / "runs")
    assert report.integrity(runs) == (3, [])
    compared, agree, differ = report.reevaluations(runs)
    assert compared == ["latent-t"] and agree >= 6 and differ == []
    cell = runs["latent-t"].cell("world-b", ("all",))
    assert cell.source == "evals/world-b-final.jsonl" and not cell.logged
    worlds = json.loads((out / "worlds.json").read_text())
    assert worlds["data/world-a"]["facts"] == json.loads((a / "stats.json").read_text())["facts"]
    rec = json.loads((run / "evals" / "world-b-final.jsonl").read_text().splitlines()[0])
    assert rec["prompt_tokens"] > 0 and rec["output_tokens"] > 0
    script = report.reproduce_script(runs, worlds)
    assert "--name world-a --index 0 --seed 0" in script
    assert "-m hemispheres.train --arm latent" in script and "--out runs/latent-t" in script
    assert "-m hemispheres.evaluate --run runs/latent-t --checkpoint final --data data/world-b" in script
    assert "world-a-step2" not in script  # training-loop evals come from the training command


def test_fetch_verifies_weights_against_the_records(tmp_path, monkeypatch):
    pytest.importorskip("huggingface_hub")
    out = tmp_path / "results"
    (out / "runs" / "r").mkdir(parents=True)
    (out / "runs" / "r" / "config.json").write_text('{"arm": "dense"}')
    blob = tmp_path / "hub" / "model.safetensors"
    blob.parent.mkdir()
    blob.write_bytes(b"weights")
    sha = records.provenance.sha256(blob)
    files = {"final": {"path": "r/checkpoints/final/model.safetensors", "sha256": sha},
             "latest": {"path": "r/checkpoints/final/model.safetensors", "sha256": sha}}
    (out / "weights.json").write_text(json.dumps({"repo": "org/repo", "revision": "abc", "runs": {"r": files}}))
    calls = []
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda repo, path, revision: calls.append(revision) or str(blob))

    records.fetch(["r"], out, tmp_path / "runs")
    for tag in ("final", "latest"):
        assert (tmp_path / "runs" / "r" / "checkpoints" / tag / "model.safetensors").read_bytes() == b"weights"
    assert json.loads((tmp_path / "runs" / "r" / "config.json").read_text()) == {"arm": "dense"}
    assert calls == ["abc", "abc"]  # pinned to the recorded commit
    records.fetch(["r"], out, tmp_path / "runs")
    assert len(calls) == 2  # already present and verified: nothing downloaded

    blob.write_bytes(b"tampered")
    with pytest.raises(SystemExit, match="does not match"):
        records.fetch(["r"], out, tmp_path / "elsewhere")


def test_code_version_counts_untracked_files_but_not_ignored_ones(tmp_path):
    import subprocess

    def run(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)
    run("init", "-q")
    (tmp_path / ".gitignore").write_text("/runs/\n")
    (tmp_path / "a.py").write_text("x = 1\n")
    run("add", ".")
    run("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init")
    code_version = records.provenance.code_version
    assert code_version(tmp_path)["dirty"] is False
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "log.txt").write_text("ignored")
    assert code_version(tmp_path)["dirty"] is False
    (tmp_path / "new_module.py").write_text("y = 2\n")
    v = code_version(tmp_path)
    assert v["dirty"] is True and v["changed"] == ["new_module.py"]
    (tmp_path / "a.py").write_text("x = 3\n")
    assert code_version(tmp_path)["changed"] == ["a.py", "new_module.py"]


def test_eval_command_keeps_the_evaluation_name():
    from types import SimpleNamespace

    from hemispheres.report import eval_command
    run = SimpleNamespace(name="r")
    args = {"checkpoint": "final", "data": "data/w", "sets": "test_1hop_ood", "n": 1000, "seed": 0,
            "name": "w-heldout-final"}
    assert eval_command(run, args).endswith("--seed 0 --name w-heldout-final")
    assert "--name" not in eval_command(run, {**args, "name": None})


def _fake_run(root, name, config, logged, evals):
    """A run record directory: `logged` one-hop accuracies by step, `evals` name -> (args, records)."""
    d = root / name
    (d / "evals").mkdir(parents=True)
    base = {"arm": "dense", "data": "data/world-b", "steps": 100, "seed": 0, "mix": [["bios", 0.5], ["qa", 0.5]],
            "dtype": "float32", "model_type": "gpt", "model": {"n_layer": 2, "d_model": 8}, "params": 1000,
            "save_every": 100}
    (d / "config.json").write_text(json.dumps({**base, **config}))
    (d / "metrics.jsonl").write_text("".join(json.dumps({"step": s, "eval": {"test_1hop_ood": {"1hop": {"acc": a}}}})
                                             + "\n" for s, a in logged))
    for ename, (args, recs) in evals.items():
        (d / "evals" / f"{ename}.json").write_text(json.dumps({"args": args, "summary": {}}))
        (d / "evals" / f"{ename}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))


def _one_hop(n_correct, n=20, tokens=True):
    extra = {"prompt_tokens": 12, "output_tokens": 4} if tokens else {}
    return [{"set": "test_1hop_ood", "hops": 1, "correct": i < n_correct, **extra} for i in range(n)]


def test_capacity_picks_the_smallest_passing_size_and_charts_it(tmp_path):
    final = {"data": "data/world-b", "checkpoint": "final", "seed": 0}
    _fake_run(tmp_path, "dense-a1", {"size": "a1", "params": 1e5}, [(50, 0.5), (100, 0.6)],
              {"world-b-final": (final, _one_hop(12))})
    _fake_run(tmp_path, "dense-a2", {"size": "a2", "params": 2e5}, [(50, 0.9), (100, 0.98)],  # passes, not converged
              {"world-b-final": (final, _one_hop(20))})
    _fake_run(tmp_path, "dense-a3", {"size": "a3", "params": 4e5}, [(50, 0.99), (100, 1.0)],
              {"world-b-final": (final, _one_hop(20))})
    _fake_run(tmp_path, "dense-a3-s1", {"size": "a3", "params": 4e5, "seed": 1}, [(50, 1.0), (100, 1.0)],
              {"world-b-step100": ({**final, "step": 100}, _one_hop(20))})
    _fake_run(tmp_path, "latent-x", {"arm": "latent", "data": "data/world-a", "params": 3e5, "model_type": "latent",
                                     "model": {"n_layer": 2, "d_model": 8, "n_reads": 3, "key_dim": 4}}, [],
              {"world-b-heldout-final": (final, _one_hop(20))})
    runs = report.load_runs(tmp_path)
    worlds = {"data/world-b": {"build": {"persons": 10000}, "facts": 62000}}
    md, cells, chart = report.capacity(runs, worlds)
    summary = next(line for line in md if line.startswith("| world-b |"))
    assert "a3 (0.40M)" in summary and "| a2 |" in summary and "latent-x 100" in summary
    assert next(line for line in md if line.startswith("| a2 |")).endswith("100* fail |")
    assert next(line for line in md if line.startswith("| a3 |")).endswith("100 / 100 pass |")
    assert {c.source for _, _, c in cells} >= {"evals/world-b-final.jsonl", "evals/world-b-step100.jsonl"}
    assert chart["dense"] == [(62000, 4e5, True)] and chart["latent"] == {"latent-x": (3e5, [(62000, True)])}
    assert report.capacity_svg(chart).startswith("<svg")
    cost = "\n".join(report.cost_table(runs, worlds))
    assert "| latent-x | world-b | 0.30M | 16.0 |" in cost


def test_question_cost_counts_weights_attention_and_store_keys():
    dense = {"params": 1000, "dtype": "float32", "model_type": "gpt", "model": {"n_layer": 2, "d_model": 8}}
    c = report.question_cost(dense, prompt_tokens=5, output_tokens=3, facts=None)
    # 7 tokens through the model in 3 forward passes: 2*1000*7 + 2*2*8*7*8 operations, 1000*4 bytes per pass.
    assert c == {"model_flops": 15792, "weight_bytes": 12000, "store_flops": 0, "store_bytes": 0}
    latent = {**dense, "model_type": "latent", "model": {"n_layer": 2, "d_model": 8, "n_reads": 3, "key_dim": 4}}
    c = report.question_cost(latent, prompt_tokens=5, output_tokens=3, facts=10)
    assert c["store_flops"] == 2 * 3 * 10 * 4 * 7 and c["store_bytes"] == 3 * 10 * 4 * 4 * 3
