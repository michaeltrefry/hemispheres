# Handoff: 2026-09-23 (end of day)

> **Historical note (2026-10-14).** The project now lives at github.com/michaeltrefry/hemispheres, and the weights at `SceneWorks/hemispheres-step1` on the Hugging Face Hub. Paths and links below are as they were on 2026-09-23. The context arm has since been fixed; see `research/context-arm-diagnosis.md`.

Read this first, then `research/results-step1.md` (all numbers) and `research/BRIEF.md` (literature and framing).

## Where things stand

- **Step 1 (synthetic-world toy) is done.** The latent arm reads a fact store inside its forward pass. With per-hop retrieval labels during training, it answers 1–3-hop questions at 100% on unseen worlds and after 1,000 edits, and holds almost no facts itself.
- **New today: hop supervision is necessary.** `latent-multi-nohop` (`--hop-weight 0`) was stopped at step 3,300 of 10,000. Retrieval stayed at chance (1 in 62.6k) and the loss plateaued at 1.13, against 0.51 with hop supervision. The likely cause is the hard top-k cold start: the right fact is never retrieved, so its score gets no gradient. Details are in the "Hop supervision is necessary" section of `results-step1.md`.
- **The question now** is how much hop supervision the model needs. The labels come free with any training data generated from the knowledge base (as in KBLaM and LMLM), so needing them is not fatal. What matters is whether they're needed only to get retrieval started, and whether retrieval learned on templates carries over to other phrasing.

## Git

- HEAD is `910fcdb`, pushed to github.com/zak-keown/hemispheres (public, MIT).
- **Not committed yet** (the user asks before each commit; commit only when asked):
  - `hemispheres/train.py`: the `--hop-until N` flag, which stops hop supervision from step N on. The weight is passed into the compiled step as an array, so switching it doesn't recompile. `hop_weight` is logged in `metrics.jsonl`.
  - `tests/test_latent.py`: `test_hop_supervision_can_stop_partway`. All 41 tests pass (`.venv/bin/python -m pytest -q`).
  - `research/results-step1.md`: the hop-supervision section and reworded claims.
  - `research/HANDOFF.md`: this file.

## Next action: the `latent-multi-hop2k` run (not started)

It's the same as `latent-multi`, except hop supervision stops at step 2,000. At step 2,000, `latent-multi` already retrieved the right fact 99.5 / 98 / 96% of the time at hops 1 / 2 / 3.

**It was held because the GPU was busy with the user's own jobs** (`aether verify` on qwen3-vl-2b, and `drumux/Experiments/YourMT3Probe`). Before starting, check that the GPU is idle and ask the user:

```
.venv/bin/python -c "from bench.throughput import gpu_utilization; print(gpu_utilization())"
pgrep -fl "aether|moeLAR|drumux|YourMT3"
```

The queue script (the scratchpad copy won't survive the session). It takes about 95 minutes to train, plus about 5 minutes of evaluation:

```zsh
#!/bin/zsh
setopt pipefail
cd /Users/zakkeown/Code/hemispheres
PY=.venv/bin/python
q() { grep --line-buffered -v MallocStackLogging; }
run=latent-multi-hop2k
echo "=== train $run $(date +%H:%M:%S)"
$PY -m hemispheres.train --arm latent --supervise all --hop-until 2000 \
  --data "data/world-a,data/pool/w3-s0,data/pool/w3-s1,data/pool/w3-s2,data/pool/w4-s0,data/pool/w4-s1,data/pool/w4-s2,data/pool/w5-s0,data/pool/w5-s1,data/pool/w5-s2,data/pool/w6-s0,data/pool/w6-s1,data/pool/w6-s2,data/pool/w7-s0,data/pool/w7-s1,data/pool/w7-s2" \
  --out runs/$run 2>&1 | q > runs/$run.log
mv runs/$run.log runs/$run/stdout.log
echo "=== eval $run $(date +%H:%M:%S)"
for w in world-b world-c; do $PY -m hemispheres.evaluate --run runs/$run --data data/$w --sets all --n 300 2>&1 | q | sed -n 1,2p; done
for k in 100 1000; do $PY -m hemispheres.evaluate --run runs/$run --data data/world-a --edits $k --n 500 2>&1 | q | sed -n 1,4p; done
$PY -m hemispheres.evaluate --run runs/$run --data data/world-a --sets test_id,test_1hop_ood --n 300 --store none 2>&1 | q | sed -n 1,3p
$PY -m hemispheres.records export runs/$run && $PY -m hemispheres.records upload runs/$run && $PY -m hemispheres.report
echo "HOP2K DONE $(date +%H:%M:%S)"
```

To watch progress, read `runs/latent-multi-hop2k/metrics.jsonl`, which holds `retrieval_top1` per hop every 100 steps and `hop_loss` (still measured after it's switched off). There are evaluations every 2,000 steps. Evaluation results go to `runs/<run>/evals/*.json`.

**How to read the outcome** (the key window is steps 2,000–4,000):

- **Retrieval holds at ~100% and worlds B and C stay at ~100%:** the labels are only needed to get retrieval started, so a short warm-up on data generated from the knowledge base is enough. Go to the harder toy.
- **Retrieval drifts down after step 2,000:** the model needs the labels throughout training. Then try the small, growing store (below) or keep a small hop weight.
- **Either way:** compare with `latent-multi` (`runs/latent-multi/metrics.jsonl`, `evals/`) and add a row to `results-step1.md`.

## Backfill: done (2026-09-25)

The 29 cells in `REPORT.md` that came only from summaries logged during training now have per-question records. These were the world-A held-out results, `dense-a-k100` and `dense-a-to-b`. They were rerun from the saved checkpoints on the same questions: 74 of 74 logged cells came out identical, so evaluation is deterministic here. The report's Integrity section repeats this comparison whenever a logged evaluation is rerun.

## Records

- **Every step-1 number traces to committed records** in `results/step1/`: configs, metrics, logs, per-question evaluation outputs, checkpoint hashes and world fingerprints.
- **`REPORT.md` and `reproduce.sh` are generated** by `python -m hemispheres.report`. `tests/test_records.py` fails if they're stale.
- **Weights** are on the Hub at `hemisphere-llm/hemispheres-step1` (public, MIT). `results/step1/weights.json` pins the commit. `records fetch <run>` downloads a run and checks each file's hash.
- **After every new run or evaluation**, run `python -m hemispheres.records export runs/<run>`. After a new run, also run `records upload runs/<run>` (it needs `pip install -e '.[hub]'` and the `HF_TOKEN`). Then run `python -m hemispheres.report`.
- **Runs now record themselves:** the git commit (and whether the tree was dirty), environment and data fingerprints go in `config.json` and each eval's `.json`. The evaluations run during training also save their per-question records to `evals/<world>-step<N>.jsonl`.

## After that, in order

1. **Harder toy.** This tests the step-2 risk: does retrieval trained on templates carry over to real text?
   - Several phrasings per relation and per question, with some held out for testing only.
   - Noisy name mentions (typos, partial names) that don't exactly match store keys.
   - Facts missing from the store, where the right answer is to abstain.
   - 4-hop questions with 3 read layers.
   - All of this is generator work in `hemispheres/synth/`.
2. **Small, growing store** (only if hop2k drifts). Each batch gets a store of the facts its documents mention plus distractors, growing to the full world. That's set-level fact alignment with no per-layer labels, which is the realistic signal in step 2.
3. **Fix the context oracle.** It trains on only 2% of its tokens and scores better on 3-hop than 1-hop, so something is wrong.
4. **Deletion test**, plus a MEMIT/AlphaEdit baseline (EasyEdit, on a rented GPU).
5. **Step 2:** the read interface attached to a frozen Qwen3 or OLMo model.

## Working rules (from the user)

- **One GPU job at a time.** Running jobs in parallel once pushed the machine into 27 GB of swap. Use sequential queue scripts.
- The user runs their own GPU jobs on this M5 Max (64 GB): `moeLAR`, `aether`, `drumux`. Check the GPU before any run or timing, and don't start runs while theirs are going.
- Commit only when asked. Commit messages end with the Co-Authored-By line.

## Practical notes

- **Environment:** `.venv` (Python 3.12, MLX 0.32.2), `pip install -e '.[dev]'`. Tests run on CPU (`HEMI_DEVICE=cpu` via `tests/conftest.py`).
- **`data/` and `runs/` are git-ignored.** To rebuild the worlds:
  - `python -m hemispheres.synth.build --name world-a --index 0 --edits 1,100,1000`
  - The same for `world-b --index 1` and `world-c --index 2`.
  - The pool: `for idx in 3..7, seed in 0..2: python -m hemispheres.synth.build --name w$idx-s$seed --index $idx --seed $seed --out-dir data/pool`
- **Runs on disk:**
  - `lookup-a`, `dense-a`, `dense-a-k100`, `dense-a-to-b`, `context-a`.
  - `latent-a`, `latent-a-all`, `latent-multi`: the headline model.
  - `latent-multi-nohop`: partial, stopped at step 3,300; checkpoint `latest` at step 2,000.
- **zsh gotchas:** use `${=VAR}` to word-split a variable. `grep` in a pipe needs `--line-buffered`. Filter the `MallocStackLogging` noise from MLX output.
- **Speed:** the small latent model trains at about 29k tok/s at batch 64 × 256, so 10k steps take about 95 minutes. It uses about 20 GB of memory.
