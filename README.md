# Hemispheres

An LLM that splits **reasoning** from **knowledge**. A knowledge-light reasoner reads from a separate store of facts. You add, change or delete a fact by editing the store, not by retraining the reasoner, and the reasoner uses updated facts in multi-step reasoning as well as a normal LLM uses what it memorized.

Start with [`research/BRIEF.md`](research/BRIEF.md). It covers prior art, the gap, the strongest case against the idea, and the experiment plan. The detailed literature reviews are in [`research/tracks/`](research/tracks/).

## Results so far (step 1: synthetic worlds)

Exact-match accuracy in %, 89–900 questions per cell (200 per hop count on world A). Models have 25.5M (baselines) or 29.2M (latent) parameters, trained from scratch on an M5 Max for 10k steps (`context`: 20k, see below). Setup and caveats are in [`research/results-step1.md`](research/results-step1.md). Every number is regenerated from committed run records by [`results/step1/REPORT.md`](results/step1/REPORT.md), which gives each cell's question count, 95% interval and source file (see [Checking the results](#checking-the-results)).

| Test | dense | dense + fine-tune on edits | lookup | context | **latent** |
|---|---|---|---|---|---|
| World A, 1 hop, held-out people | 100 | — | 100 | 99 | **100** |
| World A, 2 / 3 hops, held-out people | 5.5 / 7.5 | — | 100 / 100 | 96 / 97 | **100 / 100** |
| 100 edits: the edited fact | 0 | 100 | 100 | 99 | **100** |
| 100 edits: ripple (multi-hop through an edit) | 2.2 | 14.6 | 100 | 98 | **100** |
| 100 edits: locality (untouched questions) | 33 | 9.2 | 100 | 97.6 | **100** |
| 1,000 edits: ripple | 2.0 | — | 100 | 96.4 | **100** |
| Unseen world B, all hops | 1.1 | — | 99.6 | 97.3 | **100** |
| Unseen world C, all hops | — | — | 99.7 | 97.8 | **100** |
| Store values hidden (lower is better) | n/a | n/a | 0 | n/a | **1–2** |

### How to read the table

- **Each cell** is the share of questions answered exactly right. The whole generated answer must match; there is no partial credit.
- **The columns are where the facts live.**
  - `dense`: in the weights, like a normal LLM. This is the baseline to beat.
  - `dense + fine-tune on edits`: `dense` trained for 200 more steps on the 100 edited facts. This is the usual way to update a model.
  - `lookup`: in a separate store. The model writes visible `[LOOKUP] subject @relation` calls and the store fills in the answer. This is the ceiling: an explicit tool call per hop.
  - `context`: in the prompt. The model is shown the facts it needs plus six distractors and must copy the right answer (the in-context oracle). Trained on the same 16 worlds as `latent-multi` and for 20k steps, twice the others' budget; the column shows seed 0 of `context-multi-bios-20k` (seed 1 is 1–5 points lower on every row).
  - **`latent`**: in a separate store, read *inside* the forward pass, with no lookup tokens. This is the idea being tested. The column shows the headline run, `latent-multi`.
- **The rows test different things.**
  - **Held-out people**: people who appear in no multi-hop training question, so a right answer means the model composed facts rather than recalled a memorized answer. A "hop" is one fact in the chain, e.g. "the capital of the country of X's birthplace" is 3 hops.
  - **Edits** change facts in the store, or in the weights for `dense + fine-tune`. A good edit scores 100 on all three rows: the edited fact itself, multi-hop questions that pass through it (ripple), and questions it shouldn't affect (locality).
  - **Unseen world B / C**: swap in the store of a world the model never trained on, with all-new people, companies and places. A high score means reasoning and knowledge really are separate.
  - **Store values hidden**: the same world-A questions with the store emptied. Near 0 means the reasoner holds almost no facts of its own. Lower is better here.
- **"—"** means not run. A dense model can only learn a new world by retraining.

**The short version:** `dense` can't chain facts it memorized (the "two-hop curse"), and fine-tuning in edits barely propagates them and damages unrelated knowledge. `latent` matches the explicit-lookup ceiling everywhere, without writing any lookup calls.

### What failed, and the caveats

- **Without per-hop retrieval labels, the latent arm never learns to retrieve.** Every `latent` result above uses a training loss that tells each read layer which fact to fetch. With that loss switched off, retrieval stayed at chance (0.0016%) after 3,300 steps and the run was stopped. The labels come free with any training data generated from the knowledge base, but whether they're needed throughout training, or only to get retrieval started, is the next experiment.
- **The first latent model scored 42% on unseen world B.** It retrieved the right fact but misspelled names it had never seen, completing them from world-A spelling patterns. Supervising retrieval at every name token raised this to 86%. Training across 16 worlds raised it to 100%.
- **Models trained on world A alone can't write answer values world A never uses.** All 4 of `lookup`'s world-B misses are currencies (shilling, ducat) that no world-A country uses: it looked up the right value, then wrote a different one. `latent-a` and `latent-a-all` miss the same 4. Multi-world training fixes this too.
- **The in-context oracle arm needed a different recipe.** Trained as first published (`context-a`: questions only, no bios, world A, 10k steps) it scores 21.5% on 1 hop and 8–14% on unseen worlds: it never learns to copy from its prompt, and instead emits a familiar entity of the right type from its weights. With bios in the mix (like every other arm), 16 worlds and a 20k-step schedule it reaches 97.5–99% on 1 hop over two seeds. The copy circuit forms in a late, seed-dependent jump (step 4k–8k), so the 10k cosine schedule is not enough. Diagnosis and controls are in [issue #1](https://github.com/zak-keown/hemispheres/issues/1).
- **This is a toy.** Questions use fixed templates, names match store keys exactly, the store is always complete and correct, and no question needs more hops than the model has read layers. The latent arm has 14% more parameters than the baselines, and the only edit baseline is naive fine-tuning, not MEMIT or AlphaEdit.

## Layout

| Path | What |
|---|---|
| `research/` | Literature review and research brief |
| `hemispheres/model.py` | Dense GPT baseline in MLX (RMSNorm, RoPE, tied embeddings) |
| `hemispheres/synth/` | Synthetic-world generator: worlds, question splits, counterfactual edits, renderers |
| `hemispheres/train.py` | Training loop for all four arms |
| `hemispheres/latent.py`, `store.py` | Latent-store reasoner (read layers, key encoder) and a world's facts as store arrays |
| `hemispheres/evaluate.py` | Exact-match evaluation: held-out splits, world swap, counterfactual edits |
| `hemispheres/data.py`, `generate.py`, `checkpoint.py` | Batching, decoding with the store in the loop, run directories |
| `tests/` | Invariant tests for the generator and training plumbing (`pytest`) |
| `bench/throughput.py` | Training-throughput benchmark; sizes every experiment |
| `bench/latent_profile.py` | Where a latent-arm training step spends its time |
| `results/` | Benchmark outputs (JSON + Markdown) |
| `results/step1/` | Step-1 run records (configs, metrics, per-question eval outputs, checkpoint and data hashes), the generated comparison report and `reproduce.sh` |
| `hemispheres/records.py`, `report.py`, `provenance.py` | Export run records, rebuild the report from them, and fingerprint code, data and weights |

## Setup

Requires Apple Silicon and Python 3.11+.

```sh
python3.12 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest -q tests
HEMI_DEVICE=cpu .venv/bin/python -m pytest -q tests   # leave the GPU free for training
```

## Checking the results

`runs/` and `data/` are git-ignored: checkpoints are ~100 MB each, and worlds rebuild deterministically in seconds. The checkpoints are on the Hugging Face Hub at [`hemisphere-llm/hemispheres-step1`](https://huggingface.co/hemisphere-llm/hemispheres-step1). What the results depend on is committed under [`results/step1/`](results/step1/):

- `runs/<run>/config.json`: every training flag, the seed and the model shape.
- `runs/<run>/metrics.jsonl` and `stdout.log`: the training log, including the evaluations run during training.
- `runs/<run>/evals/*.json` and `*.jsonl`: each evaluation's arguments and summary, plus one record per question with the model's full output.
- `runs/<run>/provenance.json`: sha256 of every checkpoint, the fingerprint of every world the run read, and the code version.
- `worlds.json`: each world's build config and file hashes.
- `weights.json`: the Hub repo and commit holding each run's checkpoints.

`REPORT.md` and `reproduce.sh` are generated from these records. The report tool uses only the standard library, so checking needs Python 3.11+ but no MLX or Apple hardware:

```sh
python -m hemispheres.report --check     # recount every table from the records; fail if REPORT.md is stale
python -m hemispheres.records verify-data data/world-a   # a rebuilt world is byte-identical to the one used
pip install -e '.[hub]' && python -m hemispheres.records fetch latent-multi   # our weights, sha256-checked, into runs/
results/step1/reproduce.sh               # rebuild, retrain and re-evaluate everything (~10 GPU-hours on an M5 Max)
python -m hemispheres.report --runs runs --out repro-step1   # the same report from your runs, to diff
```

New runs record the git commit, environment and data fingerprints themselves. After a run, publish its records with `python -m hemispheres.records export runs/<run>` and its weights with `python -m hemispheres.records upload runs/<run>` (needs a Hub token with write access), then run `python -m hemispheres.report`.

## Throughput benchmark

```sh
.venv/bin/python bench/throughput.py            # full sweep, ~15 min
.venv/bin/python bench/throughput.py --quick    # smoke test, ~1 min
.venv/bin/python bench/throughput.py --sustain 600          # 10-min thermal-throttling test (gpt2, batch 16)
.venv/bin/python bench/throughput.py --sizes medium,large --batches 8,16
```

The benchmark measures peak matmul TFLOPS per dtype. It then trains GPT models from ~11M to ~300M non-embedding parameters (add `large` for ~700M) on random tokens, reporting tokens/sec, MFU against the measured matmul peak, and peak memory. Each configuration runs in its own subprocess, so an out-of-memory crash loses only that row. Run it plugged in.

## Synthetic worlds

Step 1 of the plan runs on generated worlds. Each world is a knowledge graph of 10k people, 800 companies, 60 universities, 200 cities and 25 countries, linked by 13 relations such as `works_at`, `mentor`, `hq`, `country` and `capital`. That gives about 62k facts and 32 question paths of 1 to 3 hops.

```sh
.venv/bin/python -m hemispheres.synth.build --name world-a --index 0 --edits 1,100,1000
.venv/bin/python -m hemispheres.synth.build --name world-b --index 1 --edits 1,100,1000
```

- **Worlds with different `--index` values share no entity.** Names are hash-partitioned, so worlds can be generated independently and never collide.
- **All worlds share every token.** Names are built from about 100 shared syllable tokens, and the vocabulary is 544 tokens. A reasoner trained on world A has seen every token world B uses, so a world swap tests unseen *facts*, not unseen embeddings.
- **The same facts render four ways, one per experimental arm:**
  - plain bios and questions (dense);
  - bios and questions with inline `[LOOKUP] subject @relation [RESULT] value [END]` calls, where store-supplied tokens are excluded from the loss (lookup arm);
  - raw triples (latent store);
  - gold facts plus distractors in the prompt (in-context oracle).
- **Splits follow the Grokked Transformers in-distribution vs. out-of-distribution protocol.** 20% of people appear in no multi-hop training question (`test_ood`).
- **Edit sets are nested.** The 100-edit set extends the 1-edit set. Each carries `direct` questions (the edited fact itself), `ripple` questions (multi-hop questions that pass through an edited fact) and `locality` questions (untouched, so their answers must not change). `--edit-relations works_at,mentor,hq` concentrates edits on facts that other questions pass through.

Output goes to `data/<name>/` (git-ignored; regenerate deterministically). Check `samples.txt` first: it shows one example of every rendering, along with which tokens are trained on.

## Training and evaluation

Four arms are implemented. Each trains a reasoner from scratch on world A:

| Arm | Where facts live | Trained on |
|---|---|---|
| `dense` | in the weights | bios + direct QA (answer tokens only) |
| `lookup` | in the store; the model writes `[LOOKUP] subject @relation [RESULT]` and the store answers | bios with lookups + QA with one lookup per hop; store tokens are never trained on |
| `context` | in the prompt (in-context oracle) | bios + QA with gold facts + distractors in the prompt |
| `latent` | in the store, read **inside the forward pass**: read layers retrieve facts by learned query · key and cross-attend over the retrieved objects' name tokens ([design](research/latent-arm-design.md)) | bios + direct QA, plus an optional retrieval loss per hop (`--hop-weight`) |

```sh
.venv/bin/python -m hemispheres.train --arm lookup --data data/world-a --out runs/lookup-a
.venv/bin/python -m hemispheres.evaluate --run runs/lookup-a --data data/world-a                      # held-out splits
.venv/bin/python -m hemispheres.evaluate --run runs/lookup-a --data data/world-b --sets all           # world swap
.venv/bin/python -m hemispheres.evaluate --run runs/lookup-a --data data/world-a --edits 100          # edits
```

The context arm needs the multi-world pool (built below) and twice the default steps; its copy circuit forms late. This is the recipe behind the `context` column:

```sh
.venv/bin/python -m hemispheres.train --arm context --steps 20000 --out runs/context-multi-bios-20k \
    --data data/world-a,data/pool/w3-s0,data/pool/w3-s1,data/pool/w3-s2,data/pool/w4-s0,data/pool/w4-s1,data/pool/w4-s2,data/pool/w5-s0,data/pool/w5-s1,data/pool/w5-s2,data/pool/w6-s0,data/pool/w6-s1,data/pool/w6-s2,data/pool/w7-s0,data/pool/w7-s1,data/pool/w7-s2
```

A dense model can only learn a new world or edits by further training. This is its baseline:

```sh
.venv/bin/python -m hemispheres.train --arm dense --data data/world-a --edits 100 --init runs/dense-a \
    --mix edit_facts=1,edit_qa=1 --steps 200 --eval-sets direct,ripple,locality --out runs/dense-a-k100
```

**Latent-arm options.**
- `--supervise all|first` supervises retrieval at every position that writes a name token (default), or only before each name starts.
- `--data` also takes a comma list of worlds for multi-world training. Each step draws its batch and its store from one world, and the first world is the one evaluated during training. A training pool disjoint from worlds B and C:

```sh
for idx in 3 4 5 6 7; do for seed in 0 1 2; do
  .venv/bin/python -m hemispheres.synth.build --name w$idx-s$seed --index $idx --seed $seed --out-dir data/pool
done; done
```

Scoring is exact match on generated text, never token-by-token scoring. For the lookup arm, `trace` also reports whether every lookup queried the right (subject, relation). Defaults: `small` model (29M), fp32, batch 64 × 256 tokens, 10k steps, AdamW with warmup and cosine decay. Runs checkpoint every 2k steps, and `--resume` continues one.

## License

MIT; see [`LICENSE`](LICENSE). The step-1 weights on the Hugging Face Hub are released under the same license.
