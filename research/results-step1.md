# Step 1 results: synthetic-world toy

*2026-09-23. All runs on an M5 Max (MLX 0.32.2, fp32). Each run's records are in [`results/step1/runs/`](../results/step1/runs/). [`results/step1/REPORT.md`](../results/step1/REPORT.md) regenerates every table below from them, with question counts, 95% intervals, source files and the code version of each run.*

## Setup

- **Worlds.** Typed knowledge graphs with 10k people, 800 companies, 60 universities, 200 cities and 25 countries, and 13 relations (~62.6k facts). Worlds A, B and C share every token but no entity names (`hemispheres/synth`). The multi-world pool adds 15 more worlds from name partitions 3–7, disjoint from B and C.
- **Questions.** 1–3-hop, e.g. "the capital of the country of the birthplace of X". Scored by exact match on the generated answer, never token-by-token scoring. The `test_ood` split asks about people who appear in no multi-hop training question.
- **Models.** GPT, "small" size (8 layers, d=512): 25.5M parameters. The latent arm adds 3 read layers and a key encoder, for 29.2M (+14%). Every run: 10k steps, batch 64 × 256 tokens, AdamW 1e-3 with cosine decay.

| Arm | Where facts live | How it answers |
|---|---|---|
| dense | weights | directly |
| lookup | store; the model writes `[LOOKUP] subject @rel [RESULT]` and the store inserts the value | one written lookup per hop, then the answer |
| context | prompt (gold facts + 6 distractors) | directly (in-context oracle; needs bios in the mix, 16 worlds and 20k steps, see below) |
| latent | store, read **inside the forward pass**: 3 read layers retrieve by learned query · key and cross-attend over retrieved name tokens | directly, with no lookup tokens |

Latent variants, changing one thing at a time:

- `latent-a`: world A only; retrieval supervised only before a name starts.
- `latent-a-all`: supervision at every name token (`--supervise all`).
- `latent-multi`: as `latent-a-all`, trained across 16 worlds, each step drawing its batch and store from one world.

All latent runs use hop supervision: an InfoNCE loss on each read layer's query, weight 0.5.

## Main results

Question counts per cell: world A 200 per hop count; 100 edited facts (500 for 1,000 edits); ripple and locality 500 each; worlds B and C 900. The world-A cells and `dense + FT on edits` were first logged during training without per-question records. They were rerun from the saved checkpoints on 2026-09-25, and all 74 logged cells came out identical.

| | lookup | dense | dense + FT on edits | latent-a | latent-a-all | **latent-multi** |
|---|---|---|---|---|---|---|
| World A, 1 hop, held-out people | 100 | 100 | — | 100 | 100 | **100** |
| World A, 2 / 3 hop, held-out people | 100 / 100 | 5.5 / 7.5 | — | 100 / 99 | 100 / 100 | **100 / 100** |
| 100 edits: edited fact | 100 | 0 | 100 | 98 | 100 | **100** |
| 100 edits: ripple (multi-hop through an edit) | 100 | 2.2 | 14.6 | 99.4 | 100 | **100** |
| 100 edits: locality (untouched questions) | 100 | 33 | 9.2 | 99.8 | 100 | **100** |
| 1,000 edits: ripple | 100 | 2.0 | — | 99.8 | 100 | **100** |
| World B swap (unseen), all hops | 99.6 | 1.1 | — | 42 | 86 | **100** |
| World C swap (unseen), all hops | 99.7 | — | — | 39 | 88 | **100** |

- "dense + FT on edits": `dense-a` fine-tuned for 200 steps on the 100 edited facts. It learns them (100%), but they barely propagate (14.6%), and it destroys unrelated knowledge: untouched 1-hop accuracy falls to 25%. This is the pattern MEMIT, fine-tuning and MQuAKE show for real models.
- "dense → world B": 2,000 steps of continued training on world B's bios gave 0% on world B. That's a weak baseline (bios only, short), not strong evidence.
- In every latent run, retrieval accuracy at the answer position is 100% at every hop on every world by the end of training. In the early `latent-a` smoke run it was far lower.

## The latent arm's world-swap failure and its fix

In `latent-a`, retrieval on world B was already 100% at every hop. Answers failed on multi-token names:

| World B accuracy by answer type | latent-a | latent-a-all | latent-multi |
|---|---|---|---|
| people | 0 | 99 | 100 |
| companies | 36 | 100 | 100 |
| universities | 4 | 77 | 100 |
| cities | 11 | 69 | 100 |
| countries | 4 | 66 | 100 |
| currencies | 78 | 78 | 100 |
| single-token values (years, majors, industries) | 100 | 100 | 100 |

The currency misses are all the shilling and the ducat, which no world-A country uses. A model trained on world A alone never writes them, and `lookup-a` misses the same 4 of 18 even though its lookup returns the right value.

The model copied the first name token from the retrieved fact and completed the name from world-A spelling patterns, e.g. gold "Boru Krestizor", output "Boru Vizi". Two changes fixed it:

1. **Supervising retrieval at every name token** took world B from 42 to 86%. The remaining errors were name endings, e.g. "Bragorus" became "Bragoru" and "Bizir College" became "Bizir Institute".
2. **Multi-world training** removed the remaining errors: 100% on worlds B and C.

## Leakage: do the reasoners hold facts themselves?

These use world-A questions. `--store none` hides every store value, and `--store data/world-c` serves world C's facts instead.

| | own store | values hidden | world C's store |
|---|---|---|---|
| lookup-a | 100 | 0 | 0 |
| latent-a | 99–100 | 2–7 | 0–2 |
| latent-a-all | 100 | 1–5 | 1–2 |
| latent-multi | 100 | 1–2 | 1–2 |

Every store-based model falls to chance without its store, including `latent-a`, which saw only world A. So `latent-a`'s world-swap failure was not memorized *facts*: it had learned how world-A names are *spelled*, not which fact belongs to whom. This is the first evidence in this project that the reasoner can be close to knowledge-free, at least for this entity-level knowledge.

## Hop supervision is necessary

`latent-multi-nohop` repeats `latent-multi` with `--hop-weight 0`: nothing tells a read layer which fact to fetch, so retrieval can only be learned from the next-token loss. It was stopped at step 3,300 of 10,000 because nothing had emerged.

| | latent-multi (hop weight 0.5) | latent-multi-nohop (hop weight 0) |
|---|---|---|
| Retrieval top-1, hops 1 / 2 / 3, step 500 | 97 / 45 / 1.5 | 0 / 0 / 0 |
| Retrieval top-1, hops 1 / 2 / 3, step 3,000 | 99.8 / 98.7 / 98.6 | 0.001 / 0.01 / 0.16 |
| Training loss, step 3,000 | 0.51 | 1.13 (flat since step ~2,000) |
| World A held-out people at step 2,000: 1 / 2 / 3 hop | 94.5 / 86 / 90.5 | 8 / 6 / 4 |

Retrieval stays at chance (1 in 62.6k is 0.0016%). The model ignores the store and does worse than `dense-a`, because world A is only a sixteenth of its training data, so it can't memorize it either. The unused InfoNCE loss, logged but not trained on, rose from 11 to a peak of 113 at step 1,700 (79 at step 3,000): queries and keys grow without lining up.

The likely cause is the hard top-k. Each read keeps 4 of 62.6k entries, and the retrieval score gets gradient only through the entries it keeps. At initialization the right fact almost never makes the top 4, so its score is never pushed up. This is the standard cold-start problem of learned retrieval (REALM pretrains its retriever for this reason), so more steps would not have fixed it.

**So every step-1 latent result depends on per-hop retrieval labels.** In the toy they are free, because every question is generated from a known chain. They are also free for any training data generated from the knowledge base itself, as in KBLaM and LMLM. Still open: whether the labels are needed only to get retrieval started, and whether retrieval learned on templated questions carries over to other phrasing.

## Capacity curve: parameters needed versus facts known

*2026-11-25 (HEM-F1). Dense runs on the M5 Max and the Mac mini M6, latent cores on both; MLX, fp32. Every cell below is in [`REPORT.md`](../results/step1/REPORT.md#capacity-parameters-needed-versus-facts-known) with its source records.*

The question: as a world grows from 10,000 to 100,000 people, does a model that keeps facts in its weights need more parameters, while a core that reads them from a store does not?

- **Worlds.** World B at 10,000, 30,000 and 100,000 people (62,615, 182,381 and 601,640 facts), built with `--persons`; the other entity counts stay at their defaults.
- **Pass mark.** At least 95% one-hop accuracy on 1,000 questions about held-out people, on every seed. A dense run also has to be converged: its last two training evaluations within 1 point.
- **Dense ladder.** Sizes 250k to 16m (0.23M to 16.15M parameters), each about double the last, trained on the world's bios and questions for 150 biography passes, so larger worlds train for proportionally more steps.
- **Latent cores.** Sizes 250k, 500k, 1m and 4m (0.32M to 4.52M parameters, read layers included), plus the original 29.2M `latent-multi`, trained once with the 16-world recipe on 10,000-person worlds (10,000 steps), then evaluated on the larger worlds with no further training, each with its own store. They are judged with their store keys compressed to 4 bits with TurboQuant (`evaluate --key-bits 4`); full-precision keys are the reference.
- **Second seeds** at every pass/fail boundary.

### Parameters needed

| people | facts | smallest passing dense model | largest failing dense model | smallest passing latent core, 4-bit keys | largest failing core | smallest passing latent core, full-precision keys |
|---|---|---|---|---|---|---|
| 10,000 | 62,615 | 4m, 3.74M (100 / 99.9) | 2m (97.6 / 95.2, seed 1 unconverged) | 500k, 0.67M (95.6 / 95.9) | 250k (88.3 / 89.7) | 250k, 0.32M (98.9 / 99.3) |
| 30,000 | 182,381 | 4m, 3.74M (97.2 / 98.0) | 2m (37 / 45.5) | 1m, 1.34M (97.6 / 96.9) | 500k (93.6 / 93.7) | 250k, 0.32M (98.7 / 98.8) |
| 100,000 | 601,640 | 16m, 16.15M (97.8 / 98.2) | 8m (74.7 / 73.8, both unconverged) | 1m, 1.34M (96.3 / 95.9) | 500k (87.4 / 87.3) | 250k, 0.32M (96.9 / 97.1) |

One-hop accuracy in % for seed 0 / seed 1. The dense boundary at 10,000 people rests on the every-seed rule: 2m passes on seed 0 (97.6%, converged) but its seed 1 scored 95.2% while still rising (94.3 then 96.1 at its last two training evaluations), so 2m is not shown to pass. The rule was fixed before these runs and is kept.

Two limits of the convergence check. At 30,000 people the last two training evaluations fall at steps 32,000 and 32,067, only 67 steps apart, so the check there cannot catch a slow rise: 4m gained 2.3 / 1.2 points over the 4,000 steps before them. Its pass does not depend on this, since more training would only raise it. At 100,000 people 8m was still rising (71.5 to 74.7% and 72.6 to 74.2% over its last 5,508 steps), so with more training the boundary there might be 8m (7.56M) rather than 16m. Sizes below 8m were not trained at 100,000 people; they are assumed to fail because 8m does.

### The thesis, as pre-registered

The thesis is supported if the dense model's smallest passing size grows with the number of facts while the latent core loses no more than 3 points of one-hop accuracy between 10,000 and 100,000 people. "The latent core" is the smallest one that passes 10,000 people.

1. **Dense: grows.** 3.74M at 10,000 and 30,000 people, 16.15M at 100,000: 4.3 times the parameters for 9.6 times the facts (2.0 times if 8m would pass with more training). The ladder doubles at each step, so it cannot show growth smaller than 2 times; 4m holding 30,000 people means the 10,000 to 30,000 step needed less than double.
2. **Latent core with 4-bit keys (the rule as written, judging latent cores with 4-bit keys): not supported.** The smallest core passing 10,000 people is 500k, and it falls from 95.6 / 95.9% to 87.4 / 87.3% at 100,000 people, a loss of 8.2 / 8.6 points on the two seeds.
3. **Latent core with full-precision keys: supported.** The smallest core passing 10,000 people is 250k (0.32M parameters, a twelfth of the smallest passing dense model). It falls from 98.9 / 99.3% to 96.9 / 97.1%, a loss of 2.0 / 2.2 points.

**So the outcome depends on key precision.** A fixed core reads a store ten times larger without retraining, but 4-bit keys cost small cores more as the store grows: with 4-bit keys the 500k core retrieves the right fact first 98.2% of the time at 10,000 people and 92.4% at 100,000 (seed 0), and the 250k core 92.5% and 78.0%. At full precision both stay at 99.8% or higher. One size up, the 1m core (1.34M) with 4-bit keys holds every world: 98.6 / 98.5% at 10,000 people, 97.6 / 96.9% at 30,000 and 96.3 / 95.9% at 100,000, a loss of 2.3 / 2.6 points on the two seeds. With 4-bit keys, retrieval gets worse as the store grows, so a compressed store needs a somewhat larger core, still a twelfth of the dense model at 100,000 people.

### Cost per answered question

Calculated from model shape and store size, generating one question at a time (REPORT.md, "Cost per answered question"):

| people | dense model: compute, memory read | latent core, 4-bit keys: model + store compute | memory read, weights + store keys |
|---|---|---|---|
| 10,000 | 4m: 130 MFLOP, 56 MB | 500k: 24 + 833 MFLOP | 10 + 51 MB |
| 30,000 | 4m: 130 MFLOP, 56 MB | 1m: 47 + 2,412 MFLOP | 20 + 146 MB |
| 100,000 | 16m: 563 MFLOP, 246 MB | 1m: 47 + 8,024 MFLOP | 20 + 495 MB |

**The core's own cost stays flat, but the store lookup now dominates.** The store reader scores every token's query against every fact's key, so its compute and memory read grow with the store: at 100,000 people the latent core costs 14 times the dense model's compute and twice its memory read per question. The parameter thesis holds at full precision, and with 4-bit keys for the 1m core; the energy argument needs a store search that does not touch every key (an approximate nearest-neighbour index, for example), which is a separate project.

## What this does and doesn't show

**Shown, in this toy.** Given per-hop retrieval labels during training, a reasoner that retrieves inside its forward pass chains 3 lookups, writes out names it has never seen, and follows 1,000 edits through multi-hop reasoning at 100%, with no retraining. That matches the explicit-lookup ceiling. It also holds no world knowledge of its own. A dense model on the same data cannot compose facts it has memorized (the two-hop curse), and fine-tuning in edits leaves them unpropagated and damages everything else.

**Not shown yet:**

1. **Training without hop labels.** Without them retrieval never starts (see above). Open: are the labels needed throughout training, or only to get retrieval started?
2. **Toy difficulty.** Relations use fixed templates, names match store keys exactly, the store is always correct and complete, and no question needs more hops than there are read layers.
3. **Baselines.** The in-context oracle as first trained (`context-a`: questions only, world A, 10k steps; 2% of its tokens carry loss) gets 21.5% on 1-hop and 8–14% on unseen worlds: it never learns to copy from its prompt and emits a familiar world-A entity of the right type instead, which is also why its multi-hop cells on world A look better than 1-hop (a popularity prior on world-A entities). With bios in the mix, the 16-world pool and a 20k-step schedule (`context-multi-bios-20k`, two seeds) it reaches 97.5–99% on 1-hop and 92–98% elsewhere, so the oracle now needs twice the others' step budget. Details and controls: [`context-arm-diagnosis.md`](context-arm-diagnosis.md). The dense arm gets no hop supervision. The only edit baseline for dense is naive fine-tuning, not MEMIT or AlphaEdit.
4. **Deletion.** Not measured yet. The leakage results suggest that deleting an entry leaves nothing behind, but that needs its own test.
5. **Parameter count.** In the main comparison the latent arm has 14% more parameters than the baselines. The capacity curve above compares sizes directly.

## Next

1. `latent-multi` with hop supervision for the first 2,000 steps only (`--hop-until 2000`).
2. Harder toy: paraphrased and unseen relation phrasings, noisy name mentions, missing facts (abstain), 4-hop questions with 3 read layers.
3. A deletion test, plus a MEMIT/AlphaEdit baseline (EasyEdit, on a rented GPU).
4. Step 2: the latent read interface attached to a frozen Qwen3 / OLMo model.
