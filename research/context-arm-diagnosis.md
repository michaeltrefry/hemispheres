# The context arm: why it failed and how it was fixed

*Moved here from issue #1 of the original repository (github.com/zak-keown/hemispheres), which no longer exists. The numbers are unchanged; the run records behind them are in `results/step1/runs/`.*

The in-context oracle arm (`context`) is shown the facts it needs plus six distractors and should copy the right answer. As first trained (`context-a`: world-A questions only, no bios, 10k steps) it scored 21.5% on one hop. It never learned to copy from its prompt at all. With the other arms' data mix, the 16-world pool and a 20k-step schedule, it reaches 97.5–99% on one hop over two seeds.

Everything below uses the pinned weights, worlds rebuilt and checked with `records verify-data`, greedy decoding, exact match including EOS, seed 0 and the unchanged evaluator prompts.

## 1. On an unseen world, `context-a` emits world-A names that are not in its prompt

From `context-a`'s committed `world-b-final.jsonl` (900 questions, world B's facts in the prompt):

| output is… | count |
|---|---:|
| a world-**A** entity name, which never appears in the prompt | **447** |
| a world-B entity name, the only kind the prompt contains | **2** |
| a value token (years, majors, industries, currencies) | 318 |
| not any entity | 133 |

Share of answers that are world-A names, by final relation: `capital` 25/25, `born_in` 88/91, `country` 83/89, `uni_city` 35/39, `studied_at` 91/109, `hq` 43/52, `works_at` 56/88. The model generates a familiar entity of the right type from its weights, lightly steered by the prompt.

## 2. With one fact and nothing to choose between, it cannot copy an unfamiliar name

Gold-only one-hop prompts (`n_distractors=0`), entity-valued relations, held-out people, n=119:

| gold object in the prompt | copied exactly |
|---|---:|
| the true world-A object (familiar) | 88/119 (73.9%) |
| the same prompt with the object replaced by a world-B name | **0/119** |

With the unfamiliar object, 43 answers are a different world-A entity and 76 are near-miss blends:

```
Gakor Shishas is a native of Stistin.            → Stistibrol        (expected Stistin)
Shukru Branudi graduated from Krapi Institute.   → Krazon Institute  (expected Krapi Institute)
Methix Raki is employed by Togo Collective.      → Togo Systems      (expected Togo Collective)
```

"Copying" on world A was entity recognition plus regeneration from memory. Person names (10k of them, 4–7 syllable tokens) are the class it could not regenerate, which is why `mentor` scored 0/27 on one hop. The garbled names and the wrong-subject answers are one missing capability.

## 3. No subject binding, even on training people

| probe on `context-a` | result |
|---|---:|
| one-hop questions about **training** people, six distractors, n=200 | 16.5% |
| held-out one-hop (published) | 21.5% |
| at `A:`, mean probability of the gold object's first token vs a distractor object's | 0.149 vs 0.138 |
| gold object's first token ranked top at `A:` (chance ≈ 14%) | 25.5% |
| paired subject: same prompt, only the queried subject changed; both answers right, n=200 | 1 |

On the 600 edited direct questions (100 and 1,000 edits) the pre-edit value was output only 7 times, so per-person facts were not in the weights either.

## 4. Its multi-hop scores came from a popularity prior

Gold intermediate entities are Zipf-sampled, so they are popular companies, cities and universities; distractor subjects are uniform. Reconstructing every displayed fact for the committed world-A test records:

| world A, held-out | gold subject is the most popular candidate | accuracy when it is | when it is not |
|---|---:|---:|---:|
| 2-hop, n=200 | 55.5% | 62.2% | 25.8% |
| 3-hop, n=200 | 61.5% | 74.0% | 46.8% |

Causal check, same questions, distractor subjects drawn with the gold chain's popularity:

| world A, test_ood | published distractors | popularity-matched | uniform choice |
|---|---:|---:|---:|
| 2-hop | 46.0% | **32.0%** | 30% |
| 3-hop | 63.5% | **52.0%** | 36% |

The model also ignored chain structure. Choosing uniformly among final-relation candidates whose subject appears as an earlier hop's object would score 96–100%, yet its pick was such a candidate only 48–67% of the time.

## 5. Why training never built a copy circuit

`context-a` was the only arm trained without bios (`mix qa=1`): 1.9% of tokens carried loss, all answer tokens, about 3.6 per example. Every other arm trains on bios, where a name repeats in every sentence, which is the next-token pressure that forms induction and copy heads. Without it, the cheapest fit to the loss is "emit a familiar entity of the asked-for type, favouring ones mentioned nearby", which reaches loss 0.60 and never needs the subject.

Earlier repair attempts (a distractor curriculum, paired-subject questions, supervising attention to select the right fact) kept the answer-only loss and the single world, and warm-started from `context-a`. None of them made copying cheaper than the existing shortcut, and none fixed binding.

## 6. The fix: the other arms' recipe

Bios plus QA, the 16-world pool `latent-multi` uses, and twice the step budget. Nothing else changes: `small`, fp32, 64×256, lr 1e-3, warmup 500, cosine to 1e-4.

```sh
python -m hemispheres.train --arm context --steps 20000 --seed 0 --out runs/context-multi-bios-20k \
  --data data/world-a,data/pool/w3-s0,data/pool/w3-s1,data/pool/w3-s2,data/pool/w4-s0,data/pool/w4-s1,data/pool/w4-s2,data/pool/w5-s0,data/pool/w5-s1,data/pool/w5-s2,data/pool/w6-s0,data/pool/w6-s1,data/pool/w6-s2,data/pool/w7-s0,data/pool/w7-s1,data/pool/w7-s2
```

`bios=0.5,qa=0.5` is now the context arm's default mix.

| unchanged evaluator | `context-a` | 10k steps | 20k, seed 0 | 20k, seed 1 |
|---|---:|---:|---:|---:|
| world A, 1 hop, held-out | 21.5% | 91.5% | **99.0%** | **97.5%** |
| world A test_ood, 2 / 3 hops | 46.0 / 63.5% | 75.5 / 78.5% | 96.0 / 97.0% | 93.5 / 94.0% |
| 100 edits: direct / ripple / locality | 8 / 34 / 48% | 90 / 77 / 80% | 99 / 98 / 97.6% | 98 / 92 / 95% |
| 1,000 edits: direct / ripple / locality | 11 / 25 / 49% | 87 / 72 / 75% | 97.2 / 96.4 / 97.8% | 95.6 / 94.2 / 95.2% |
| unseen world B, 1 / 2 / 3 hops | 8 / 14 / 13% | 89 / 74 / 74% | 98.7 / 97.7 / 95.7% | 95.7 / 94.7 / 94.3% |
| unseen world C, 1 / 2 / 3 hops | not run | 89 / 78 / 75% | 98.3 / 98.3 / 96.7% | 95.7 / 96.3 / 93.0% |

| probe | `context-a` | 20k, seed 0 | 20k, seed 1 |
|---|---:|---:|---:|
| paired subject, both answers right, n=200 | 1 | 190 | 189 |
| six-distractor prompt with the gold object replaced by a world-B name: answered with that planted value, n=106 | 0 | 102 | 102 |
| gold-only prompt with a world-B object: copied, n=119 | 0 | 117 | 119 |
| probability of gold vs distractor first token at `A:` | 0.149 vs 0.138 | 0.989 vs 0.026 | 0.978 vs 0.027 |

Paired-subject accuracy sits on a 95% line: 95.0% and 94.5%, pooled 379/400 (95% interval roughly 92–96.5%). The misses are the same one-hop errors counted twice, not a separate failure.

## 7. The copy circuit forms in a late, seed-dependent jump

One-hop accuracy during training, n=200 per evaluation:

| step | 2k | 4k | 6k | 8k | 10k | 12k | 14k | 16k | 18k | 20k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `context-a` | 12 | 13 | 19 | 15.5 | 21.5 | | | | | |
| 10k schedule | 18 | 20 | 48.5 | 85 | 92 | | | | | |
| 20k schedule, seed 0 | 19.5 | 91.5 | 93 | 94 | 97 | 94.5 | 96.5 | 98.5 | 98.5 | 99 |
| 20k schedule, seed 1 | 17 | 15.5 | 15.5 | 92 | 93 | 94.5 | 95.5 | 98 | 97 | 97.5 |

The jump happens anywhere from step 4k to 8k depending on the seed. The longer schedule matters because it leaves enough high-learning-rate steps after the jump; the 10k cosine schedule has mostly decayed by then. Mean training loss does not show the jump, because it is dominated by bio tokens that are unpredictable across 16 worlds.

## 8. Ablation: loss on the prompt does not help

The 10k recipe plus loss on the facts block and question as well (78% of tokens trained): one-hop 16.5 / 18 / 18.5 / 17.5 / 16.5% at steps 2k–10k. Diluting the answer signal removed the jump within the budget. Its weights are kept locally but not published, and its flag is not in the code.

## Lessons for later steps

- Compare arms on the same data mix before blaming an architecture.
- When a world swap is available, check whether outputs are names that appear nowhere in the prompt. It is the fastest test of whether a model reads its context.
- Capabilities that form in a jump need step-budget sweeps and at least two seeds.
