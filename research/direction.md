# Direction: where Hemispheres goes next

*Written 2026-11-20 from design conversations between Michael Trefry and Claude, October to November 2026. It records Michael's direction for the project, how it relates to Zak Keown's original plan, and the reasoning and evidence behind each step. Confidence levels are Claude's assessments at the time of writing.*

## 1. The goal

**Higher intelligence at lower energy consumption, and so lower cost.** Hosting large language models is expensive, training them costs orders of magnitude more, and both use energy at a rate that is not sustainable. The project measures success as capability per unit of energy, for training and for running a model, not capability alone.

## 2. Two visions, one project

**Zak's vision** (see `BRIEF.md`) separates *facts* from the model that reasons over them. A small reasoner holds almost no facts and reads a separate store of facts inside its forward pass. Facts are added, changed or deleted by editing the store, with no retraining. His brief names four properties that nobody has shown together: a fact-free core at 1B+ parameters that reasons as well as a normal model, per-fact writes and deletes with no training step, updated facts carrying through multi-step reasoning, and verifiable deletion.

**Michael's vision** extends the separation from facts to skills:

- A base model that understands prompts and writes output.
- A fact store it reads, as in Zak's design.
- Skills loaded as additive modules, each with more specific modules beneath it:
  - language understanding and writing, with language-specific modules;
  - coding, with modules per programming language;
  - math, with modules such as basic arithmetic, algebra, geometry, trigonometry and calculus;
  - creative writing, with style modules.
- Memory used equals the core plus whatever modules are loaded, and no more.

The motivating problem: every current model merges all of its expertise into one set of weights, where the pieces cannot be separated. The aim is to make them separable.

**Longer term**, Michael sees a full system as many cooperating models, loosely inspired by the brain's division of labor: perception models for vision and hearing, an orchestrating model doing most of the thinking (a "frontal lobe"), models that distill input into memories and decide what is worth keeping (a "hippocampus"), the knowledge store as the long-term fact memory, and a personality layer that shapes the whole system's voice. Many of these can be subagents spawned by the orchestrator, which process large inputs and hand back short summaries.

The two visions fit together. Zak's step 1 is the separable-knowledge half and has toy-scale evidence. Separable skills are the other half and have no clean demonstration anywhere.

## 3. What step 1 established

From `results-step1.md` and `context-arm-diagnosis.md`, all on synthetic worlds with 25 to 29M-parameter models:

- The latent store reader answers 1 to 3-hop questions at 100% on worlds it never saw, after 1,000 edits, and drops to near zero with its store emptied. It holds almost no facts itself.
- It needs a training signal telling each read layer which fact to fetch; without it, retrieval never starts.
- The in-context baseline needed bios in its training mix, many worlds and a 20k-step schedule. Its copy ability forms in a late, seed-dependent jump.
- Limits: fixed templates, names that match store keys exactly, a complete and correct store, and no question needing more hops than the model has read layers.

Claude's assessment: the mechanism is real at toy scale (high confidence). Whether it survives messy real text is untested.

## 4. Why today's models are inseparable

- **Shared weights carry everything at once.** Training gives a model no reason to keep one domain apart from another, so it packs them into the same neurons. Nothing marks a boundary to cut along.
- **Knowledge can't be cut out afterward.** Unlearning research shows removal hides knowledge rather than deleting it: a few unrelated fine-tuning examples bring it back. Separability has to be built in during training.
- **Mixture-of-experts models are physically separable but not by domain.** Each expert is its own block of weights, but routing picks experts per token and per layer, experts track token patterns rather than subjects (the Mixtral paper found no clear topic specialization), training adds a load-balancing penalty that works against domain experts, shared layers run for every token, and experts are trained to depend on each other. Removing one is not removing "the Rust expert".
- **The nearest design to Michael's** is domain-routed mixtures such as DEMix (2021) and Branch-Train-Merge (2022): routing by domain chosen per request, no load balancing, and domain blocks that can be added or dropped. LoRA libraries with routing (LoraHub, 2023) and adapter stacking (MAD-X, 2020, which layers language adapters with task adapters) are the nearest work on additive skills.

## 5. Knowledge versus skill

The two kinds of expertise need different mechanisms:

| kind | example | where it lives | mechanism |
|---|---|---|---|
| Knowledge | facts about 18th-century literature | a store | Zak's fact store, read inside the forward pass |
| Skill | writing idiomatic Rust | weights | adapters such as LoRA, stacked general to specific |

**The memory catch.** Adapters are small, tens of megabytes, and make a model more specialized, not smaller. Memory is set by the base model. Moving facts out is what can shrink the base: research on how much models memorize (the "Physics of Language Models" series) estimates about 2 bits of knowledge per parameter, so much of a large model's size goes to stored facts.

**The floor.** Reasoning quality still scales with model size. A small core reasons worse than a large one however many modules it loads. The realistic goal is a core as small as the required reasoning allows, with everything domain-specific loaded on demand.

## 6. The energy argument

**Where the design saves energy**
- **Facts out of the weights.** RETRO (DeepMind, 2021) matched models about 25 times its size on some language tasks by retrieving from a database. Atlas (Meta, 2022), at 11B parameters, beat the 540B-parameter PaLM on few-shot question answering. Both are knowledge-heavy tasks, not reasoning.
- **Data movement dominates chip energy.** Reading a number from memory costs far more energy than computing with it, and every generated token reads the model's weights. A smaller core saves energy on every token; a store lookup touches a few entries.
- **No retraining to update knowledge.** A store write replaces fine-tuning.

**Where it doesn't save, or costs more**
- Activating only part of a model is already standard: DeepSeek-V3 has 671B parameters and uses about 37B per token. This design's edge over that has to be memory, since mixture-of-experts models still keep every expert loaded.
- Many cooperating models add overhead in orchestration and the text passed between them. Today's agent systems often use more compute than one model.
- Nothing shows that modularity makes reasoning itself cheaper.
- The current store reader scores every token against every fact, so its cost grows in step with the store. Faster search is needed before the energy argument holds at scale.

**The brain comparison.** The brain runs on about 20 watts, mostly thanks to its hardware: memory and computation sit together and neurons fire sparsely. A brain-like layout on today's chips won't produce brain-like energy use by itself.

## 7. The multi-model system: assessment

The idea is coherent and has serious precedents: Minsky's *Society of Mind* (1986); LeCun's 2022 modular proposal for autonomous machine intelligence (perception, world model, short-term memory, planner, configurator); *Generative Agents* (2023), whose memory stream scores memories for importance; MemGPT (2023), a model that manages its own memory; and today's orchestrator-plus-subagent agent systems. The hippocampus-and-neocortex pairing is already the analogy in `BRIEF.md`.

Cautions:
- **The brain is inspiration, not blueprint.** Regions are densely interconnected and most functions are distributed; memory is formed by the hippocampus and stored across the cortex.
- **How parts communicate is the central decision.** Text between parts is inspectable and swappable but loses information; captioning pipelines lost to models that see images directly. Learned vectors carry more but tie the parts to each other through joint training, which works against swapping them. "Trained to work together" and "separable" pull in opposite directions.
- **A modular system probably won't win on raw capability.** End-to-end training keeps beating hand-built pipelines. It can win on cost, updatability, inspectability and learning from experience.
- **Learning from experience is the most valuable missing piece.** Writing facts to a store is cheap; deciding what to keep, resolving conflicts with old facts and rejecting manipulated input are hard. Turning experience into new skills would mean consolidating it into a module, as sleep consolidates memories.
- **A swappable personality or values layer makes guardrails swappable too.** That has to be a deliberate choice.

## 8. Hard open problems

1. **Composing independently trained skill modules.** Real requests mix skills: Rust code that computes a trajectory needs coding, Rust and trigonometry at once. Stacking adapters works partially today and degrades as more stack.
2. **Keeping the core skill-light.** Facts can be kept out of the core's training signal; skills can't, because the core must see code to learn anything. Some training pressure has to push domain behavior into modules. How is unknown.
3. **Transfer.** Shared training is partly why models are good: code improves reasoning, and many languages improve each one. Full separation may lose some of that.
4. **Choosing which modules to load.** Explicit choice is easy; automatic routing is hard. Start explicit.
5. **Fast search over a growing store.**

**The test of separability** for any module is a double dissociation: unplug it and only its skill collapses; plug it back and the skill returns; plug in a separately trained module and it works with no retraining of the core. Step 1's world swaps and emptied-store tests already run this test for facts.

## 9. Roadmap

1. **Capacity curve (current, feature HEM-F1).** On synthetic worlds of 10,000, 30,000 and 100,000 people, find how many parameters a model needs to hold the facts when they live in its weights, versus when a fixed-size core reads them from a store. This is the energy argument in one picture. The first step needs no training: evaluate the existing `latent-multi` checkpoint, trained on 10,000-person worlds, against larger worlds' stores.
2. **Step-1 de-risking.** Test whether retrieval needs its training signal only early (`--hop-until 2000`), then a harder toy: varied phrasings, noisy name mentions, missing facts to abstain on, and 4-hop questions with 3 read layers.
3. **Synthetic-skills toy.** Made-up languages, arithmetic skills and formatting styles as modules on a small core, tested for the four properties: adding a module gains its skill, removing it loses only that skill, two modules compose, and memory scales with what is loaded.
4. **Faster store search**, so lookup cost grows slower than the store.
5. **Real models.** The store reader on a frozen open model such as Qwen3-1.7B or OLMo-2-1B, evaluated against retrieval-augmented generation on editing and multi-hop benchmarks; then real skill modules on the same base.
6. **A skill-light, knowledge-light core trained from scratch** at 125M to 1B parameters, only if the earlier steps work. This is where the memory and energy claims are proven or disproven.
