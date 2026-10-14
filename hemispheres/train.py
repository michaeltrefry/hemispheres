"""Train a reasoner on a synthetic world.

  # The three arms, trained from scratch on world A
  python -m hemispheres.train --arm dense   --data data/world-a --out runs/dense-a
  python -m hemispheres.train --arm lookup  --data data/world-a --out runs/lookup-a
  python -m hemispheres.train --arm context --data data/world-a --out runs/context-a
  python -m hemispheres.train --arm latent  --data data/world-a --out runs/latent-a

  # Multi-world training: each step draws its batch (and store) from one of several worlds;
  # the first world is the one evaluated during training
  python -m hemispheres.train --arm latent --data data/world-a,data/pool/w3-s0,data/pool/w4-s0 --out runs/latent-multi

  # The context arm (in-context oracle) only learns to copy from its prompt with bios in the mix (its
  # default), several worlds and a longer schedule: use the full 16-world pool and --steps 20000
  python -m hemispheres.train --arm context --steps 20000 --data data/world-a,data/pool/w3-s0,... --out runs/context-multi-bios-20k

  # Supervise which fact each read layer fetches only for the first 2000 steps
  python -m hemispheres.train --arm latent --data data/world-a --hop-until 2000 --out runs/latent-a-hop2k

  # Updating a dense model is training: continue from its weights on new facts
  python -m hemispheres.train --arm dense --data data/world-b --init runs/dense-a --mix bios=1 \\
      --steps 2000 --out runs/dense-a-to-b
  python -m hemispheres.train --arm dense --data data/world-a --edits 100 --init runs/dense-a \\
      --mix edit_facts=1,edit_qa=1 --steps 200 --out runs/dense-a-k100

Every `--eval-every` steps the model is scored by exact match on held-out
questions (see evaluate.py). A run can be resumed with --resume.
"""

import argparse
import json
import math
import random
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from . import checkpoint, provenance
from .data import ARMS, ExampleSampler, Packer, WorldData, parse_mix
from .evaluate import evaluate, format_summary, write
from .latent import LatentConfig, LatentGPT, latent_loss
from .model import GPT, SIZES, config_for, masked_lm_loss
from .store import Store
from .synth.render import Tokenizer


def make_optimizer(args) -> optim.Optimizer:
    warmup = min(args.warmup, args.steps // 10)
    decay = optim.cosine_decay(args.lr, max(args.steps - warmup, 1), end=args.lr * args.min_lr_frac)
    schedule = optim.join_schedules([optim.linear_schedule(0.0, args.lr, warmup), decay], [warmup]) if warmup else decay
    return optim.AdamW(learning_rate=schedule, betas=(0.9, 0.95), weight_decay=args.weight_decay)


def hop_weight_at(step: int, args) -> float:
    """Weight of the retrieval-supervision loss at `step`: --hop-weight, then 0 from --hop-until on."""
    return args.hop_weight if not args.hop_until or step < args.hop_until else 0.0


def log(run: Path, record: dict) -> None:
    with open(run / "metrics.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")


def train(args) -> None:
    run = Path(args.out)
    resuming = args.resume and (run / "checkpoints" / "latest" / "state.pkl").exists()
    if run.exists() and any(run.iterdir()) and not resuming:
        raise SystemExit(f"{run} is not empty; pass --resume to continue it or choose another --out")
    run.mkdir(parents=True, exist_ok=True)

    # The first world is primary: it gets --edits and is the one evaluated during training.
    datas = [WorldData(d, args.edits if i == 0 else 0) for i, d in enumerate(args.data.split(","))]
    data = datas[0]
    tok = Tokenizer(data.vocab)
    if any(d.vocab != data.vocab for d in datas):
        raise SystemExit("all --data worlds must share one vocabulary")
    mix = parse_mix(args.mix, args.arm)
    latent = ARMS[args.arm].latent_store

    if args.init:
        model, init_config = checkpoint.load_model(args.init, args.init_checkpoint)
        if init_config["arm"] != args.arm:
            raise SystemExit(f"--init run was trained as {init_config['arm']!r}, not {args.arm!r}")
        model_cfg = model.cfg
    elif latent:
        model_cfg = LatentConfig(**SIZES[args.size], vocab_size=64 * math.ceil(len(tok) / 64),
                                 n_reads=args.n_reads, top_k=args.top_k)
        model = LatentGPT(model_cfg)
    else:
        model_cfg = config_for(args.size, vocab_size=64 * math.ceil(len(tok) / 64))
        model = GPT(model_cfg)
    model.set_dtype(getattr(mx, args.dtype))
    mx.eval(model.parameters())

    config = {**{k: v for k, v in vars(args).items() if k != "resume"}, "mix": mix,
              "model_type": "latent" if latent else "gpt", "model": asdict(model_cfg),
              "params": model.num_params(), "vocab": len(tok),
              "provenance": provenance.record(args.data.split(","))}
    if args.init:
        init_weights = Path(args.init) / "checkpoints" / args.init_checkpoint / "model.safetensors"
        config["provenance"]["init_sha256"] = provenance.sha256(init_weights)
    (run / "config.json").write_text(json.dumps(config, indent=2))  # a resume records its new settings

    optimizer = make_optimizer(args)
    samplers = [ExampleSampler(d, args.arm, mix, args.seed + i) for i, d in enumerate(datas)]
    # Stores are padded to one size so every world's store has the same shapes (no recompiles).
    store_size = max(len(d.world.facts) for d in datas)
    store_size = model_cfg.retrieval_group * math.ceil(store_size / model_cfg.retrieval_group) if latent else 0
    stores = [Store(d.world, tok, size=store_size) if latent else None for d in datas]
    packers = [Packer(smp, tok, args.batch_size, args.seq_len, store=st, n_reads=model_cfg.n_reads if latent else 0,
                      max_supervision=args.max_supervision, supervise=args.supervise)
               for smp, st in zip(samplers, stores)]
    chooser = random.Random(f"worlds:{args.seed}")
    start = 0
    if resuming:
        model.load_weights(str(run / "checkpoints" / "latest" / "model.safetensors"))
        state = checkpoint.load_optimizer_state(run, "latest", optimizer)
        start = state["step"]
        sampler_states = state["sampler"] if isinstance(state["sampler"], list) else [state["sampler"]]
        for smp, st in zip(samplers, sampler_states):
            smp.set_state(st)
        if "chooser" in state:
            chooser.setstate(state["chooser"])
        print(f"resumed {run} at step {start}")

    def train_state(step: int) -> dict:
        return {"step": step, "sampler": [smp.state() for smp in samplers], "chooser": chooser.getstate()}

    def next_batch() -> tuple:
        """(store arrays, batch) from a randomly chosen world."""
        w = chooser.randrange(len(datas))
        return (stores[w].arrays if latent else {}), packers[w].batch()

    def loss_fn(store_arrays, hop_weight, *batch):
        """(total, lm, hop, hop_hits, hop_counts): hop is the retrieval-supervision loss and
        hop_hits/hop_counts give per-read-layer top-1 retrieval accuracy (all zero without a store)."""
        if latent:
            return latent_loss(model, store_arrays, *batch, hop_weight=hop_weight)
        lm = masked_lm_loss(model, *batch)
        return lm, lm, mx.array(0.0), mx.zeros((1,)), mx.zeros((1,))

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    state = [model.state, optimizer.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step_fn(store_arrays, hop_weight, *batch):
        (loss, lm, hop, hits, counts), grads = loss_and_grad(store_arrays, hop_weight, *batch)
        grads, grad_norm = optim.clip_grad_norm(grads, args.grad_clip)
        optimizer.update(model, grads)
        return loss, lm, hop, hits, counts, grad_norm

    def run_eval(step: int) -> None:
        sets = args.eval_sets.split(",")
        summary, records = evaluate(model, tok, data, args.arm, sets, args.eval_n, args.seed)
        log(run, {"step": step, "eval": summary})
        write(run / "evals", f"{data.path.name}{f'-k{args.edits}' if args.edits else ''}-step{step}", summary, records,
              args={"data": str(data.path), "edits": args.edits, "sets": args.eval_sets, "n": args.eval_n,
                    "seed": args.seed, "step": step})
        print(format_summary(summary))

    print(f"{args.arm}: {model.num_params() / 1e6:.1f}M params, {len(tok)} tokens, mix {mix}, "
          f"{len(datas)} world(s), {args.steps} steps × {args.batch_size}×{args.seq_len}")
    store_arrays, batch = next_batch()
    t_last, trained_tokens, losses, hits_sum, counts_sum = time.perf_counter(), 0, [], 0, 0
    for step in range(start, args.steps):
        hop_weight = hop_weight_at(step, args)
        loss, lm, hop, hits, counts, grad_norm = step_fn(store_arrays, mx.array(hop_weight), *batch)
        mx.async_eval(loss, lm, hop, hits, counts, grad_norm, state)
        trained_tokens += int(batch[2].sum().item())
        store_arrays, batch = next_batch()  # build the next batch while this step runs
        losses.append((loss.item(), lm.item(), hop.item()))
        hits_sum, counts_sum = hits_sum + hits, counts_sum + counts

        if (step + 1) % args.log_every == 0 or step + 1 == args.steps:
            dt = time.perf_counter() - t_last
            n = len(losses)
            rec = {"step": step + 1, "loss": sum(x[0] for x in losses) / n, "grad_norm": grad_norm.item(),
                   "lr": float(optimizer.learning_rate), "tok_per_s": args.batch_size * args.seq_len * n / dt,
                   "trained_frac": trained_tokens / (args.batch_size * args.seq_len * n)}
            if latent:
                rec["lm_loss"] = sum(x[1] for x in losses) / n
                rec["hop_loss"] = sum(x[2] for x in losses) / n
                rec["hop_weight"] = hop_weight
                rec["retrieval_top1"] = [h / c if c else None for h, c in zip(hits_sum.tolist(), counts_sum.tolist())]
            log(run, rec)
            parts = ""
            if latent:
                ret = " ".join(f"h{i} {a:.0%}" for i, a in enumerate(rec["retrieval_top1"]) if a is not None)
                parts = f" (lm {rec['lm_loss']:.4f}, hop {rec['hop_loss']:.4f}; retrieval {ret})"
            print(f"step {rec['step']:>6} loss {rec['loss']:.4f}{parts} gnorm {rec['grad_norm']:.2f} "
                  f"lr {rec['lr']:.2e} {rec['tok_per_s']:,.0f} tok/s ({rec['trained_frac']:.0%} trained)", flush=True)
            t_last, trained_tokens, losses, hits_sum, counts_sum = time.perf_counter(), 0, [], 0, 0
        if args.save_every and (step + 1) % args.save_every == 0:
            checkpoint.save(run, "latest", model, optimizer, train_state(step + 1))
        if args.eval_every and (step + 1) % args.eval_every == 0 and step + 1 != args.steps:
            run_eval(step + 1)
            t_last = time.perf_counter()

    checkpoint.save(run, "final", model)
    checkpoint.save(run, "latest", model, optimizer, train_state(args.steps))
    if args.eval_every:
        run_eval(args.steps)
    print(f"saved {run}/checkpoints/final")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", choices=sorted(ARMS), required=True)
    p.add_argument("--data", required=True,
                   help="built world directory (synth/build.py), or a comma list for multi-world training "
                        "(the first is evaluated and gets --edits)")
    p.add_argument("--out", required=True, help="run directory")
    p.add_argument("--edits", type=int, default=0, help="train on the world with this many edits applied")
    p.add_argument("--mix", default=None, help="sources and weights, e.g. bios=0.5,qa=0.5 (default: per arm)")
    p.add_argument("--init", default=None, help="run directory to initialise weights from")
    p.add_argument("--init-checkpoint", default="final")
    p.add_argument("--size", choices=sorted(SIZES), default="small")
    p.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    p.add_argument("--steps", type=int, default=10_000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min-lr-frac", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=2000, help="0 disables evaluation")
    p.add_argument("--eval-sets", default="test_id,test_ood,test_1hop_ood")
    p.add_argument("--eval-n", type=int, default=200, help="questions per hop count per set")
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    g = p.add_argument_group("latent arm")
    g.add_argument("--n-reads", type=int, default=3, help="read layers (one per hop)")
    g.add_argument("--top-k", type=int, default=4, help="store entries retrieved per read")
    g.add_argument("--hop-weight", type=float, default=0.5, help="weight of the retrieval-supervision loss (0 = off)")
    g.add_argument("--hop-until", type=int, default=0,
                   help="turn the retrieval-supervision loss off from this step on (0 = never)")
    g.add_argument("--max-supervision", type=int, default=1024, help="supervised retrieval slots per batch")
    g.add_argument("--supervise", choices=("all", "first"), default="all",
                   help="supervise retrieval at every position writing a name token, or only before its first")
    train(p.parse_args())


if __name__ == "__main__":
    main()
