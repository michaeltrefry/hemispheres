"""Training data: sample rendered examples from a world and pack them into batches.

An arm fixes how facts are rendered (see synth/render.py); a mix fixes which
sources examples are drawn from, by probability per example:

  bios        a bio of a random entity (every entity; facts to be memorized, or
              lookup calls for the lookup arm)
  qa          a question from the world's train split
  edit_facts  a statement of one edited fact (for updating a dense model)
  edit_qa     a 1-hop question about one edited fact
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

from .store import Store
from .synth import render
from .synth.edits import Edit, apply_edits
from .synth.render import Example, Tokenizer
from .synth.schema import BOS, EOS, PAD
from .synth.world import World


@dataclass(frozen=True)
class Arm:
    bio_lookup: bool          # bios use [LOOKUP] calls instead of stating facts
    qa_style: str             # "direct", "lookup" or "context"
    default_mix: tuple        # ((source, weight), ...)
    latent_store: bool = False  # the model reads a Store inside its forward pass


ARMS = {
    "dense": Arm(bio_lookup=False, qa_style="direct", default_mix=(("bios", 0.5), ("qa", 0.5))),
    "lookup": Arm(bio_lookup=True, qa_style="lookup", default_mix=(("bios", 0.5), ("qa", 0.5))),
    "context": Arm(bio_lookup=False, qa_style="context", default_mix=(("bios", 0.5), ("qa", 0.5))),
    "latent": Arm(bio_lookup=False, qa_style="direct", default_mix=(("bios", 0.5), ("qa", 0.5)), latent_store=True),
}
SOURCES = ("bios", "qa", "edit_facts", "edit_qa")


class WorldData:
    """A built world directory (see synth/build.py), optionally with k edits applied."""

    def __init__(self, path: str | Path, edits: int = 0):
        self.path = Path(path)
        base = World.from_json(json.loads((self.path / "world.json").read_text()))
        splits = json.loads((self.path / "splits.json").read_text())
        self.splits = {s: [(subj, tuple(p.split("/"))) for subj, p in qs] for s, qs in splits.items()}
        self.base = base
        self.edit_set: dict | None = None
        self.edits: list[Edit] = []
        if edits:
            f = self.path / "edits" / f"k{edits}.json"
            if not f.exists():
                raise FileNotFoundError(f"{f} not found; build the world with --edits including {edits}")
            self.edit_set = json.loads(f.read_text())
            self.edits = [Edit(**e) for e in self.edit_set["edits"]]
        # The world as it currently is: every rendering and store lookup uses this.
        self.world = apply_edits(base, self.edits) if self.edits else base

    @property
    def vocab(self) -> list[str]:
        return json.loads((self.path / "vocab.json").read_text())


def parse_mix(spec: str | None, arm: str) -> list[tuple[str, float]]:
    if not spec:
        return list(ARMS[arm].default_mix)
    mix = []
    for part in spec.split(","):
        name, _, w = part.partition("=")
        if name not in SOURCES:
            raise ValueError(f"unknown source {name!r}; choose from {SOURCES}")
        mix.append((name, float(w or 1)))
    return mix


class ExampleSampler:
    def __init__(self, data: WorldData, arm: str, mix: list[tuple[str, float]], seed: int = 0):
        self.data, self.arm, self.rng = data, ARMS[arm], random.Random(f"sampler:{seed}")
        if any(s.startswith("edit_") for s, _ in mix) and not data.edits:
            raise ValueError("edit_* sources need a WorldData loaded with edits")
        self.sources = [s for s, _ in mix]
        self.cum = []
        total = 0.0
        for _, w in mix:
            total += w
            self.cum.append(total)
        self.entities = list(data.world.entities)

    def sample(self) -> Example:
        source = self.rng.choices(self.sources, cum_weights=self.cum)[0]
        w, rng = self.data.world, self.rng
        if source == "bios":
            return render.bio(w, rng.choice(self.entities), rng, lookup=self.arm.bio_lookup)
        if source == "qa":
            subject, path = rng.choice(self.data.splits["train"])
            return render.qa(w, subject, path, rng, self.arm.qa_style)
        e = rng.choice(self.data.edits)
        if source == "edit_qa":
            return render.qa(w, e.subject, (e.relation,), rng, self.arm.qa_style)
        ex = Example(meta={"kind": "edit_fact"})
        ex.add([BOS])
        render.statement(ex, w, e.subject, e.relation, rng, lookup=self.arm.bio_lookup)
        ex.add([EOS])
        return ex

    def state(self):
        return self.rng.getstate()

    def set_state(self, state) -> None:
        self.rng.setstate(state)


class Packer:
    """Packs whole examples into rows of seq_len + 1 tokens; never splits an example.

    With a `store`, batches also carry retrieval supervision: up to
    `max_supervision` (hop, row, position) slots, each with the store index of
    the fact read layer `hop` should retrieve there, and a validity flag.
    `supervise="first"` keeps only the position before each name starts;
    "all" keeps every position that writes a name token.
    """

    def __init__(self, sampler: ExampleSampler, tokenizer: Tokenizer, batch_size: int, seq_len: int,
                 store: Store | None = None, n_reads: int = 0, max_supervision: int = 1024,
                 supervise: str = "all"):
        if supervise not in ("all", "first"):
            raise ValueError(f"supervise must be 'all' or 'first', not {supervise!r}")
        self.sampler, self.tok = sampler, tokenizer
        self.batch_size, self.seq_len = batch_size, seq_len
        self.pad = tokenizer.token_id(PAD)
        self.store, self.n_reads, self.max_supervision = store, n_reads, max_supervision
        self.supervise = supervise
        self._pending: Example | None = None

    def _next_example(self) -> Example:
        while True:
            ex, self._pending = (self._pending or self.sampler.sample()), None
            if len(ex.tokens) <= self.seq_len + 1:
                return ex
            # Too long to ever fit: drop it (none of the current formats come close).

    def row(self) -> tuple[list[int], list[int], list[tuple[int, int, int]]]:
        ids, mask, sup = [], [], []
        while True:
            ex = self._next_example()
            if len(ids) + len(ex.tokens) > self.seq_len + 1:
                self._pending = ex
                break
            if self.store is not None:
                sup += [(hop, len(ids) + pos, self.store.index[(s, r)])
                        for pos, hop, s, r, offset in ex.supervision
                        if hop < self.n_reads and (self.supervise == "all" or offset == 0)]
            ids += self.tok.encode(ex.tokens)
            mask += ex.mask
        n_pad = self.seq_len + 1 - len(ids)
        return ids + [self.pad] * n_pad, mask + [0] * n_pad, sup

    def batch(self) -> tuple[mx.array, ...]:
        """(inputs, targets, weights[, sup_idx, sup_fact, sup_valid]).

        weights[t] = 1 where targets[t] is trained on.
        """
        rows = [self.row() for _ in range(self.batch_size)]
        ids = mx.array([r[0] for r in rows], dtype=mx.int32)
        weights = mx.array([r[1] for r in rows], dtype=mx.float32)
        out = (ids[:, :-1], ids[:, 1:], weights[:, 1:])
        if self.store is None:
            return out
        slots = [(hop, i, pos, fact) for i, r in enumerate(rows) for hop, pos, fact in r[2]]
        if len(slots) > self.max_supervision:
            slots = self.sampler.rng.sample(slots, self.max_supervision)
        n_pad = self.max_supervision - len(slots)
        slots += [(0, 0, 0, 0)] * n_pad
        sup_idx = mx.array([s[:3] for s in slots], dtype=mx.int32)
        sup_fact = mx.array([s[3] for s in slots], dtype=mx.int32)
        sup_valid = mx.array([1.0] * (len(slots) - n_pad) + [0.0] * n_pad)
        return out + (sup_idx, sup_fact, sup_valid)
