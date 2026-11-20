import json
import random

import mlx.core as mx
import pytest

from hemispheres.data import ExampleSampler, Packer, WorldData, parse_mix
from hemispheres.evaluate import Question, parse_output, prompt_tokens
from hemispheres.generate import generate, store_reply
from hemispheres.model import GPT, config_for
from hemispheres.synth import Tokenizer, WorldSizes, edit_eval_set, enumerate_paths, generate_world, qa, sample_edits
from hemispheres.synth.build import make_splits
from hemispheres.synth.schema import END, PAD, RESULT, vocabulary


@pytest.fixture(scope="module")
def world_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("world")
    w = generate_world("t", 0, sizes=WorldSizes(persons=300, companies=30, universities=6, cities=15, countries=4))
    splits = make_splits(w)
    (d / "world.json").write_text(json.dumps(w.to_json()))
    (d / "vocab.json").write_text(json.dumps(vocabulary()))
    (d / "splits.json").write_text(json.dumps({s: [list(q) for q in qs] for s, qs in splits.items()}))
    (d / "edits").mkdir()
    edits = sample_edits(w, 20)
    (d / "edits" / "k20.json").write_text(json.dumps(edit_eval_set(w, edits, enumerate_paths(), n_locality=50)))
    return d


@pytest.mark.parametrize("arm", ["dense", "lookup", "context"])
def test_packed_weights_mark_exactly_the_trained_tokens(world_dir, arm):
    data = WorldData(world_dir)
    tok = Tokenizer(data.vocab)
    packer = Packer(ExampleSampler(data, arm, parse_mix(None, arm), seed=1), tok, batch_size=4, seq_len=128)
    inputs, targets, weights = packer.batch()
    assert inputs.shape == targets.shape == weights.shape == (4, 128)
    pad = tok.token_id(PAD)
    ids, w = targets.tolist(), weights.tolist()
    for row_ids, row_w in zip(ids, w):
        for t, m in zip(row_ids, row_w):
            if t == pad:
                assert m == 0
    if arm == "lookup":  # store-supplied tokens are never trained on
        vocab = data.vocab
        for row_ids, row_w in zip(ids, w):
            toks = [vocab[t] for t in row_ids]
            for i, t in enumerate(toks):
                if t == END:
                    assert row_w[i] == 0


def test_edit_sources_train_on_the_edited_world(world_dir):
    data = WorldData(world_dir, edits=20)
    edited = {(e.subject, e.relation): e.new for e in data.edits}
    assert all(data.world.facts[k] == v for k, v in edited.items())
    sampler = ExampleSampler(data, "dense", parse_mix("edit_qa=1", "dense"), seed=0)
    for _ in range(20):
        ex = sampler.sample()
        answer = [t for t, m in zip(ex.tokens, ex.mask) if m][:-1]
        assert any(tuple(answer) == data.world.surface(v) for v in edited.values())


def test_store_reply_answers_the_last_query(world_dir):
    data = WorldData(world_dir)
    w = data.world
    p = w.of_type("person")[0].id
    ex = qa(w, p, ("works_at", "hq"), random.Random(0), "lookup")
    first_result = ex.tokens.index(RESULT)
    reply = store_reply(ex.tokens[:first_result + 1], w)
    assert tuple(reply[:-1]) == w.surface(w.facts[(p, "works_at")]) and reply[-1] == END
    assert store_reply(["[LOOKUP]", "Nobody", "@works_at", RESULT], w) == [END]


def test_generate_inserts_store_replies(world_dir):
    """An untrained model almost never emits [RESULT]; force it via a model that always does."""
    data = WorldData(world_dir)
    tok = Tokenizer(data.vocab)
    w = data.world
    p = w.of_type("person")[0].id

    class AlwaysResult:
        def __call__(self, x):
            row = mx.where(mx.arange(576) == tok.token_id(RESULT), 1.0, 0.0)
            return mx.broadcast_to(row, (*x.shape, 576))

    prompt = ["[LOOKUP]", *w.surface(p), "@works_at"]
    out = generate(AlwaysResult(), tok, [prompt], max_new=len(w.surface(w.facts[(p, "works_at")])) + 2, store=w)[0]
    assert out[0] == RESULT and tuple(out[1:out.index(END)]) == w.surface(w.facts[(p, "works_at")])


def test_parse_output_and_prompts(world_dir):
    data = WorldData(world_dir)
    w = data.world
    p = w.of_type("person")[2].id
    path = ("works_at", "hq")
    ex = qa(w, p, path, random.Random(0), "lookup")
    completion = ex.tokens[ex.tokens.index("A:") + 1:]
    final, queries = parse_output(completion, "lookup")
    assert tuple(final) == w.surface(w.answer(p, path))
    assert queries == [(w.surface(s), r) for s, r, _ in w.chain(p, path)]
    assert parse_output(completion[:-1], "lookup")[0] is None  # no [EOS]: never finished
    q = Question("x", p, path, w.answer(p, path))
    assert prompt_tokens(data, q, "direct")[-1] == "A:"


def test_untrained_model_runs_end_to_end(world_dir):
    from hemispheres.evaluate import evaluate
    data = WorldData(world_dir)
    tok = Tokenizer(data.vocab)
    model = GPT(config_for("tiny", vocab_size=576))
    summary, records = evaluate(model, tok, data, "lookup", ["test_id"], n=4)
    assert records and "test_id" in summary


def test_ladder_sizes_double_and_build_for_both_arms():
    from hemispheres.latent import LatentConfig, LatentGPT
    from hemispheres.model import LADDER, SIZES
    nominal = {"250k": 0.25e6, "500k": 0.5e6, "1m": 1e6, "2m": 2e6, "4m": 4e6, "8m": 8e6, "16m": 16e6}
    counts = [GPT(config_for(s, vocab_size=576)).num_params() for s in LADDER]
    for s, n in zip(LADDER, counts):
        assert 0.8 <= n / nominal[s] <= 1.25, (s, n)
    for a, b in zip(counts, counts[1:]):
        assert 1.6 <= b / a <= 2.5
    latent = [LatentGPT(LatentConfig(**SIZES[s], vocab_size=576)).num_params() for s in LADDER]
    for d, n in zip(counts, latent):
        assert 1.05 <= n / d <= 1.45  # read layers and key encoder on top of the dense core
    for a, b in zip(latent, latent[1:]):
        assert 1.6 <= b / a <= 2.5
    depths = [SIZES[s]["n_layer"] for s in LADDER]
    assert depths == sorted(depths)


def test_bio_passes_sets_and_records_the_step_count(world_dir, tmp_path, monkeypatch):
    import math
    import sys

    from hemispheres import train
    out = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["train", "--arm", "dense", "--data", str(world_dir), "--out", str(out),
                                      "--size", "250k", "--bio-passes", "0.05", "--batch-size", "4",
                                      "--seq-len", "128", "--eval-every", "0", "--save-every", "0", "--log-every", "1"])
    train.main()
    config = json.loads((out / "config.json").read_text())
    est = config["bio_passes_estimate"]
    assert config["bio_passes"] == 0.05 and est["entities"] == len(WorldData(world_dir).world.entities)
    assert config["steps"] == est["steps"] == math.ceil(0.05 * est["entities"] / est["bios_per_step"])
    assert max(json.loads(line)["step"] for line in (out / "metrics.jsonl").read_text().splitlines()) == config["steps"]
