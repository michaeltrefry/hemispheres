import json

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten

from hemispheres.data import ExampleSampler, Packer, WorldData, parse_mix
from hemispheres.evaluate import evaluate
from hemispheres.latent import LatentConfig, LatentGPT, latent_loss
from hemispheres.store import Store
from hemispheres.synth import Tokenizer, WorldSizes, generate_world, vocabulary
from hemispheres.synth.build import make_splits


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    d = tmp_path_factory.mktemp("world")
    w = generate_world("t", 0, sizes=WorldSizes(persons=200, companies=20, universities=5, cities=12, countries=3))
    (d / "world.json").write_text(json.dumps(w.to_json()))
    (d / "vocab.json").write_text(json.dumps(vocabulary()))
    (d / "splits.json").write_text(json.dumps({s: [list(q) for q in qs] for s, qs in make_splits(w).items()}))
    return WorldData(d)


@pytest.fixture(scope="module")
def model():
    mx.random.seed(0)
    return LatentGPT(LatentConfig(n_layer=4, n_head=4, d_model=64, vocab_size=576, n_reads=3, top_k=3, key_dim=32,
                                  retrieval_chunk=64))


def test_store_entries_match_the_world(data):
    tok = Tokenizer(data.vocab)
    store = Store(data.world, tok)
    assert len(store) == len(data.world.facts)
    (s, r), j = next(iter(store.index.items()))
    val = store.arrays["val"][j].tolist()[:int(store.arrays["val_mask"][j].sum().item())]
    assert tuple(tok.decode(val)) == data.world.surface(data.world.facts[(s, r)])


def test_read_layers_are_spread_through_the_stack():
    assert LatentConfig(n_layer=8, n_reads=3).read_after == [2, 4, 6]
    assert LatentConfig(n_layer=6, n_reads=3).read_after == [2, 3, 4]


def test_retrieval_is_exact_top_k(model):
    q = mx.random.normal((2, 5, 32))
    keys = mx.random.normal((300, 32))
    idx = model.reads[0].retrieve(q, keys)
    scores = (q.reshape(-1, 32) @ keys.T)
    for row, got in zip(scores.tolist(), idx.reshape(-1, 3).tolist()):
        best = sorted(range(len(row)), key=lambda j: -row[j])[:3]
        assert set(got) == set(best)


def test_forward_and_loss_train_queries_and_keys(data, model):
    tok = Tokenizer(data.vocab)
    store = Store(data.world, tok)
    packer = Packer(ExampleSampler(data, "latent", parse_mix(None, "latent")), tok, batch_size=2, seq_len=96,
                    store=store, n_reads=3, max_supervision=32)
    batch = packer.batch()
    logits, queries, keys = model.forward(batch[0], store.arrays)
    assert logits.shape == (2, 96, 576) and len(queries) == 3 and keys.shape == (len(store), 32)

    def loss(*b):
        return latent_loss(model, store.arrays, *b, hop_weight=1.0)[0]

    _, grads = nn.value_and_grad(model, loss)(*batch)
    g = dict(tree_flatten(grads))
    for name in ("key_encoder.out.weight", "reads.0.query.weight", "reads.2.query.weight", "reads.1.k.weight"):
        assert mx.abs(g[name]).sum().item() > 0, name


def _supervised(data, mix, supervise="first"):
    tok = Tokenizer(data.vocab)
    store = Store(data.world, tok)
    packer = Packer(ExampleSampler(data, "latent", parse_mix(mix, "latent"), seed=3), tok, batch_size=4,
                    seq_len=128, store=store, n_reads=3, max_supervision=4096, supervise=supervise)
    _, targets, _, sup_idx, sup_fact, sup_valid = packer.batch()
    slots = [(h, r, p, f) for (h, r, p), f, v in zip(sup_idx.tolist(), sup_fact.tolist(), sup_valid.tolist()) if v]
    assert slots
    return tok, store, targets.tolist(), slots


def test_bio_supervision_points_at_the_object_about_to_be_written(data):
    tok, store, targets, slots = _supervised(data, "bios=1")
    for hop, row, pos, fact in slots:
        assert hop == 0
        obj = data.world.facts[store.facts[fact]]
        assert tok.vocab[targets[row][pos]] == data.world.surface(obj)[0]


def test_qa_supervision_is_one_fact_per_hop_ending_at_the_answer(data):
    tok, store, targets, slots = _supervised(data, "qa=1")
    by_pos = {}
    for hop, row, pos, fact in slots:
        by_pos.setdefault((row, pos), {})[hop] = fact
    for (row, pos), hops in by_pos.items():
        assert sorted(hops) == list(range(len(hops)))
        chain = [store.facts[hops[h]] for h in sorted(hops)]
        for (s1, r1), (s2, _) in zip(chain, chain[1:]):
            assert data.world.facts[(s1, r1)] == s2  # each hop starts where the last one ended
        answer = data.world.facts[chain[-1]]
        assert tok.vocab[targets[row][pos]] == data.world.surface(answer)[0]


def test_supervise_all_covers_every_token_of_every_name(data):
    for mix in ("bios=1", "qa=1"):
        tok, store, targets, slots = _supervised(data, mix, supervise="all")
        runs = {}
        for hop, row, pos, fact in slots:
            runs.setdefault((row, fact, hop), []).append(pos)
        finals = {}
        for (row, fact, hop), positions in runs.items():
            finals.setdefault(row, []).append((fact, hop, sorted(positions)))
        for row, entries in finals.items():
            for fact, hop, positions in entries:
                # Consecutive positions; the last hop's object is what gets written there.
                assert positions == list(range(positions[0], positions[0] + len(positions)))
                last = max(h for f, h, p in entries if p == positions)
                final_fact = next(f for f, h, p in entries if p == positions and h == last)
                name = data.world.surface(data.world.facts[store.facts[final_fact]])
                assert len(positions) == len(name)
                assert [tok.vocab[targets[row][q]] for q in positions] == list(name)


def test_untrained_latent_model_evaluates(data, model):
    summary, records = evaluate(model, Tokenizer(data.vocab), data, "latent", ["test_id"], n=3)
    assert records and "test_id" in summary


def _reference_read(layer, h, keys, store, wte, val_pos):
    """The direct implementation: project every retrieved token at every position."""
    import math
    B, T, D = h.shape
    H, dh, k, L = layer.n_head, layer.head_dim, layer.cfg.top_k, 8
    x = layer.norm(h)
    query = layer.query(x)
    idx = layer.retrieve(query, keys)
    score = (query[:, :, None, :] * keys[idx]).sum(-1) / math.sqrt(layer.cfg.key_dim)
    vals = (wte(store["val"][idx]) + val_pos.weight).reshape(B, T, k * L, D)
    valid = store["val_mask"][idx].reshape(B, T, k * L)
    bias = mx.where(valid > 0, mx.repeat(layer.score_scale * score, L, axis=-1), -1e9)
    qh = layer.q(x).reshape(B, T, H, 1, dh)
    kh = layer.k(vals).reshape(B, T, k * L, H, dh).transpose(0, 1, 3, 2, 4)
    vh = layer.v(vals).reshape(B, T, k * L, H, dh).transpose(0, 1, 3, 2, 4)
    logits = (qh @ kh.swapaxes(-1, -2)) / math.sqrt(dh) + bias[:, :, None, None, :]
    null = mx.broadcast_to(layer.null_logit.reshape(1, 1, H, 1, 1), (B, T, H, 1, 1))
    attn = mx.softmax(mx.concatenate([logits, null], axis=-1), axis=-1)
    out = attn[..., :-1] @ vh + attn[..., -1:] * layer.null_value[None, None, :, None, :]
    return h + layer.o(out.reshape(B, T, D))


def test_read_layer_matches_the_direct_implementation(data, model):
    store = Store(data.world, Tokenizer(data.vocab)).arrays
    layer = model.reads[1]
    layer.null_value = mx.random.normal(layer.null_value.shape)  # nonzero so the null path is checked
    h = mx.random.normal((2, 7, 64))
    keys = model.encode_keys(store)
    fast = layer(h, keys, store, model.wte, model.val_pos)[0]
    ref = _reference_read(layer, h, keys, store, model.wte, model.val_pos)
    assert mx.allclose(fast, ref, atol=1e-4).item()

    g_fast = nn.value_and_grad(layer, lambda hh: layer(hh, keys, store, model.wte, model.val_pos)[0].square().sum())
    g_ref = nn.value_and_grad(layer, lambda hh: _reference_read(layer, hh, keys, store, model.wte, model.val_pos).square().sum())
    (_, ga), (_, gb) = g_fast(h), g_ref(h)
    for (name, a), (_, b) in zip(tree_flatten(ga), tree_flatten(gb)):
        # Equal up to float32 summation order: compare against each gradient's scale.
        assert mx.abs(a - b).max().item() <= 1e-2 * mx.abs(b).max().item() + 1e-6, name


@pytest.mark.parametrize("n,group", [(1000, 64), (1024, 256), (70, 256), (4099, 128)])
def test_two_stage_top_k_is_exact(n, group):
    from hemispheres.latent import top_k_indices
    mx.random.seed(n)
    scores = mx.random.normal((50, n))
    got = top_k_indices(scores, 4, group)
    for row, idx in zip(scores.tolist(), got.tolist()):
        assert all(0 <= j < n for j in idx)
        assert sorted(row[j] for j in idx) == sorted(row)[-4:]


def test_loss_reports_retrieval_accuracy_per_hop(data, model):
    tok = Tokenizer(data.vocab)
    store = Store(data.world, tok)
    packer = Packer(ExampleSampler(data, "latent", parse_mix("qa=1", "latent"), seed=5), tok, batch_size=4,
                    seq_len=128, store=store, n_reads=3, max_supervision=256)
    batch = packer.batch()
    _, _, _, hits, counts = latent_loss(model, store.arrays, *batch, hop_weight=0.5)
    hops = batch[3][:, 0].tolist()
    valid = batch[5].tolist()
    assert counts.tolist() == [sum(1 for h, v in zip(hops, valid) if v and h == i) for i in range(3)]
    assert all(0 <= h <= c for h, c in zip(hits.tolist(), counts.tolist()))


def test_evaluation_reports_retrieval_per_hop(data, model):
    summary, records = evaluate(model, Tokenizer(data.vocab), data, "latent", ["test_ood"], n=4)
    for r in records:
        assert len(r["ret_top1"]) == len(r["ret_topk"]) == r["hops"]
        assert all(t1 <= tk for t1, tk in zip(r["ret_top1"], r["ret_topk"]))  # top-1 hit implies top-k hit
    assert set(summary["test_ood"]["3hop"]["retrieval"]) == {"h0", "h1", "h2"}


def test_latent_training_runs_end_to_end(data, tmp_path, monkeypatch):
    from hemispheres import train
    out = tmp_path / "run"
    monkeypatch.setattr("sys.argv", ["train", "--arm", "latent", "--data", str(data.path), "--out", str(out),
                                     "--size", "tiny", "--steps", "3", "--batch-size", "2", "--seq-len", "128",
                                     "--log-every", "1", "--eval-every", "3", "--eval-n", "2", "--save-every", "0",
                                     "--warmup", "1"])
    train.main()
    logs = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    assert any("retrieval_top1" in r for r in logs) and any("eval" in r for r in logs)
    assert (out / "checkpoints" / "final" / "model.safetensors").exists()


def test_hop_supervision_can_stop_partway(data, tmp_path, monkeypatch):
    from hemispheres import train
    out = tmp_path / "run"
    monkeypatch.setattr("sys.argv", ["train", "--arm", "latent", "--data", str(data.path), "--out", str(out),
                                     "--size", "tiny", "--steps", "4", "--batch-size", "2", "--seq-len", "128",
                                     "--log-every", "1", "--eval-every", "0", "--save-every", "0", "--warmup", "1",
                                     "--hop-until", "2"])
    train.main()
    logs = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    assert [r["hop_weight"] for r in logs] == [0.5, 0.5, 0.0, 0.0]
    assert all(r["hop_loss"] > 0 for r in logs)  # still measured after it stops being trained
    assert all(r["loss"] > r["lm_loss"] for r in logs[:2])
    assert all(r["loss"] == pytest.approx(r["lm_loss"]) for r in logs[2:])


def test_padding_a_store_changes_nothing(data, model):
    tok = Tokenizer(data.vocab)
    plain, padded = Store(data.world, tok), Store(data.world, tok, size=len(data.world.facts) + 300)
    ids = mx.array([tok.encode(["[BOS]", "Q:", "What", "is"])], dtype=mx.int32)
    a, qa_, ka = model.forward(ids, plain.arrays)
    b, qb, kb = model.forward(ids, padded.arrays)
    assert mx.allclose(a, b, atol=1e-5).item()
    assert mx.allclose(kb[:len(plain)], ka, atol=1e-6).item()
    # Even a query aimed straight at a padding entry never retrieves it.
    q = kb[len(plain) + 5][None, None, :] * 100
    assert (model.reads[0].retrieve(q, kb, padded.arrays["bias"]) < len(plain)).all().item()


def _world_dir(root, name, index):
    from pathlib import Path
    d = Path(root) / name
    d.mkdir()
    w = generate_world(name, index, sizes=WorldSizes(persons=150, companies=15, universities=4, cities=10, countries=3))
    (d / "world.json").write_text(json.dumps(w.to_json()))
    (d / "vocab.json").write_text(json.dumps(vocabulary()))
    (d / "splits.json").write_text(json.dumps({s: [list(q) for q in qs] for s, qs in make_splits(w).items()}))
    return d


@pytest.mark.parametrize("arm", ["latent", "dense"])
def test_multi_world_training_and_resume(tmp_path, monkeypatch, arm):
    from hemispheres import train
    worlds = ",".join(str(_world_dir(tmp_path, n, i)) for n, i in (("a", 0), ("p3", 3), ("p4", 4)))
    out = tmp_path / "run"
    common = ["train", "--arm", arm, "--data", worlds, "--out", str(out), "--size", "tiny", "--batch-size", "2",
              "--seq-len", "128", "--log-every", "1", "--eval-every", "0", "--save-every", "2", "--warmup", "1"]
    monkeypatch.setattr("sys.argv", common + ["--steps", "4"])
    train.main()
    monkeypatch.setattr("sys.argv", common + ["--steps", "6", "--resume"])
    train.main()
    steps = [json.loads(line)["step"] for line in (out / "metrics.jsonl").read_text().splitlines()]
    assert steps == [1, 2, 3, 4, 5, 6]
    config = json.loads((out / "config.json").read_text())
    assert config["data"] == worlds


def test_store_overrides_for_leakage_tests(data, model):
    tok = Tokenizer(data.vocab)
    other = generate_world("o", 5, sizes=WorldSizes(persons=100, companies=10, universities=3, cities=8, countries=2))
    for override in ("none", other):
        summary, records = evaluate(model, tok, data, "latent", ["test_id"], n=3, store_override=override)
        assert records and "test_id" in summary
        assert ("ret_top1" in records[0]) == (override == "none")  # diagnostics only when the facts are in the store


def test_stored_keys_give_the_same_answers_as_re_encoding(data, model):
    tok = Tokenizer(data.vocab)
    arrays = Store(data.world, tok).arrays
    ids = mx.array([tok.encode(["[BOS]", "Q:", "What", "is"])], dtype=mx.int32)
    stored = {**arrays, "keys": model.encode_keys(arrays)}
    assert mx.array_equal(model(ids, arrays), model(ids, stored))
    plain = evaluate(model, tok, data, "latent", ["test_id", "test_1hop_ood"], n=4)[1]
    for bits in (32, 4):
        _, recs = evaluate(model, tok, data, "latent", ["test_id", "test_1hop_ood"], n=4, key_bits=bits)
        assert len(recs) == len(plain) and all("ret_top1" in r for r in recs)
        if bits == 32:
            assert [r["output"] for r in recs] == [r["output"] for r in plain]
