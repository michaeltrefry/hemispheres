import math

import mlx.core as mx
import pytest

from hemispheres.quantize import NORM_BYTES, TurboQuant, lloyd_max_gaussian


def test_lloyd_max_levels_match_the_known_gaussian_quantizers():
    assert lloyd_max_gaussian(1) == pytest.approx([-math.sqrt(2 / math.pi), math.sqrt(2 / math.pi)], abs=1e-6)
    assert lloyd_max_gaussian(2) == pytest.approx([-1.5104, -0.4528, 0.4528, 1.5104], abs=1e-4)


@pytest.mark.parametrize("bits", [3, 4, 8])
def test_decompressing_uses_only_the_stored_bits(bits):
    tq = TurboQuant(dim=32, bits=bits, seed=1)
    x = mx.random.normal((4000, 32), key=mx.random.key(2)) * 3
    q = tq.quantize(x)
    # bits - 1 bits per rounded coordinate (nearly every level used; the outermost are rare), 1 sign bit, two norms.
    used = set(q["codes"].flatten().tolist())
    assert used <= set(range(2 ** (bits - 1))) and len(used) >= 2 ** (bits - 1) - 2
    assert set(q["signs"].flatten().tolist()) == {-1, 1}
    assert set(q) == {"codes", "signs", "norm", "residual_norm"}
    assert q["norm"].shape == q["residual_norm"].shape == (4000, 1)
    assert mx.allclose(tq.dequantize({k: q[k] for k in ("codes", "signs", "norm", "residual_norm")}), tq.compress(x))
    assert tq.bytes_per_vector() == 32 * bits / 8 + NORM_BYTES


def _relative_errors(bits, coarse_only=False, pairs=40, seeds=60):
    """(estimate - true) / true over many quantizer seeds and correlated vector pairs (true dot products are
    positive, so rounding's shrinkage toward zero shows as a negative mean instead of cancelling out)."""
    xs = mx.random.normal((pairs, 64), key=mx.random.key(5))
    ys = xs + 0.5 * mx.random.normal((pairs, 64), key=mx.random.key(6))
    true = (xs * ys).sum(-1)
    scale = true
    errs = []
    for s in range(seeds):
        tq = TurboQuant(64, bits, seed=s)
        q = tq.quantize(xs)
        est = (q["norm"] * (tq.levels[q["codes"]] @ tq.rotation) if coarse_only else tq.dequantize(q))
        errs += (((est * ys).sum(-1) - true) / scale).tolist()
    mean = sum(errs) / len(errs)
    se = math.sqrt(sum((e - mean) ** 2 for e in errs) / (len(errs) - 1) / len(errs))
    return mean, se


@pytest.mark.parametrize("bits", [3, 4])
def test_inner_product_estimates_are_unbiased_unlike_plain_rounding(bits):
    mean, se = _relative_errors(bits)
    assert abs(mean) < 4 * se                      # TurboQuant: no bias beyond sampling error
    coarse_mean, coarse_se = _relative_errors(bits, coarse_only=True)
    assert coarse_mean < -6 * coarse_se            # rounding alone shrinks dot products: detectably biased


def test_estimates_tighten_with_bits():
    spread = {b: _relative_errors(b, pairs=20, seeds=20)[1] for b in (3, 4, 8)}
    assert spread[3] > spread[4] > spread[8]


def test_two_bits_is_the_minimum():
    with pytest.raises(ValueError):
        TurboQuant(16, 1)
