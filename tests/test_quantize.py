import math

import mlx.core as mx
import pytest

from hemispheres.quantize import NORM_BYTES, TurboQuant, lloyd_max_gaussian


def test_lloyd_max_levels_match_the_known_gaussian_quantizers():
    assert lloyd_max_gaussian(1) == pytest.approx([-math.sqrt(2 / math.pi), math.sqrt(2 / math.pi)], abs=1e-6)
    assert lloyd_max_gaussian(2) == pytest.approx([-1.5104, -0.4528, 0.4528, 1.5104], abs=1e-4)


@pytest.mark.parametrize("bits", [3, 4, 8])
def test_codes_fit_the_bit_budget(bits):
    tq = TurboQuant(dim=32, bits=bits, seed=1)
    q = tq.quantize(mx.random.normal((50, 32), key=mx.random.key(2)) * 3)
    assert int(q["codes"].max()) < 2 ** (bits - 1) and int(q["codes"].min()) >= 0
    assert set(q["signs"].flatten().tolist()) <= {-1, 1}
    assert q["norm"].shape == q["residual_norm"].shape == (50, 1)
    assert tq.bytes_per_vector() == 32 * bits / 8 + NORM_BYTES


def test_inner_product_estimates_are_unbiased_and_tighten_with_bits():
    x = mx.random.normal((1, 64), key=mx.random.key(3))
    y = mx.random.normal((64,), key=mx.random.key(4))
    true = float((x @ y).item())
    spread = {}
    for bits in (3, 4, 8):
        est = [float((TurboQuant(64, bits, seed=s).compress(x) @ y).item()) for s in range(300)]
        mean = sum(est) / len(est)
        sd = math.sqrt(sum((e - mean) ** 2 for e in est) / (len(est) - 1))
        assert abs(mean - true) < 4 * sd / math.sqrt(len(est)) + 1e-6  # no bias beyond sampling error
        spread[bits] = sd
    assert spread[3] > spread[4] > spread[8]


def test_two_bits_is_the_minimum():
    with pytest.raises(ValueError):
        TurboQuant(16, 1)
