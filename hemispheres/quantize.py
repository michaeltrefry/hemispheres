"""TurboQuant compression of store keys.

TurboQuant (Zandieh et al., "TurboQuant: Online Vector Quantization with Near-optimal Distortion
Rate", arXiv 2504.19874, ICLR 2026) compresses a vector to a few bits per number with no training
and no codebook fitted to the data:

  1. Unit-normalize the vector (its norm is kept) and rotate it by a fixed random rotation. After
     the rotation each coordinate is distributed like a coordinate of a random unit vector, close to
     a Gaussian with variance 1/d, whatever the data.
  2. Round each rotated coordinate to the nearest of 2^(bits-1) levels of a Lloyd-Max quantizer
     for that Gaussian (the levels that minimize mean squared error).
  3. Keep a 1-bit QJL sketch of what the rounding lost: the signs of S r for a fixed Gaussian
     matrix S and the residual r, plus the residual's norm. sqrt(pi/2)/d * |r| * (S y) . sign(S r)
     estimates y . r without bias for any query y.

So a key costs `bits` bits per number plus two 32-bit norms. `compress` returns, for each key, the
vector whose dot product with any query equals TurboQuant's estimate of that query's dot product
with the original key, so code that scores queries against keys works unchanged.
"""

import math

import mlx.core as mx

NORM_BYTES = 8  # the key's norm and its residual's norm, as 32-bit numbers


def _phi(x: float) -> float:
    return 0.0 if math.isinf(x) else math.exp(-x * x / 2) / math.sqrt(2 * math.pi)


def _cdf(x: float) -> float:
    return (1 + math.erf(x / math.sqrt(2))) / 2


def _cdf_inv(p: float) -> float:
    lo, hi = -10.0, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if _cdf(mid) < p else (lo, mid)
    return (lo + hi) / 2


def lloyd_max_gaussian(bits: int, tolerance: float = 1e-12, max_iterations: int = 200_000) -> list[float]:
    """The 2^bits levels minimizing mean squared error for a standard Gaussian, in increasing order,
    iterated until no level moves by more than `tolerance`."""
    n = 2 ** bits
    levels = [_cdf_inv((i + 0.5) / n) for i in range(n)]
    for _ in range(max_iterations):
        edges = [-math.inf] + [(a + b) / 2 for a, b in zip(levels, levels[1:])] + [math.inf]
        new = [(_phi(a) - _phi(b)) / (_cdf(b) - _cdf(a)) for a, b in zip(edges, edges[1:])]
        if max(abs(a - b) for a, b in zip(new, levels)) < tolerance:
            return new
        levels = new
    raise RuntimeError(f"Lloyd-Max at {bits} bits did not converge in {max_iterations} iterations")


class TurboQuant:
    """TurboQuant for `dim`-dimensional vectors at `bits` total bits per number (bits - 1 for the
    rounded coordinates, 1 for the residual's sign sketch). The rotation and sketch come from `seed`."""

    def __init__(self, dim: int, bits: int, seed: int = 0):
        if bits < 2:
            raise ValueError("TurboQuant needs at least 2 bits: 1 for the levels and 1 for the sign sketch")
        self.dim, self.bits = dim, bits
        k_rot, k_sketch = mx.random.split(mx.random.key(seed))
        q, r = mx.linalg.qr(mx.random.normal((dim, dim), key=k_rot), stream=mx.cpu)
        self.rotation = q * mx.sign(mx.diag(r))               # a uniformly random rotation
        self.sketch = mx.random.normal((dim, dim), key=k_sketch)
        levels = lloyd_max_gaussian(bits - 1)
        self.levels = mx.array(levels) / math.sqrt(dim)        # rotated unit-vector coordinates have variance 1/d
        self.edges = mx.array([(a + b) / 2 for a, b in zip(levels, levels[1:])]) / math.sqrt(dim)

    def quantize(self, x: mx.array) -> dict:
        """The stored form of vectors `x` (N, dim): level codes, residual signs and the two norms."""
        x = x.astype(mx.float32)
        norm = mx.linalg.norm(x, axis=-1, keepdims=True)
        unit = x / mx.maximum(norm, 1e-12)
        rotated = unit @ self.rotation.T
        codes = (rotated[..., None] > self.edges).sum(axis=-1).astype(mx.uint8)     # level index per coordinate
        residual = unit - self.levels[codes] @ self.rotation
        signs = mx.where(residual @ self.sketch.T >= 0, 1, -1).astype(mx.int8)
        return {"codes": codes, "signs": signs, "norm": norm,
                "residual_norm": mx.linalg.norm(residual, axis=-1, keepdims=True)}

    def dequantize(self, q: dict) -> mx.array:
        """Vectors whose dot product with any query is TurboQuant's (unbiased) estimate for the originals."""
        coarse = self.levels[q["codes"]] @ self.rotation
        scale = math.sqrt(math.pi / 2) / self.dim * q["residual_norm"]
        return q["norm"] * (coarse + scale * (q["signs"].astype(mx.float32) @ self.sketch))

    def compress(self, x: mx.array) -> mx.array:
        return self.dequantize(self.quantize(x))

    def bytes_per_vector(self) -> float:
        return self.dim * self.bits / 8 + NORM_BYTES
