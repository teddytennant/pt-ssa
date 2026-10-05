"""Checks for parallel time-aligned spiking self-attention."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["JAX_ENABLE_X64"] = "1"

import jax
import jax.numpy as jnp
import numpy as np

from pt_ssa import attention_output, firing_indicators, log_threshold, pt_ssa, threshold

jax.config.update("jax_enable_x64", True)


def _numpy_forward(Q, K, V, a_l, alpha, D):
    """Independent NumPy reference of Algorithm 1. No softmax."""
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    scale = float(D)

    def slices(X):
        counts = scale * X
        out = []
        for d in range(D):
            out.append((counts > d).astype(np.float64) / scale)
        return out

    Qs = slices(Q)
    Ks = slices(K)
    Vs = slices(V)
    acc = np.zeros_like(Q)
    scores = []
    for d in range(D):
        A = Qs[d] @ np.swapaxes(Ks[d], -1, -2)
        scores.append(A)
        acc = acc + A @ Vs[d]
    O = float(alpha) * acc
    theta = float(np.exp(np.clip(a_l, -8.0, 8.0)))
    U = O / theta
    Z = np.round(np.clip(U, 0.0, scale)) / scale
    return Z, O, theta, U, Qs, Ks, Vs, scores


def _numpy_surrogate(Q, K, V, a_l, alpha, D, G_Z):
    """Independent surrogate: decoupled threshold, full-tensor M, unit STE."""
    _, O, theta, U, Qs, Ks, Vs, scores = _numpy_forward(Q, K, V, a_l, alpha, D)
    G_Z = np.asarray(G_Z, dtype=np.float64)
    mask = ((U >= 0.0) & (U <= D)).astype(np.float64)
    G_U = (1.0 / D) * G_Z * mask
    G_O = G_U
    M = O.size
    G_a = -1.0 / np.sqrt(M * D) * np.sum(G_U * O / theta)
    G_Q = np.zeros_like(O)
    G_K = np.zeros_like(O)
    G_V = np.zeros_like(O)
    G_R = float(alpha) * G_O
    for d in range(D):
        G_Vd = np.swapaxes(scores[d], -1, -2) @ G_R
        G_A = G_R @ np.swapaxes(Vs[d], -1, -2)
        G_Q = G_Q + G_A @ Ks[d]
        G_K = G_K + np.swapaxes(G_A, -1, -2) @ Qs[d]
        G_V = G_V + G_Vd
    return G_Q, G_K, G_V, G_a, G_O


def _away_from_half_integers(U):
    nearest_half = np.floor(U) + 0.5
    return float(np.min(np.abs(U - nearest_half))) > 1e-3


def test_forward_matches_numpy():
    rng = np.random.default_rng(0)
    Q = rng.random((5, 3))
    K = rng.random((5, 3))
    V = rng.random((5, 3))
    D = 4
    alpha = 0.8
    a_l = -0.25
    Z_ref, _, _, U, *_ = _numpy_forward(Q, K, V, a_l, alpha, D)
    assert _away_from_half_integers(U)
    Z = pt_ssa(Q, K, V, a_l, alpha, D)
    assert np.allclose(np.asarray(Z), Z_ref, rtol=1e-10, atol=1e-10)


def test_zero_log_threshold_sets_theta_to_one():
    rng = np.random.default_rng(1)
    Q = rng.random((5, 3))
    K = rng.random((5, 3))
    V = rng.random((5, 3))
    D = 4
    alpha = 1.3
    theta = np.asarray(log_threshold(0.0))
    assert theta.shape == ()
    assert theta == 1.0
    O = np.asarray(attention_output(Q, K, V, alpha, D))
    U = O / theta
    assert np.allclose(U, O)
    Z = np.asarray(pt_ssa(Q, K, V, 0.0, alpha, D))
    Z_from_U = np.round(np.clip(U, 0.0, D)) / D
    assert _away_from_half_integers(U)
    assert np.allclose(Z, Z_from_U, rtol=1e-10, atol=1e-10)


def test_integer_rate_indicators():
    D = 4
    ks = np.arange(D + 1, dtype=np.float64)
    X = (ks / D).reshape(1, D + 1)
    indicators = np.asarray(firing_indicators(X, D))
    assert indicators.shape == (D, 1, D + 1)
    for k in range(D + 1):
        expected = (np.arange(D) < k).astype(np.float64)
        assert np.array_equal(indicators[:, 0, k], expected)


def test_decoupled_threshold_grad():
    """Linear loss, U inside (0, D), a_l inside the clip range.

    Stopping 1/theta makes dL/dO identical at two theta values. Dividing by
    theta would scale the two gradients apart.
    """
    D = 4
    rng = np.random.default_rng(2)
    O = jnp.asarray(rng.uniform(0.3, 1.2, size=(3, 2)))
    W = jnp.asarray(rng.normal(size=(3, 2)))

    def loss(O_in, a_l):
        return jnp.sum(threshold(O_in, a_l, D) * W)

    g0 = jax.grad(loss)(O, jnp.array(0.0))
    g1 = jax.grad(loss)(O, jnp.array(1.0))
    assert np.all(np.asarray(O / jnp.exp(1.0)) > 0.0)
    assert np.all(np.asarray(O) < D)
    assert np.allclose(np.abs(g0), np.abs(g1), rtol=1e-10, atol=1e-10)
    assert np.allclose(g0, W / D, rtol=1e-10, atol=1e-10)
    assert not np.allclose(g0, (W / D) / jnp.exp(1.0))

    Q = jnp.asarray(rng.uniform(0.2, 0.9, size=(4, 3)))
    K = jnp.asarray(rng.uniform(0.2, 0.9, size=(4, 3)))
    V = jnp.asarray(rng.uniform(0.2, 0.9, size=(4, 3)))
    alpha = 0.5

    def rate_loss(Q_in, a_l):
        return jnp.sum(pt_ssa(Q_in, K, V, a_l, alpha, D))

    q0 = jax.grad(rate_loss)(Q, jnp.array(0.0))
    q1 = jax.grad(rate_loss)(Q, jnp.array(1.0))
    O_hat = attention_output(Q, K, V, alpha, D)
    for a_l in (0.0, 1.0):
        U = np.asarray(O_hat / log_threshold(a_l))
        assert np.all(U >= 0.0) and np.all(U <= D)
    assert np.allclose(q0, q1, rtol=1e-10, atol=1e-10)
    assert np.any(np.abs(np.asarray(q0)) > 1e-8)


def test_log_threshold_grad_matches_formula():
    """Hand-built tensors. M is the full element count of O, batch included."""
    D = 2
    alpha = 0.5
    a_l = 0.3
    Q = np.array(
        [
            [0.2, 0.7],
            [0.4, 0.1],
            [0.9, 0.5],
        ],
        dtype=np.float64,
    )
    K = np.array(
        [
            [0.3, 0.8],
            [0.6, 0.2],
            [0.1, 0.95],
        ],
        dtype=np.float64,
    )
    V = np.array(
        [
            [0.4, 0.25],
            [0.15, 0.7],
            [0.55, 0.35],
        ],
        dtype=np.float64,
    )
    W = np.array(
        [
            [1.0, -0.5],
            [0.25, 2.0],
            [-1.5, 0.75],
        ],
        dtype=np.float64,
    )
    _, O, theta, U, *_ = _numpy_forward(Q, K, V, a_l, alpha, D)
    assert np.all(U > 0.0) and np.all(U < D)
    _, _, _, G_a, _ = _numpy_surrogate(Q, K, V, a_l, alpha, D, W)
    assert O.size == 6

    def loss(a):
        return jnp.sum(pt_ssa(Q, K, V, a, alpha, D) * W)

    got = jax.grad(loss)(jnp.array(a_l))
    assert np.allclose(got, G_a, rtol=1e-10, atol=1e-10)

    Qb = np.stack([Q, Q], axis=0)
    Kb = np.stack([K, K], axis=0)
    Vb = np.stack([V, V], axis=0)
    Wb = np.stack([W, W], axis=0)
    _, Ob, _, Ub, *_ = _numpy_forward(Qb, Kb, Vb, a_l, alpha, D)
    assert Ob.shape == (2, 3, 2)
    assert Ob.size == 12
    assert np.all(Ub > 0.0) and np.all(Ub < D)
    _, _, _, G_a_batch, _ = _numpy_surrogate(Qb, Kb, Vb, a_l, alpha, D, Wb)

    def loss_batch(a):
        return jnp.sum(pt_ssa(Qb, Kb, Vb, a, alpha, D) * Wb)

    got_batch = jax.grad(loss_batch)(jnp.array(a_l))
    assert np.allclose(got_batch, G_a_batch, rtol=1e-10, atol=1e-10)
    per_example = 2.0 * G_a
    assert not np.allclose(G_a_batch, per_example, rtol=1e-6, atol=1e-8)
    assert np.allclose(G_a_batch, np.sqrt(2.0) * G_a, rtol=1e-10, atol=1e-10)


def test_unit_ste_sums_slice_grads():
    D = 2
    alpha = 1.1
    a_l = 0.0
    Q = np.array([[0.1, 0.2], [0.8, 0.3]], dtype=np.float64)
    K = np.array([[0.9, 0.7], [0.4, 0.85]], dtype=np.float64)
    V = np.array([[0.6, 0.55], [0.75, 0.35]], dtype=np.float64)
    W = np.array([[0.5, -1.0], [1.5, 0.25]], dtype=np.float64)
    _, _, _, U, Qs, Ks, _, _ = _numpy_forward(Q, K, V, a_l, alpha, D)
    assert np.all((U >= 0.0) & (U <= D))
    G_Q, G_K, G_V, _, _ = _numpy_surrogate(Q, K, V, a_l, alpha, D, W)

    def loss(Q_in):
        return jnp.sum(pt_ssa(Q_in, K, V, a_l, alpha, D) * W)

    got = np.asarray(jax.grad(loss)(jnp.asarray(Q)))
    assert np.allclose(got, G_Q, rtol=1e-10, atol=1e-10)
    assert np.any(np.abs(got) > 1e-8)
    assert not np.allclose(got, 0.0)
    assert not np.allclose(got, G_Q / D)

    # Routing through the indicator would drop slices where Q has not fired.
    # Q and K supports differ, so that gated sum is not the unit STE.
    _, _, _, _, Qs, Ks, Vs, _ = _numpy_forward(Q, K, V, a_l, alpha, D)
    mask = ((U >= 0.0) & (U <= D)).astype(np.float64)
    G_R = alpha * (W / D) * mask
    gated = np.zeros_like(G_Q)
    for d in range(D):
        G_A = G_R @ np.swapaxes(Vs[d], -1, -2)
        indicator = (D * Q > d).astype(np.float64)
        gated = gated + (G_A @ Ks[d]) * indicator
    assert not np.allclose(G_Q, gated)
    assert np.allclose(got, G_Q)

    def loss_k(K_in):
        return jnp.sum(pt_ssa(Q, K_in, V, a_l, alpha, D) * W)

    def loss_v(V_in):
        return jnp.sum(pt_ssa(Q, K, V_in, a_l, alpha, D) * W)

    assert np.allclose(jax.grad(loss_k)(jnp.asarray(K)), G_K, rtol=1e-10, atol=1e-10)
    assert np.allclose(jax.grad(loss_v)(jnp.asarray(V)), G_V, rtol=1e-10, atol=1e-10)


def test_batched_shapes():
    rng = np.random.default_rng(3)
    B, S, C, D = 2, 4, 3, 3
    Q = rng.random((B, S, C))
    K = rng.random((B, S, C))
    V = rng.random((B, S, C))
    alpha = 0.6
    a_l = -0.4
    Z = pt_ssa(Q, K, V, a_l, alpha, D)
    assert tuple(Z.shape) == (B, S, C)
    Z_ref, *_ = _numpy_forward(Q, K, V, a_l, alpha, D)
    assert np.allclose(np.asarray(Z), Z_ref, rtol=1e-10, atol=1e-10)

    def loss(Q_in, K_in, V_in, a):
        return jnp.sum(pt_ssa(Q_in, K_in, V_in, a, alpha, D))

    gQ, gK, gV, gA = jax.grad(loss, argnums=(0, 1, 2, 3))(
        jnp.asarray(Q), jnp.asarray(K), jnp.asarray(V), jnp.array(a_l)
    )
    assert tuple(gQ.shape) == (B, S, C)
    assert tuple(gK.shape) == (B, S, C)
    assert tuple(gV.shape) == (B, S, C)
    assert tuple(gA.shape) == ()
