"""Parallel time-aligned spiking self-attention (Algorithm 1).

Forward is linear spike attention: hard rate indicators, no softmax.
Backward is a custom_vjp surrogate.

M in the log-threshold gradient is the number of elements in O, the full
pre-threshold attention output. Batch axes are included. A per-example
reduction (M = S * C) is not used.
"""

from __future__ import annotations

import os
from functools import partial

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["JAX_ENABLE_X64"] = "1"

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

Array = jax.Array


def _as_float(x) -> Array:
    return jnp.asarray(x)


def _check_D(D) -> int:
    D = int(D)
    if D < 1:
        raise ValueError("D must be an integer >= 1")
    return D


def log_threshold(a_l) -> Array:
    """theta = exp(clip(a_l, -8, 8))."""
    return jnp.exp(jnp.clip(_as_float(a_l), -8.0, 8.0))


def firing_indicators(X_bar, D: int) -> Array:
    """Hard indicators 1[D * X_bar > d] for d = 0 .. D-1.

    Returns shape (D, *X_bar.shape). A rate of k/D is 1 for d < k and 0 otherwise.
    """
    X_bar = _as_float(X_bar)
    D = _check_D(D)
    counts = jnp.asarray(D, dtype=X_bar.dtype) * X_bar
    thresholds = jnp.arange(D, dtype=X_bar.dtype).reshape((D,) + (1,) * counts.ndim)
    return (counts[None, ...] > thresholds).astype(X_bar.dtype)


def _rate_slices(X_bar, D: int) -> Array:
    """X_tilde_d = 1[D * X_bar > d] / D, stacked on axis 0."""
    X_bar = _as_float(X_bar)
    scale = jnp.asarray(D, dtype=X_bar.dtype)
    return firing_indicators(X_bar, D) / scale


def attention_output(Q_bar, K_bar, V_bar, alpha, D: int) -> Array:
    """Pre-threshold output O = alpha * sum_d Q_tilde_d @ K_tilde_d^T @ V_tilde_d.

    No softmax. Matmuls use the last two axes. Q, K, V have shape (..., S, C).
    """
    D = _check_D(D)
    Q_bar = _as_float(Q_bar)
    K_bar = _as_float(K_bar)
    V_bar = _as_float(V_bar)
    Qs = _rate_slices(Q_bar, D)
    Ks = _rate_slices(K_bar, D)
    Vs = _rate_slices(V_bar, D)
    acc = jnp.zeros(Q_bar.shape, dtype=Q_bar.dtype)
    for d in range(D):
        scores = Qs[d] @ jnp.swapaxes(Ks[d], -1, -2)
        acc = acc + scores @ Vs[d]
    return _as_float(alpha) * acc


def _threshold_parts(O, a_l, D: int):
    """Return Z_bar, theta, U from the pre-threshold output."""
    D = _check_D(D)
    O = _as_float(O)
    theta = log_threshold(a_l).astype(O.dtype)
    U = O / theta
    upper = jnp.asarray(D, dtype=O.dtype)
    Z = jnp.round(jnp.clip(U, 0.0, upper)) / upper
    return Z, theta, U


def _threshold_cotangents(O, U, theta, D: int, G_Z):
    """Decoupled threshold surrogate.

    G_U = (1/D) * G_Z * 1[0 <= U <= D]
    G_O = G_U
    G_a = -1/sqrt(M * D) * sum(G_U * O / theta)
    M = number of elements in O, batch included.
    """
    G_Z = _as_float(G_Z)
    upper = jnp.asarray(D, dtype=U.dtype)
    inside = ((U >= 0.0) & (U <= upper)).astype(G_Z.dtype)
    G_U = G_Z * inside / jnp.asarray(D, dtype=G_Z.dtype)
    G_O = G_U
    M = jnp.size(O)
    scale = jnp.sqrt(jnp.asarray(M, dtype=G_Z.dtype) * jnp.asarray(D, dtype=G_Z.dtype))
    G_a = -jnp.sum(G_U * O / theta) / scale
    return G_O, G_a


def _slice_cotangents(Q_bar, K_bar, V_bar, alpha, D: int, G_O):
    """Unit STE: G_X_bar = sum_d G_{X_tilde_d}.

    Does not multiply by 1/D and does not route through the indicator.
    """
    D = _check_D(D)
    Q_bar = _as_float(Q_bar)
    K_bar = _as_float(K_bar)
    V_bar = _as_float(V_bar)
    G_O = _as_float(G_O)
    Qs = _rate_slices(Q_bar, D)
    Ks = _rate_slices(K_bar, D)
    Vs = _rate_slices(V_bar, D)
    G_Q = jnp.zeros(Q_bar.shape, dtype=G_O.dtype)
    G_K = jnp.zeros(K_bar.shape, dtype=G_O.dtype)
    G_V = jnp.zeros(V_bar.shape, dtype=G_O.dtype)
    G_R = _as_float(alpha).astype(G_O.dtype) * G_O
    for d in range(D):
        Qd = Qs[d]
        Kd = Ks[d]
        Vd = Vs[d]
        scores = Qd @ jnp.swapaxes(Kd, -1, -2)
        G_V = G_V + jnp.swapaxes(scores, -1, -2) @ G_R
        G_scores = G_R @ jnp.swapaxes(Vd, -1, -2)
        G_Q = G_Q + G_scores @ Kd
        G_K = G_K + jnp.swapaxes(G_scores, -1, -2) @ Qd
    return G_Q, G_K, G_V


@partial(jax.custom_vjp, nondiff_argnums=(2,))
def threshold(O, a_l, D: int):
    """Map pre-threshold attention output O to Z_bar.

    Surrogate backward stops the 1/theta path: G_O = G_U, not G_U / theta.
    """
    Z, _, _ = _threshold_parts(O, a_l, D)
    return Z


def _threshold_fwd(O, a_l, D):
    Z, theta, U = _threshold_parts(O, a_l, D)
    return Z, (O, U, theta, jnp.shape(a_l))


def _threshold_bwd(D, residuals, G_Z):
    O, U, theta, a_shape = residuals
    G_O, G_a = _threshold_cotangents(O, U, theta, D, G_Z)
    return G_O, jnp.reshape(G_a, a_shape)


threshold.defvjp(_threshold_fwd, _threshold_bwd)


@partial(jax.custom_vjp, nondiff_argnums=(4, 5))
def pt_ssa(Q_bar, K_bar, V_bar, a_l, alpha, D: int):
    """Parallel time-aligned spiking self-attention.

    Q_bar, K_bar, V_bar are firing rates in [0, 1] with shape (..., S, C).
    a_l is the scalar log-threshold. alpha is a scalar multiplier. D is the
    integer expansion length (static, not differentiated).

    Returns Z_bar with the same shape as the inputs. Gradients w.r.t. Q_bar,
    K_bar, V_bar, and a_l use the Algorithm 1 surrogate, not the true
    almost-everywhere-zero indicator derivative.
    """
    O = attention_output(Q_bar, K_bar, V_bar, alpha, D)
    Z, _, _ = _threshold_parts(O, a_l, D)
    return Z


def _pt_ssa_fwd(Q_bar, K_bar, V_bar, a_l, alpha, D):
    O = attention_output(Q_bar, K_bar, V_bar, alpha, D)
    Z, theta, U = _threshold_parts(O, a_l, D)
    return Z, (Q_bar, K_bar, V_bar, O, U, theta, jnp.shape(a_l))


def _pt_ssa_bwd(alpha, D, residuals, G_Z):
    Q_bar, K_bar, V_bar, O, U, theta, a_shape = residuals
    G_O, G_a = _threshold_cotangents(O, U, theta, D, G_Z)
    G_Q, G_K, G_V = _slice_cotangents(Q_bar, K_bar, V_bar, alpha, D, G_O)
    return G_Q, G_K, G_V, jnp.reshape(G_a, a_shape)


pt_ssa.defvjp(_pt_ssa_fwd, _pt_ssa_bwd)
