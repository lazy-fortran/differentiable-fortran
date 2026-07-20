"""JAX translations of the seven Enzyme README workloads."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def lstm(x: jax.Array, n: int) -> jax.Array:
    del n

    def step(carry, item):
        cell, hidden = carry
        forget = jax.nn.sigmoid(0.7 * item + 0.2)
        ingate = jax.nn.sigmoid(-0.4 * hidden + 0.1)
        outgate = jax.nn.sigmoid(0.5 * item - 0.3)
        change = jnp.tanh(0.8 * hidden + 0.6 * item)
        cell = cell * forget + ingate * change
        hidden = outgate * jnp.tanh(cell)
        return (cell, hidden), jnp.log(2.0 + jnp.exp(hidden)) - 0.1 * hidden

    _, values = jax.lax.scan(step, (0.2, -0.1), x)
    return jnp.mean(values)


def ba(x: jax.Array, n: int) -> jax.Array:
    points = x.reshape(n, 3) - jnp.array([0.1, -0.2, -3.0])
    axis = jnp.array([0.03, -0.04, 0.02])
    theta = jnp.linalg.norm(axis)
    axis /= theta
    rotated = (
        points * jnp.cos(theta)
        + jnp.cross(jnp.broadcast_to(axis, points.shape), points) * jnp.sin(theta)
        + jnp.outer(points @ axis, axis) * (1.0 - jnp.cos(theta))
    )
    projected = rotated[:, :2] / rotated[:, 2, None]
    radius2 = jnp.sum(projected * projected, axis=1)
    distortion = 1.0 + 0.01 * radius2 - 0.001 * radius2 * radius2
    pixels = 800.0 * projected * distortion[:, None] + jnp.array([320.0, 240.0])
    residual = pixels - jnp.array([321.0, 239.0])
    return jnp.mean(jnp.sum(residual * residual, axis=1))


def gmm(x: jax.Array, n: int) -> jax.Array:
    points = x.reshape(n, 4)
    k = jnp.arange(4, dtype=x.dtype)[:, None]
    d = jnp.arange(1, 5, dtype=x.dtype)[None, :]
    means = 0.15 * (k - 1.0) * d
    precision = jnp.exp(0.02 * (k + 1.0) * d)
    centered = points[:, None, :] - means[None, :, :]
    terms = 0.1 * (k[:, 0] - 1.0) - 0.5 * jnp.sum(
        precision[None, :, :] * centered * centered, axis=2
    )
    return -jnp.mean(jax.scipy.special.logsumexp(terms, axis=1))


def euler(x: jax.Array, n: int) -> jax.Array:
    dt = 2.1 / n

    def step(state, forcing):
        state = state + dt * (-1.2 * state + 0.05 * forcing)
        return state, None

    state, _ = jax.lax.scan(step, jnp.array(1.0), x)
    return state


def rk4(x: jax.Array, n: int) -> jax.Array:
    dt = 2.1 / n

    def step(state, forcing):
        k1 = -1.2 * state + 0.05 * forcing
        k2 = -1.2 * (state + 0.5 * dt * k1) + 0.05 * forcing
        k3 = -1.2 * (state + 0.5 * dt * k2) + 0.05 * forcing
        k4 = -1.2 * (state + dt * k3) + 0.05 * forcing
        state = state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        return state, None

    state, _ = jax.lax.scan(step, jnp.array(1.0), x)
    return state


def fft(x: jax.Array, n: int) -> jax.Array:
    re, im = x[:n], x[n:]
    block = 2
    while block <= n:
        half = block // 2
        ar, br = re.reshape(-1, block)[:, :half], re.reshape(-1, block)[:, half:]
        ai, bi = im.reshape(-1, block)[:, :half], im.reshape(-1, block)[:, half:]
        angle = -2.0 * jnp.pi * jnp.arange(half, dtype=x.dtype) / block
        tr = jnp.cos(angle) * br - jnp.sin(angle) * bi
        ti = jnp.sin(angle) * br + jnp.cos(angle) * bi
        re = jnp.concatenate((ar + tr, ar - tr), axis=1).reshape(-1)
        im = jnp.concatenate((ai + ti, ai - ti), axis=1).reshape(-1)
        block *= 2
    return jnp.mean(jnp.log(1.0 + re * re + im * im))


def brusselator(x: jax.Array, n: int) -> jax.Array:
    u, v = x[: n * n].reshape(n, n), x[n * n :].reshape(n, n)
    alpha = 0.01 * (n - 1) * (n - 1)

    def laplace(field):
        padded = jnp.pad(field, 1, mode="edge")
        return (
            padded[:-2, 1:-1]
            + padded[2:, 1:-1]
            + padded[1:-1, :-2]
            + padded[1:-1, 2:]
            - 4.0 * field
        )

    u2v = u * u * v
    du = alpha * laplace(u) + 1.0 + u2v - 4.4 * u
    dv = alpha * laplace(v) + 3.4 * u - u2v
    return jnp.mean(du * du + dv * dv)


WORKLOADS = {
    1: ("LSTM", lstm),
    2: ("BA", ba),
    3: ("GMM", gmm),
    4: ("Euler", euler),
    5: ("RK4", rk4),
    6: ("FFT", fft),
    7: ("Bruss", brusselator),
}


def compiled(workload: int, n: int):
    function = partial(WORKLOADS[workload][1], n=n)
    return jax.jit(function), jax.jit(jax.grad(function))
