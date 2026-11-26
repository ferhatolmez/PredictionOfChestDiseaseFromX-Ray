"""Check the from-scratch neural network, above all its backpropagation."""

import numpy as np

from from_scratch.neural_net import (
    MLP,
    SGDMomentum,
    cross_entropy,
    numerical_gradient,
    softmax,
    train,
)

RNG = np.random.default_rng(0)


def test_softmax_rows_sum_to_one_and_survive_large_scores():
    z = np.array([[1.0, 2.0, 3.0], [1000.0, 1000.0, 1000.0]])
    p = softmax(z)
    assert np.allclose(p.sum(axis=1), 1)
    assert np.allclose(p[1], 1 / 3)  # no overflow


def test_cross_entropy_of_perfect_and_uniform_prediction():
    y = np.array([0, 2])
    assert cross_entropy(np.eye(3)[y], y) < 1e-9
    assert np.isclose(cross_entropy(np.full((2, 3), 1 / 3), y), np.log(3))


def test_backprop_matches_numerical_gradient():
    """The analytic gradients of every parameter equal finite differences."""
    model = MLP([5, 4, 3], seed=1)
    x = RNG.normal(size=(6, 5))
    y = RNG.integers(0, 3, size=6)
    w = RNG.uniform(0.5, 2.0, size=6)  # also checks the weighted loss

    model.loss_and_backward(x, y, w)
    analytic = [grad.copy() for _, grad in model.params_and_grads()]

    def loss():
        return cross_entropy(model.predict_proba(x), y, w)

    for (param, _), grad in zip(model.params_and_grads(), analytic, strict=True):
        assert np.allclose(grad, numerical_gradient(loss, param), atol=1e-6)


def test_training_learns_a_separable_problem():
    x = RNG.normal(size=(300, 2))
    y = (x[:, 0] + x[:, 1] > 0).astype(int)
    model = MLP([2, 16, 2], seed=0)
    losses = train(model, x, y, epochs=40, optimizer=SGDMomentum(0.05))
    accuracy = (model.predict_proba(x).argmax(axis=1) == y).mean()
    assert losses[-1] < losses[0] and accuracy > 0.95
