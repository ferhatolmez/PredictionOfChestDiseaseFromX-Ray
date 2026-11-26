"""A small neural network written from scratch with NumPy.

Everything that Keras does for us in ``chestxray/`` is written out by hand
here: the forward pass, softmax cross-entropy, backpropagation with the chain
rule, and mini-batch gradient descent with momentum.

Shapes: a batch ``x`` has shape (N, D): N examples, D features. A dense layer
holds ``W`` (D_in, D_out) and ``b`` (D_out,).

Forward:   z = x W + b                      (matrix multiplication)
Backward:  dL/dW = x^T  dL/dz               (D_in, N) @ (N, D_out)
           dL/db = sum over the batch of dL/dz
           dL/dx = dL/dz  W^T               passed on to the layer before
"""

import numpy as np


class Dense:
    """Fully connected layer: ``z = x W + b``."""

    def __init__(self, n_in: int, n_out: int, rng: np.random.Generator):
        """Initialise the weights with He initialisation (good for ReLU)."""
        self.W = rng.normal(0.0, np.sqrt(2.0 / n_in), size=(n_in, n_out))
        self.b = np.zeros(n_out)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Return ``x W + b`` and remember ``x`` for the backward pass."""
        self._x = x
        return x @ self.W + self.b

    def backward(self, grad_z: np.ndarray) -> np.ndarray:
        """Store dL/dW and dL/db, and return dL/dx."""
        self.dW = self._x.T @ grad_z
        self.db = grad_z.sum(axis=0)
        return grad_z @ self.W.T

    def params_and_grads(self):
        """Return (parameter, gradient) pairs for the optimizer."""
        return [(self.W, self.dW), (self.b, self.db)]


class ReLU:
    """Rectified linear unit: ``max(0, z)``; its derivative is 1 where z > 0."""

    def __init__(self):
        """Create the layer (it has no parameters)."""
        self._positive = None

    def forward(self, z: np.ndarray) -> np.ndarray:
        """Return ``max(0, z)``."""
        self._positive = z > 0
        return z * self._positive

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Let the gradient through only where the input was positive."""
        return grad * self._positive

    def params_and_grads(self):
        """Return no parameters."""
        return []


def softmax(z: np.ndarray) -> np.ndarray:
    """Turn scores into probabilities that sum to 1 in each row.

    Subtracting the row maximum does not change the result (it cancels in
    the fraction) but stops ``exp`` from overflowing.
    """
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy(
    probs: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None
) -> float:
    """Weighted mean of ``-log p(true class)``.

    Args:
        probs: predicted probabilities (N, C).
        y: true class indices (N,).
        weights: one weight per example (N,), e.g. class weights; None = equal.
    """
    weights = np.ones(len(y)) if weights is None else weights
    log_p = np.log(probs[np.arange(len(y)), y] + 1e-12)
    return float(-(weights * log_p).sum() / weights.sum())


class MLP:
    """Multi-layer perceptron: Dense -> ReLU -> ... -> Dense -> softmax."""

    def __init__(self, sizes: list[int], seed: int = 0):
        """Build layers for ``sizes``, e.g. [1024, 128, 64, 4]."""
        rng = np.random.default_rng(seed)
        self.layers = []
        for i, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:], strict=True)):
            self.layers.append(Dense(n_in, n_out, rng))
            if i < len(sizes) - 2:  # no ReLU after the last layer
                self.layers.append(ReLU())

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Return the class scores (logits) for a batch."""
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Return class probabilities (N, C)."""
        return softmax(self.forward(x))

    def loss_and_backward(
        self, x: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None
    ) -> float:
        """Run forward and backward pass; every layer stores its gradients.

        For softmax followed by cross-entropy, the chain rule simplifies to
        dL/dz = (p - onehot(y)) / N for each example; with weights,
        dL/dz_i = w_i (p_i - onehot(y_i)) / sum(w).
        """
        weights = np.ones(len(y)) if weights is None else weights
        probs = softmax(self.forward(x))
        loss = cross_entropy(probs, y, weights)

        grad = probs.copy()
        grad[np.arange(len(y)), y] -= 1.0
        grad *= (weights / weights.sum())[:, None]
        for layer in reversed(self.layers):  # chain rule, last layer first
            grad = layer.backward(grad)
        return loss

    def params_and_grads(self):
        """Return all (parameter, gradient) pairs."""
        return [pg for layer in self.layers for pg in layer.params_and_grads()]


class SGDMomentum:
    """Gradient descent with momentum: ``v = mu v - lr g;  p = p + v``.

    Momentum keeps a running direction, which smooths noisy mini-batch
    gradients and speeds up progress along consistent directions.
    """

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9):
        """Store the hyperparameters."""
        self.learning_rate = learning_rate
        self.momentum = momentum
        self._velocity: dict[int, np.ndarray] = {}

    def step(self, params_and_grads) -> None:
        """Update every parameter in place."""
        for param, grad in params_and_grads:
            v = self._velocity.get(id(param), np.zeros_like(param))
            v = self.momentum * v - self.learning_rate * grad
            self._velocity[id(param)] = v
            param += v


def numerical_gradient(f, param: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Estimate df/dparam with central differences, one entry at a time.

    ``(f(p + eps) - f(p - eps)) / (2 eps)`` -- used to check backpropagation.
    """
    grad = np.zeros_like(param)
    for index in np.ndindex(param.shape):
        old = param[index]
        param[index] = old + eps
        plus = f()
        param[index] = old - eps
        minus = f()
        param[index] = old
        grad[index] = (plus - minus) / (2 * eps)
    return grad


def train(
    model: MLP,
    x: np.ndarray,
    y: np.ndarray,
    epochs: int = 30,
    batch_size: int = 64,
    optimizer: SGDMomentum | None = None,
    weights: np.ndarray | None = None,
    seed: int = 0,
) -> list[float]:
    """Train with shuffled mini-batches; return the mean loss of each epoch."""
    optimizer = optimizer or SGDMomentum()
    weights = np.ones(len(y)) if weights is None else weights
    rng = np.random.default_rng(seed)
    losses = []
    for _ in range(epochs):
        order = rng.permutation(len(y))
        batch_losses = []
        for start in range(0, len(y), batch_size):
            idx = order[start : start + batch_size]
            batch_losses.append(model.loss_and_backward(x[idx], y[idx], weights[idx]))
            optimizer.step(model.params_and_grads())
        losses.append(float(np.mean(batch_losses)))
    return losses
