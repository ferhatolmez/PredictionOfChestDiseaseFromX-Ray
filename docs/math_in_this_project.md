# The mathematics behind this project

This note links each mathematical concept to the place in the code where it is used. Each section ends with questions to practise explaining it out loud.

---

## 1. Linear algebra

### Images are matrices, pixels are vectors
A grayscale image is a matrix $I \in \mathbb{R}^{H \times W}$. A colour image is a tensor $H \times W \times 3$: every pixel is a vector $(B, G, R)$. A whole batch for the network has the shape $(N, 224, 224, 3)$.

### Inner product and convolution — `from_scratch/sobel.py`
`correlate2d` computes, for every pixel, the **inner product** between the 3×3 kernel $K$ and the 3×3 image patch $P$ around that pixel:

$$\text{out}(y, x) = \langle K, P \rangle = \sum_{i,j} K_{ij}\, P_{ij}$$

The same operation is what every `Conv2D` layer inside MobileNetV2 computes, but with kernels that are *learned* instead of fixed.

### Norm — gradient magnitude
The Sobel gradient at a pixel is the vector $g = (G_x, G_y)$. Edge strength is its **Euclidean (L2) norm**:

$$\|g\|_2 = \sqrt{G_x^2 + G_y^2} \qquad\text{(`np.hypot(gx, gy)` in the code)}$$

and the edge direction is $\theta = \operatorname{atan2}(G_y, G_x)$.

### Matrix multiplication — the Dense layers
A `Dense` layer computes $y = \sigma(Wx + b)$ with $W \in \mathbb{R}^{m \times n}$. For the head of our model:
`GlobalAveragePooling2D` → $x \in \mathbb{R}^{1280}$ → `Dense(256)`: $W \in \mathbb{R}^{256 \times 1280}$ → `Dense(4)`: $W \in \mathbb{R}^{4 \times 256}$.
Number of parameters of `Dense(256)`: $1280 \cdot 256 + 256 = 327\,936$.

### Transpose, diagonal, trace, determinant, inverse
- `SOBEL_Y = SOBEL_X.T` — the vertical kernel is the **transpose** of the horizontal one.
- The Sobel kernel is an **outer product** of two vectors (it is *separable*), so its rank is 1 and its **determinant is 0** → it is **not invertible**:
  $$K_x = \begin{bmatrix}1\\2\\1\end{bmatrix}\begin{bmatrix}-1 & 0 & 1\end{bmatrix}$$
  Practical meaning: a 3×3 convolution can be done as two 1-D convolutions (6 instead of 9 multiplications per pixel).
- **Trace** = sum of the diagonal. Trace of $K_x$ is $-1 + 0 + 1 = 0$. The trace also equals the sum of the eigenvalues.
- The **confusion matrix** is a 4×4 matrix: its **diagonal** holds the correctly classified images, so
  $$\text{accuracy} = \frac{\operatorname{trace}(C)}{\sum_{ij} C_{ij}}$$

### Optional: eigenvectors and eigenvalues
For a square matrix $A$: $A v = \lambda v$. A rank-1 matrix $u v^\top$ has at most one non-zero eigenvalue, equal to $v^\top u$ (its trace). For $K_x$ this is $-1 + 0 + 1 = 0$, so **all** eigenvalues are 0 — consistent with $\det = \prod \lambda_i = 0$ and $\operatorname{trace} = \sum \lambda_i = 0$.

**Practise:** *Why is the Sobel kernel not invertible? What does the diagonal of the confusion matrix mean? How many parameters does `Dense(4)` have?* ($256 \cdot 4 + 4 = 1028$)

---

## 2. Probability and statistics

### Probability distribution and CDF — `from_scratch/histogram_equalization.py`
Divide the histogram by the number of pixels $N$ and you get the **probability distribution** of the gray levels:
$$p(v) = \frac{h(v)}{N}, \qquad \sum_v p(v) = 1$$
The **cumulative distribution function** is $\text{CDF}(v) = \sum_{k \le v} p(k)$. Histogram equalization maps each level through the CDF:
$$v' = \operatorname{round}\!\left(\frac{\text{CDF}(v) - \text{CDF}_{\min}}{1 - \text{CDF}_{\min}} \cdot 255\right)$$
If you treat the pixel value as a **random variable** $X$, then $\text{CDF}(X)$ is (approximately) **uniformly** distributed — that is exactly why the output histogram becomes flat.

### Mean and variance
- **Mean** $\mu = \frac{1}{N}\sum x_i$, **variance** $\sigma^2 = \frac{1}{N}\sum (x_i - \mu)^2$.
- Normalisation in the notebook divides by 255 → values in $[0, 1]$. The `Rescaling(2, offset=-1)` layer maps them to $[-1, 1]$, the range MobileNetV2 was trained on.
- `BatchNormalization` (inside MobileNetV2) standardises activations: $\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}$.
- `GlobalAveragePooling2D` takes the **mean** of every 7×7 feature map.

### Gaussian distribution
$$\mathcal{N}(x \mid \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}}\, e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$
About 68% of values lie within $\mu \pm \sigma$, 95% within $\mu \pm 2\sigma$. In images: sensor noise is modelled as Gaussian, and a Gaussian blur kernel is a sampled 2-D Gaussian. The $[1, 2, 1]$ smoothing inside Sobel is a small approximation of a Gaussian.

### Softmax output as a probability distribution
The last layer outputs $p_k = \frac{e^{z_k}}{\sum_j e^{z_j}}$, so $p_k \ge 0$ and $\sum_k p_k = 1$: a probability distribution over the 4 classes.

### Optional: Bayes' theorem — why precision ≠ recall
$$P(A \mid B) = \frac{P(B \mid A)\,P(A)}{P(B)}$$
In the first (feature-extraction only) run, NORMAL had recall 1.00 but precision 0.64: $P(\text{predicted NORMAL} \mid \text{NORMAL})$ is high, but $P(\text{NORMAL} \mid \text{predicted NORMAL})$ is lower, because many sick images are also predicted NORMAL. The prior $P(\text{class})$ (class imbalance) matters — that is why `class_weight='balanced'` is used.

### Optional: covariance and correlation
$\operatorname{cov}(X, Y) = E[(X - \mu_X)(Y - \mu_Y)]$, $\rho = \frac{\operatorname{cov}(X,Y)}{\sigma_X \sigma_Y} \in [-1, 1]$. Neighbouring pixels are strongly correlated — which is why convolutions (local operations) work so well on images.

**Practise:** *What is a CDF and why does mapping through it flatten the histogram? Why does the softmax output sum to 1? Why can recall be 1.0 while precision is 0.64?*

---

## 3. Calculus

### Derivatives — Sobel is a numerical derivative
The derivative is the rate of change: $f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$. On a pixel grid $h = 1$, and the **central difference** is
$$\frac{\partial I}{\partial x} \approx \frac{I(x+1) - I(x-1)}{2}$$
The row $[-1, 0, 1]$ of the Sobel kernel is exactly this (without the ½). $G_x$ and $G_y$ are the **partial derivatives**; together they form the **gradient** $\nabla I$, which points in the direction of the steepest brightness increase.

### Chain rule — how the network learns
The loss is a composition of functions: input → convolutions → pooling → Dense → softmax → cross-entropy. **Backpropagation** is the chain rule applied layer by layer:
$$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial W}$$
A useful result: for softmax + categorical cross-entropy, $\frac{\partial L}{\partial z_k} = p_k - y_k$ (prediction minus one-hot label).

### Gradient descent and the learning rate
Weights are updated as $W \leftarrow W - \eta \frac{\partial L}{\partial W}$ (Adam is an adaptive version). In this project: $\eta = 10^{-3}$ when training the head, $\eta = 10^{-5}$ when fine-tuning, so the pretrained weights change only a little.

### Optional: convexity
A function is convex if the line between any two points on its graph lies above the graph; then every local minimum is global. Cross-entropy is convex in the logits, but a deep network's loss is **not** convex in its weights — that is why initialisation (pretrained weights) and the learning rate matter.

**Practise:** *Derive the central difference. Explain backpropagation through `Dense → softmax → cross-entropy` with the chain rule. Why is a smaller learning rate used for fine-tuning?*

---

## 4. Notation used in the code and this note

| Symbol | Meaning |
|---|---|
| $\sum_{i}$ | sum over index $i$ |
| $\in$ | "is an element of", e.g. $x \in \mathbb{R}^n$ (vector of $n$ real numbers) |
| $\|x\|_2$ | Euclidean norm |
| $\langle a, b \rangle$ | inner product |
| $\nabla f$ | gradient (vector of partial derivatives) |
| $\frac{\partial f}{\partial x}$ | partial derivative |
| $E[X]$ | expected value (mean) of random variable $X$ |
