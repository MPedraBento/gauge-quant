# GaugeQuant: Theoretical Foundations

## 1. The Core Objective

Standard post-training quantization (PTQ) fails on LLMs because of massive activation outliers. These outliers are produced by upstream projections ($W_V$, $W_{\text{up}}$, $W_{\text{gate}}$). Min-Max quantizers clip these features, destroying the linguistic manifold. GaugeQuant solves this by learning **per-boundary gauge symmetries** to democratize variance *before* quantization.

## 2. Continuous Gauge Symmetry

> **Critical invariant: R never rotates the layer input X.**
> The gauge acts exclusively on *internal boundary representations* between paired projections:
> - **Attention**: $R \in SO(\text{head\_dim})^{\text{num\_kv\_heads}}$ lives between the V output and O input
> - **MLP**: $R \in SO(\text{block\_size})^{d_{ff}/\text{block\_size}}$ lives between the SwiGLU output $H$ and $W_{\text{down}}$
>
> The input activations X flowing into a transformer block are never touched by R.

### Attention Boundary (V → O)

For GQA architectures, each KV head has an independent $R_k \in SO(\text{head\_dim})$. Query heads sharing the same KV head share the same rotation:

$$Y^{(h)} = \text{Attn}(Q, K, V \cdot R_k) \cdot (R_k^\top W_O^{(h)}) = \text{Attn}(Q, K, V) \cdot W_O^{(h)}$$

The rotation passes through the linear attention mechanism (softmax-weighted sum preserves linearity in V). This allows **full fusion** into $W_V$ and $W_O$ with zero runtime cost:

$$W_V^{\text{new}} = R_{\text{kv}}^\top \cdot W_V, \quad b_V^{\text{new}} = b_V \cdot R_{\text{kv}}, \quad W_O^{\text{new}} = W_O \cdot R_{\text{full}}$$

where $R_{\text{full}}$ repeats each $R_k$ for all query heads in its group.

### MLP Boundary (SwiGLU → down_proj)

$$\text{MLP}(X) = \underbrace{[\sigma(X W_{\text{gate}}) \odot (X W_{\text{up}})]}_H \cdot W_{\text{down}}$$

The gauge $R$ is block-diagonal: $R = \bigoplus_k R_k$ with $R_k \in SO(b)$, $b = 64$.

Since SiLU and Hadamard product are non-linear element-wise operations, $R$ **cannot** pass through them. Therefore:
- $W_{\text{down}}^{\text{new}} = W_{\text{down}} \cdot R$ is permanently fused
- $H \cdot R$ must be applied online at inference (one block-diagonal matmul per layer, ~1.5% overhead)

### Why the Q-K boundary is not exploited

RoPE positional embeddings constrain the Q-K symmetry to $U(1)^{d_k/2}$ (block-diagonal 2D rotations that commute with $\Theta_m$). This provides only $d_k/2$ free angles per head — too few parameters to meaningfully suppress outliers. The V-O and MLP boundaries have orders of magnitude more capacity.

## 3. The Cayley Transform

We learn a skew-symmetric matrix $A = -A^\top$ and map it to $SO(d)$ via:

$$R = (I - A)(I + A)^{-1}$$

**Critical**: The inversion must be computed in float32 to prevent bf16 mantissa drift. All fusion operations (weight rotations) are also computed in float32 then cast back.

Initializing $A \approx 0$ gives $R \approx I$, allowing smooth departure from identity during training.

## 4. The Dual-Graph Optimization

### Graph A: The Brain (Cross-Entropy)

Base weights $W$ are updated via standard next-token prediction. The forward pass is strictly blind to $R$ (gauge invariance).

### Graph B: The Geometer (Gauge Loss)

We compute the LogSumExp outlier penalty on the rotated boundary activations:

$$\mathcal{L}_{\text{gauge}} = \frac{1}{\beta} \log \left( \sum_i \exp(\beta \, |z_i|) \right), \quad z = H_{\text{detached}} \cdot R$$

This is a smooth, dense approximation of $\|z\|_\infty$ that routes gradient to every dimension (not just the single max outlier).

### The Gradient Firewall (`.detach()`)

$H$ is detached before rotation. This ensures:
- $\nabla_W \mathcal{L}_{\text{gauge}} = 0$ (base weights never see gauge gradients)
- $\nabla_R \mathcal{L}_{\text{gauge}} \neq 0$ (gauge parameters receive the penalty signal)

The base weights co-adapt organically to the evolving gauge through the CE loss alone.
