# Why the MLP Gauge Requires an Online Rotation

## The SwiGLU MLP block

The feed-forward block computes:

$$\text{MLP}(X) = \bigl[\sigma(X W_{\text{gate}}) \odot (X W_{\text{up}})\bigr] \, W_{\text{down}}$$

Let $H = \sigma(X W_{\text{gate}}) \odot (X W_{\text{up}})$ denote the intermediate activations after the non-linearity.

## The gauge symmetry

We want to inject $R \in SO(d_{\text{ff}})$ at the $H \to W_{\text{down}}$ boundary:

$$H \, W_{\text{down}} = (H \, R)(R^\top W_{\text{down}}) = H \, R \, R^\top \, W_{\text{down}}$$

This is exact because $R R^\top = I$.

## Why we cannot fuse R upstream

To eliminate the online $H \cdot R$ multiplication, we would need to absorb $R$ into the upstream projections — i.e., find $W'_{\text{gate}}, W'_{\text{up}}$ such that:

$$\sigma(X W'_{\text{gate}}) \odot (X W'_{\text{up}}) = H \cdot R$$

Expanding the left side:

$$\sigma(X W'_{\text{gate}}) \odot (X W'_{\text{up}})$$

The element-wise SiLU $\sigma(\cdot)$ and the Hadamard product $\odot$ are both **non-linear, element-wise** operations. An orthogonal rotation $R$ mixes dimensions — it maps element $i$ into a linear combination of all elements. For the equality above to hold, we would need:

$$\sigma\bigl(\sum_j a_j\bigr) = \sum_j \sigma(a_j) \quad \text{(false in general)}$$

Since $\sigma$ does not distribute over addition, there is no $W'_{\text{gate}}$ that can absorb $R$ through the non-linearity. The rotation **cannot pass through SiLU or the Hadamard product**.

## What we can do

We split the rotation into two parts:

1. **Fuse $R^\top$ into $W_{\text{down}}$** (permanent, zero cost):
$$W_{\text{down}}^{\text{new}} = W_{\text{down}} \cdot R$$

2. **Apply $H \cdot R$ online at inference** (one matmul per layer):
$$\text{output} = (H \cdot R) \cdot (W_{\text{down}}^{\text{new}})^\top = H \cdot R \cdot R^\top \cdot W_{\text{down}}^\top = H \cdot W_{\text{down}}^\top$$

## Why this is acceptable

The online rotation $H \cdot R$ is block-diagonal ($R = \bigoplus_k R_k$, with $R_k \in SO(64)$), so the cost is $O(d_{\text{ff}} \cdot b)$ — linear in the hidden dimension and negligible compared to the $O(d_{\text{ff}} \cdot d_{\text{model}})$ cost of $W_{\text{down}}$ itself. This is the same approach used by SpinQuant and QuaRot.

## Contrast with Attention

In attention, the V→O boundary has no non-linearity between the upstream projection ($W_V$) and the downstream projection ($W_O$). The attention mechanism is a **linear combination** of V vectors:

$$\text{Attn}(Q, K, V \cdot R) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) (V \cdot R) = \left[\text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V\right] \cdot R$$

The rotation passes through linearly. Therefore we can fully fuse $R$ into both $W_V$ and $W_O$ with **zero runtime cost**.
