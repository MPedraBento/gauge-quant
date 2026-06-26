import torch


def make_skew_symmetric(A: torch.Tensor) -> torch.Tensor:
    """
    Forces any square matrix (or batch of matrices) to be strictly skew-symmetric.
    """
    return A - A.transpose(-1, -2)


def cayley_transform(A: torch.Tensor) -> torch.Tensor:
    """
    Maps a skew-symmetric matrix A to an orthogonal matrix R via the Cayley transform.
    Mathematically: R = (I - A)(I + A)^(-1)

    Includes crucial FP32 upcasting to prevent bfloat16 matrix inversion collapse
    and singular matrix errors during the training loop.
    """
    # 1. Force skew-symmetry and upcast to float32 for numerical stability
    skew = make_skew_symmetric(A).to(torch.float32)

    # 2. Generate the Identity matrix matching the block dimensions
    dim = skew.size(-1)
    I = torch.eye(dim, device=skew.device, dtype=torch.float32)

    # Expand Identity to match batch dimensions if necessary (e.g., block-diagonal)
    for _ in range(skew.dim() - 2):
        I = I.unsqueeze(0)
    I = I.expand_as(skew)

    # 3. Compute the safe FP32 inversion
    # R = (I - skew) @ (I + skew)^-1
    R = torch.matmul(I - skew, torch.linalg.inv(I + skew))

    # 4. Cast back to the original precision (e.g., bfloat16) for the forward pass
    return R.to(A.dtype)


def logsumexp_outlier_penalty(x: torch.Tensor, beta: float = 10.0) -> torch.Tensor:
    """
    A smooth, fully differentiable approximation of the L_infinity norm.
    Unlike `amax`, which produces a sparse subgradient targeting only the single
    largest outlier, this routes proportional gradient signals to every element
    in the tensor, drastically accelerating gauge convergence.
    """
    # We penalize the absolute magnitude of the activations/weights
    abs_x = x.abs()

    # Compute LogSumExp along the hidden dimension
    # Dividing by beta scales the penalty back to the true L_inf magnitude
    lse = torch.logsumexp(beta * abs_x, dim=-1) / beta

    return lse.mean()


def check_orthogonality(R: torch.Tensor, eps: float = 1e-4) -> bool:
    """
    Mathematical sanity check to ensure R * R^T = I.
    Useful for defensive assertions in testing scripts.
    """
    I_approx = torch.matmul(R, R.transpose(-1, -2))

    dim = R.size(-1)
    I_true = torch.eye(dim, device=R.device, dtype=R.dtype)

    for _ in range(I_approx.dim() - 2):
        I_true = I_true.unsqueeze(0)
    I_true = I_true.expand_as(I_approx)

    # Check if the maximum deviation from the Identity matrix is within tolerance
    max_error = torch.max(torch.abs(I_approx - I_true))
    return max_error.item() < eps