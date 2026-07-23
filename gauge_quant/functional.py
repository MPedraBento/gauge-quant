import torch


def make_skew_symmetric(A: torch.Tensor) -> torch.Tensor:
    """Projects A to the skew-symmetric subspace: A -> (A - A^T)."""
    return A - A.transpose(-1, -2)


def cayley_transform(A: torch.Tensor) -> torch.Tensor:
    """
    Maps A to SO(d) via Cayley: R = (I - skew(A))(I + skew(A))^{-1}.
    Computed in float32 to preserve orthogonality.
    """
    skew = make_skew_symmetric(A).to(torch.float32)
    dim = skew.size(-1)
    I = torch.eye(dim, device=skew.device, dtype=torch.float32)
    for _ in range(skew.dim() - 2):
        I = I.unsqueeze(0)
    I = I.expand_as(skew)
    R = torch.matmul(I - skew, torch.linalg.inv(I + skew))
    return R.to(A.dtype)


def logsumexp_outlier_penalty(x: torch.Tensor, beta: float = 10.0) -> torch.Tensor:
    """Smooth L_inf approximation: (1/beta) * log(sum(exp(beta * |x_i|)))."""
    lse = torch.logsumexp(beta * x.abs(), dim=-1) / beta
    return lse.mean()


def check_orthogonality(R: torch.Tensor, eps: float = 1e-4) -> bool:
    """Returns True if R @ R^T ≈ I within tolerance eps."""
    I_approx = torch.matmul(R, R.transpose(-1, -2))
    dim = R.size(-1)
    I_true = torch.eye(dim, device=R.device, dtype=R.dtype)
    for _ in range(I_approx.dim() - 2):
        I_true = I_true.unsqueeze(0)
    I_true = I_true.expand_as(I_approx)
    return torch.max(torch.abs(I_approx - I_true)).item() < eps
