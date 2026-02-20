import torch
import math

def normal_interval_probability(
    interval_bounds,
    mu,
    sigma,
    trim_percentile=0.0
):
    """
    The total probability of a continuous normal distribution falling within the bounds.

    :param interval_bounds: list of 2 values: lower and upper bounds of interval
    :param mu: the normal distribution parameter
    :param sigma: the normal distribution parameter
    :param trim_percentile: the percentile after which the normal distribution is considered to be zero. After the cutoff, the distribution is normalized.
    :return:
    """
    bounds = torch.as_tensor(interval_bounds, dtype=torch.float32)
    if bounds.numel() != 2:
        raise ValueError("interval_bounds must contain at least 2 values")

    left, right = bounds[0], bounds[1]
    if left > right:
        left, right = right, left

    mu    = torch.as_tensor(mu,    dtype=torch.float32)
    sigma = torch.as_tensor(sigma, dtype=torch.float32)
    sigma = torch.clamp(sigma, min=1e-12)

    def norm_cdf(x):
        z = (x - mu) / (sigma * torch.sqrt(torch.tensor(2.0)))
        return 0.5 * (1.0 + torch.erf(z))

    p_raw = norm_cdf(right) - norm_cdf(left)

    if trim_percentile <= 0.0:
        return p_raw

    if trim_percentile >= 0.5:
        raise ValueError("trim_percentile must be less than 0.5")

    alpha = torch.as_tensor(trim_percentile, dtype=mu.dtype, device=mu.device)
    remaining_mass = torch.clamp(1.0 - 2.0 * alpha, min=1e-12)

    z_alpha  = torch.erfinv(2 * alpha - 1) * torch.sqrt(torch.tensor(2.0))
    z_1alpha = torch.erfinv(1 - 2 * alpha) * torch.sqrt(torch.tensor(2.0))

    trim_left  = mu + z_alpha  * sigma
    trim_right = mu + z_1alpha * sigma

    clipped_left  = torch.maximum(left,  trim_left)
    clipped_right = torch.minimum(right, trim_right)

    p_clipped = torch.where(
        clipped_left < clipped_right,
        norm_cdf(clipped_right) - norm_cdf(clipped_left),
        torch.zeros_like(p_raw)
    )

    p_norm = p_clipped / remaining_mass

    return torch.tensor([p_norm], dtype=mu.dtype, device=mu.device).unsqueeze(dim=-1).unsqueeze(dim=-1)

def normal_pdf(x, mu=0.0, sigma=1.0):
    x = torch.as_tensor(x, dtype=torch.float32)
    mu = torch.as_tensor(mu, dtype=torch.float32)
    sigma = torch.as_tensor(sigma, dtype=torch.float32)

    x = x.unsqueeze(-1)
    mu = mu.unsqueeze(0)
    sigma = sigma.unsqueeze(0)

    sigma = torch.clamp(sigma, min=1e-12)

    result = (
        1.0 / (sigma * math.sqrt(2.0 * math.pi)) *
        torch.exp(- (x - mu) ** 2 / (2.0 * sigma ** 2))
    )

    return result

def normal_bin_pmf(interval_bounds, mu, sigma):
    interval_bounds = torch.tensor(interval_bounds, dtype=torch.float32)
    z_lower = interval_bounds[:-1].unsqueeze(-1)  # (K, 1)
    z_upper = interval_bounds[1:].unsqueeze(-1)  # (K, 1)

    # (K, 1), (1, M)
    mu = mu.unsqueeze(0)
    sigma = sigma.unsqueeze(0)

    sigma = torch.clamp(sigma, min=1e-12)

    def normal_cdf(t, mu, sigma):
        return 0.5 * (1.0 + torch.erf((t - mu) / (sigma * torch.sqrt(torch.tensor(2.0)))))

    pmf = normal_cdf(z_upper,mu,sigma) - normal_cdf(z_lower,mu,sigma)
    zeros = torch.zeros((1, pmf.shape[1]), dtype=pmf.dtype)
    pmf = torch.cat([zeros,pmf], dim=0)

    return pmf

def truncated_normal_bin_pmf(interval_bounds, mu, sigma, left_bound, right_bound):
    if type(interval_bounds) is torch.Tensor:
        pass

    M = 1

    if len(mu.shape)>0:
        M = mu.shape[0]

    mu = mu.unsqueeze(0)                      # (1, M)
    sigma = torch.clamp(sigma.unsqueeze(0), min=1e-12)

    interval_bounds = torch.tensor(interval_bounds, dtype=torch.float32)
    z_lower = interval_bounds[:-1].unsqueeze(-1)  # (K, 1)
    z_upper = interval_bounds[1:].unsqueeze(-1)  # (K, 1)

    z_l = torch.maximum(z_lower, torch.tensor(left_bound)).repeat(1, M)
    z_u = torch.minimum(z_upper, torch.tensor(right_bound)).repeat(1, M)

    valid = (z_u > z_l)

    def normal_cdf(t):
        return 0.5 * (1.0 + torch.erf(t / math.sqrt(2.0)))

    a = (left_bound - mu) / sigma
    b = (right_bound - mu) / sigma
    Z = normal_cdf(b) - normal_cdf(a)
    Z = torch.clamp(Z, min=1e-12)

    left = (z_l - mu) / sigma
    right = (z_u - mu) / sigma

    pmf = (normal_cdf(right.masked_fill(~valid, 0)) - normal_cdf(left.masked_fill(~valid, 0))) / Z
    zeros = torch.zeros((1, pmf.shape[1]), dtype=pmf.dtype)
    pmf = torch.cat([zeros, pmf], dim=0)

    return pmf

def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))

def fit_truncated_normal_per_column(probs: torch.Tensor, bounds: list):
    bounds_tensor = torch.tensor(bounds, dtype=torch.float32, device="cpu")
    probs_mids = probs[0, 1:-1]

    mids = (bounds_tensor[:-1] + bounds_tensor[1:]) / 2
    deltas = bounds_tensor[1:] - bounds_tensor[:-1]
    total_p = torch.sum(probs_mids, dim=0)
    p_norm = probs_mids / total_p.unsqueeze(0)
    mu = mids @ p_norm
    var = (mids.pow(2) + (deltas.pow(2) / 12)) @ p_norm - mu.pow(2)
    sigma = torch.clamp(var.sqrt(), min=1e-6)

    return mu, sigma