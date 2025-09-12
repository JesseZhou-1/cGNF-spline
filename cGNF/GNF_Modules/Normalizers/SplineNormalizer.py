import math
import torch
import torch.nn as nn
from torch.nn import Parameter
from nflows.transforms.splines.rational_quadratic import unconstrained_rational_quadratic_spline
from .Normalizer import Normalizer

class SplineNormalizer(Normalizer):
    def __init__(
        self,
        cond_size: int,
        num_bins: int = 8,
        bound: float = 3.0,
        hidden_dims=(128, 64, 32),          # a bit wider by default
        cat_dims=None,
        mu=None,
        sigma=None,
        activation: str = "silu",
        use_layernorm: bool = True,
        dropout_p: float = 0.0,
        eps: float = 1e-6                   # domain guard for inputs near ±bound
    ):
        super().__init__()
        self.cat_dims = cat_dims
        # always present; NormalizingFlowStep will overwrite .data with dataset stats
        self.mu = Parameter(mu if mu is not None else torch.zeros(1), requires_grad=False)
        self.sigma = Parameter(sigma if sigma is not None else torch.ones(1), requires_grad=False)

        self.num_bins = int(num_bins)
        self.bound = float(bound)
        self.eps = float(eps)

        Act = nn.SiLU if activation.lower() in ("silu", "swish") else nn.ReLU

        # Trunk MLP that processes per-dimension conditioner features h[:, d, :]
        layers = []
        in_dim = cond_size
        for hdim in hidden_dims:
            layers.append(nn.Linear(in_dim, hdim))
            if use_layernorm:
                layers.append(nn.LayerNorm(hdim))
            layers.append(Act())
            if dropout_p and dropout_p > 0:
                layers.append(nn.Dropout(dropout_p))
            in_dim = hdim
        self.trunk = nn.Sequential(*layers) if layers else nn.Identity()

        K = self.num_bins
        # Three heads: widths (K), heights (K), derivatives (K-1)
        self.width_head  = nn.Linear(in_dim, K)
        self.height_head = nn.Linear(in_dim, K)
        self.deriv_head  = nn.Linear(in_dim, K - 1)

        self._init_identity()

    def _init_identity(self):
        """
        Make initial transform close to identity:
        - widths/heights logits ~ 0 -> uniform bins
        - interior derivatives ~ 1
        """
        nn.init.zeros_(self.width_head.weight)
        nn.init.zeros_(self.width_head.bias)
        nn.init.zeros_(self.height_head.weight)
        nn.init.zeros_(self.height_head.bias)

        nn.init.zeros_(self.deriv_head.weight)
        # Bias so that softplus(bias) + min_derivative ≈ 1.0
        # i.e., set bias = log(exp(1 - min_derivative) - 1)
        NFLOWS_MIN_DERIVATIVE = 1e-3
        target = 1.0 - NFLOWS_MIN_DERIVATIVE
        bias_val = math.log(max(math.exp(target) - 1.0, 1e-6))
        nn.init.constant_(self.deriv_head.bias, bias_val)

    def _compute_params(self, h_flat: torch.Tensor):
        """
        h_flat: [B*D, cond_size] -> raw (unnormalized) widths, heights, derivatives
        """
        trunk_out = self.trunk(h_flat)
        widths_raw  = self.width_head(trunk_out)
        heights_raw = self.height_head(trunk_out)
        derivs_raw  = self.deriv_head(trunk_out)

        # quick shape checks during dev (comment out for speed later)
        K = self.num_bins
        assert widths_raw.shape[-1]  == K,     f"widths_raw last dim {widths_raw.shape[-1]} != K={K}"
        assert heights_raw.shape[-1] == K,     f"heights_raw last dim {heights_raw.shape[-1]} != K={K}"
        assert derivs_raw.shape[-1]  == K - 1, f"derivs_raw last dim {derivs_raw.shape[-1]} != K-1={K-1}"
        return widths_raw, heights_raw, derivs_raw

    def _apply_spline(self, x_flat, widths_raw, heights_raw, derivs_raw, inverse: bool):
        # keep inputs strictly inside (-bound, bound) to avoid rare domain errors
        x_flat = x_flat.clamp(min=-self.bound + self.eps, max=self.bound - self.eps)
        z_flat, ldj_flat = unconstrained_rational_quadratic_spline(
            inputs=x_flat,
            unnormalized_widths=widths_raw,
            unnormalized_heights=heights_raw,
            unnormalized_derivatives=derivs_raw,
            inverse=inverse,
            tails="linear",
            tail_bound=self.bound
        )
        return z_flat, ldj_flat

    def forward(self, x: torch.Tensor, h: torch.Tensor, context=None):
        """
        x: [B, D], h: [B, D, cond_size]
        returns: z [B, D], absdet [B, D] (per-dim absolute Jacobian factors)
        """
        B, D = x.shape
        x_flat = x.reshape(-1)                    # [B*D]
        h_flat = h.reshape(-1, h.size(-1))        # [B*D, cond_size]

        widths_raw, heights_raw, derivs_raw = self._compute_params(h_flat)
        z_flat, ldj_flat = self._apply_spline(x_flat, widths_raw, heights_raw, derivs_raw, inverse=False)

        z = z_flat.view(B, D)
        absdet = ldj_flat.view(B, D).exp()
        return z, absdet

    def inverse_transform(self, z: torch.Tensor, h: torch.Tensor, context=None):
        """
        z: [B, D], h: [B, D, cond_size]  -> x [B, D]
        """
        B, D = z.shape
        z_flat = z.reshape(-1)
        h_flat = h.reshape(-1, h.size(-1))

        widths_raw, heights_raw, derivs_raw = self._compute_params(h_flat)
        x_flat, _ = self._apply_spline(z_flat, widths_raw, heights_raw, derivs_raw, inverse=True)

        return x_flat.view(B, D)