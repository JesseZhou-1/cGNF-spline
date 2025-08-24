# cGNF/GNF_Modules/Normalizers/SplineNormalizer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn import Parameter
from nflows.transforms.splines.rational_quadratic import unconstrained_rational_quadratic_spline
from .Normalizer import Normalizer

class SplineNormalizer(Normalizer):
    def __init__(
        self,
        cond_size: int,
        num_bins: int = 8,
        bound: float = 3.0,
        hidden_dims=(64, 64),
        cat_dims=None,
        mu=None,
        sigma=None
    ):
        super().__init__()
        # interface attributes
        self.cat_dims = cat_dims
        # normalization parameters (always present)
        mu_tensor = mu if mu is not None else torch.zeros(1)
        sigma_tensor = sigma if sigma is not None else torch.ones(1)
        self.mu = Parameter(mu_tensor, requires_grad=False)
        self.sigma = Parameter(sigma_tensor, requires_grad=False)
        # optional dequantization noise
        if self.cat_dims is not None:
            self.U_noise = MultivariateNormal(
                torch.zeros(len(self.cat_dims)),
                torch.eye(len(self.cat_dims)) / (6 * 4)
            )
        self.num_bins = num_bins
        self.bound = bound
        # param_dim = 3*K - 1: K widths, K heights, K-1 derivatives
        param_dim = 3 * num_bins - 1
        layers = []
        input_dim = cond_size
        for hdim in hidden_dims:
            layers.append(nn.Linear(input_dim, hdim))
            layers.append(nn.ReLU())
            input_dim = hdim
        layers.append(nn.Linear(input_dim, param_dim))
        self.cond_net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, h: torch.Tensor, context=None):
        """
        x: [batch, D], h: [batch, D, cond_size]
        Returns:
          z: [batch, D], jac: [batch]
        """
        batch, D = x.shape

        # Flatten across batch and feature dims
        x_flat = x.view(-1)
        h_flat = h.view(-1, h.size(-1))  # [batch*D, cond_size]

        # Compute spline parameters per feature
        raw = self.cond_net(h_flat)  # [batch*D, param_dim]
        K = self.num_bins
        widths_raw = raw[:, :K]
        heights_raw = raw[:, K:2*K]
        derivs_raw = raw[:, 2*K:]

        # sanity checks (safe to remove later)
        assert widths_raw.shape[1] == K, f"widths_raw has {widths_raw.shape[1]} cols, expected {K}"
        assert heights_raw.shape[1] == K, f"heights_raw has {heights_raw.shape[1]} cols, expected {K}"
        assert derivs_raw.shape[1] == K - 1, f"derivs_raw has {derivs_raw.shape[1]} cols, expected {K - 1}"

        # Apply spline
        z_flat, ldj_flat = unconstrained_rational_quadratic_spline(
            inputs=x_flat,
            unnormalized_widths=widths_raw,
            unnormalized_heights=heights_raw,
            unnormalized_derivatives=derivs_raw,
            inverse=False,
            tails="linear",
            tail_bound=self.bound
        )

        # Reshape outputs
        z = z_flat.view(batch, D)
        # Convert log-dets to absolute dets per feature
        absdet = torch.exp(ldj_flat).view(batch, D)
        return z, absdet

    def inverse_transform(self, z: torch.Tensor, h: torch.Tensor, context=None):
        """
        z: [batch, D], h: [batch, D, cond_size]
        Returns:
          x_rec: [batch, D], jac: [batch] (optional)
        """
        batch, D = z.shape
        z_flat = z.view(-1)
        h_flat = h.view(-1, h.size(-1))

        raw = self.cond_net(h_flat)
        K = self.num_bins
        widths_raw = raw[:, :K]
        heights_raw = raw[:, K:2*K]
        derivs_raw = raw[:, 2*K:]

        # sanity checks (safe to remove later)
        assert widths_raw.shape[1] == K, f"widths_raw has {widths_raw.shape[1]} cols, expected {K}"
        assert heights_raw.shape[1] == K, f"heights_raw has {heights_raw.shape[1]} cols, expected {K}"
        assert derivs_raw.shape[1] == K - 1, f"derivs_raw has {derivs_raw.shape[1]} cols, expected {K - 1}"

        x_flat, ldj_flat = unconstrained_rational_quadratic_spline(
            inputs=z_flat,
            unnormalized_widths=widths_raw,
            unnormalized_heights=heights_raw,
            unnormalized_derivatives=derivs_raw,
            inverse=True,
            tails="linear",
            tail_bound=self.bound
        )

        x_rec = x_flat.view(batch, D)
        return x_rec
