from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class SpectrumEncoder(nn.Module):
    """
    Peak-set neural encoder for tandem mass spectra.

    Input
    -----
    peaks:
        Float tensor with shape:

            [batch, max_peaks, 2]

        peaks[..., 0] = scaled m/z
        peaks[..., 1] = normalized intensity

    mask:
        Boolean tensor with shape:

            [batch, max_peaks]

    precursor_mz:
        Float tensor with shape:

            [batch]

    Output
    ------
    L2-normalized spectrum embedding:

        [batch, embedding_dim]
    """

    def __init__(
        self,
        peak_input_dim: int = 2,
        peak_hidden_dim: int = 128,
        precursor_hidden_dim: int = 32,
        pooled_hidden_dim: int = 256,
        embedding_dim: int = 128,
        dropout: float = 0.10,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim

        # =================================================
        # Peak encoder
        # =================================================

        self.peak_encoder = nn.Sequential(
            nn.Linear(
                peak_input_dim,
                peak_hidden_dim,
            ),
            nn.LayerNorm(
                peak_hidden_dim
            ),
            nn.GELU(),
            nn.Linear(
                peak_hidden_dim,
                peak_hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(
                dropout
            ),
        )

        # =================================================
        # Learned attention score for every peak
        # =================================================

        self.attention = nn.Sequential(
            nn.Linear(
                peak_hidden_dim,
                peak_hidden_dim // 2,
            ),
            nn.GELU(),
            nn.Linear(
                peak_hidden_dim // 2,
                1,
            ),
        )

        # =================================================
        # Precursor embedding
        # =================================================

        self.precursor_encoder = nn.Sequential(
            nn.Linear(
                1,
                precursor_hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(
                precursor_hidden_dim,
                precursor_hidden_dim,
            ),
            nn.GELU(),
        )

        # =================================================
        # Final projection
        #
        # attention pooled peaks
        # +
        # max pooled peaks
        # +
        # precursor embedding
        # =================================================

        pooled_input_dim = (
            peak_hidden_dim * 2
            + precursor_hidden_dim
        )

        self.projection = nn.Sequential(
            nn.Linear(
                pooled_input_dim,
                pooled_hidden_dim,
            ),
            nn.LayerNorm(
                pooled_hidden_dim
            ),
            nn.GELU(),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                pooled_hidden_dim,
                embedding_dim,
            ),
        )

    def attention_pool(
        self,
        peak_features: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Learned weighted pooling over valid peaks.

        Compatible with float32, float16, and bfloat16.
        """

        attention_logits = (
            self.attention(
                peak_features
            )
            .squeeze(-1)
        )

        # -------------------------------------------------
        # Mixed-precision-safe masking
        #
        # Do NOT use -1e9 here because that value cannot
        # be represented by float16.
        # -------------------------------------------------

        minimum_value = torch.finfo(
            attention_logits.dtype
        ).min

        attention_logits = (
            attention_logits.masked_fill(
                ~mask,
                minimum_value,
            )
        )

        weights = torch.softmax(
            attention_logits,
            dim=1,
        )

        # Explicitly zero padded positions.
        weights = (
            weights
            * mask.to(
                weights.dtype
            )
        )

        # -------------------------------------------------
        # Renormalize.
        #
        # For a completely padded spectrum, the denominator
        # would otherwise be zero.
        # -------------------------------------------------

        denominator = (
            weights.sum(
                dim=1,
                keepdim=True,
            )
        )

        denominator = denominator.clamp_min(
            torch.finfo(
                weights.dtype
            ).tiny
        )

        weights = (
            weights
            / denominator
        )

        pooled = torch.sum(
            peak_features
            * weights.unsqueeze(-1),
            dim=1,
        )

        # -------------------------------------------------
        # Extra numerical safety
        # -------------------------------------------------

        pooled = torch.nan_to_num(
            pooled,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        return pooled

    @staticmethod
    def max_pool(
        peak_features: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Maximum pooling over valid peaks.
        """

        # Use dtype-safe minimum instead of -inf.
        minimum_value = torch.finfo(
            peak_features.dtype
        ).min

        masked_features = (
            peak_features.masked_fill(
                ~mask.unsqueeze(-1),
                minimum_value,
            )
        )

        pooled = masked_features.max(
            dim=1
        ).values

        # -------------------------------------------------
        # Handle completely empty spectra.
        # -------------------------------------------------

        valid_spectrum = mask.any(
            dim=1
        )

        pooled = torch.where(
            valid_spectrum.unsqueeze(-1),
            pooled,
            torch.zeros_like(
                pooled
            ),
        )

        pooled = torch.nan_to_num(
            pooled,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        return pooled

    def forward(
        self,
        peaks: torch.Tensor,
        mask: torch.Tensor,
        precursor_mz: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode one batch of MS/MS spectra.
        """

        # =================================================
        # Input validation
        # =================================================

        if peaks.ndim != 3:
            raise ValueError(
                "peaks must have shape "
                "[batch, peaks, features]."
            )

        if mask.ndim != 2:
            raise ValueError(
                "mask must have shape "
                "[batch, peaks]."
            )

        if precursor_mz.ndim != 1:
            raise ValueError(
                "precursor_mz must have shape "
                "[batch]."
            )

        if (
            peaks.shape[0]
            != mask.shape[0]
            or peaks.shape[0]
            != precursor_mz.shape[0]
        ):
            raise ValueError(
                "Batch dimensions of peaks, mask, "
                "and precursor_mz must match."
            )

        if (
            peaks.shape[1]
            != mask.shape[1]
        ):
            raise ValueError(
                "Peak dimension of peaks and mask "
                "must match."
            )

        # Ensure boolean mask.
        mask = mask.bool()

        # =================================================
        # Encode individual peaks
        # =================================================

        peak_features = (
            self.peak_encoder(
                peaks
            )
        )

        # Remove representations produced at padded
        # positions by layer biases.
        peak_features = (
            peak_features
            * mask.unsqueeze(-1).to(
                peak_features.dtype
            )
        )

        # =================================================
        # Global peak pooling
        # =================================================

        attention_pooled = (
            self.attention_pool(
                peak_features,
                mask,
            )
        )

        max_pooled = (
            self.max_pool(
                peak_features,
                mask,
            )
        )

        # =================================================
        # Precursor feature
        # =================================================

        precursor_feature = (
            self.precursor_encoder(
                precursor_mz.unsqueeze(-1)
            )
        )

        # =================================================
        # Combine global features
        # =================================================

        global_features = torch.cat(
            [
                attention_pooled,
                max_pooled,
                precursor_feature,
            ],
            dim=-1,
        )

        # =================================================
        # Projection to embedding space
        # =================================================

        embedding = (
            self.projection(
                global_features
            )
        )

        # Protect against unexpected numerical problems
        # before normalization.
        embedding = torch.nan_to_num(
            embedding,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        # =================================================
        # L2-normalized metric-learning embedding
        # =================================================

        embedding = F.normalize(
            embedding,
            p=2,
            dim=-1,
            eps=1e-8,
        )

        return embedding