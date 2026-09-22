from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoleculeEncoder(nn.Module):
    """
    Encode a fixed-length molecular fingerprint into a
    normalized dense embedding.

    Input:
        [batch, fingerprint_dim]

    Output:
        [batch, embedding_dim]

    The output is L2-normalized so cosine similarity can be
    used directly against spectrum embeddings.
    """

    def __init__(
        self,
        fingerprint_dim: int = 2048,
        hidden_dim_1: int = 1024,
        hidden_dim_2: int = 512,
        embedding_dim: int = 128,
        dropout: float = 0.10,
    ):
        super().__init__()

        self.fingerprint_dim = int(
            fingerprint_dim
        )

        self.embedding_dim = int(
            embedding_dim
        )

        self.network = nn.Sequential(
            nn.Linear(
                fingerprint_dim,
                hidden_dim_1,
            ),
            nn.LayerNorm(
                hidden_dim_1
            ),
            nn.GELU(),
            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden_dim_1,
                hidden_dim_2,
            ),
            nn.LayerNorm(
                hidden_dim_2
            ),
            nn.GELU(),
            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden_dim_2,
                embedding_dim,
            ),
        )

    def forward(
        self,
        fingerprints: torch.Tensor,
    ) -> torch.Tensor:

        if fingerprints.ndim != 2:
            raise ValueError(
                "fingerprints must have shape "
                "[batch, fingerprint_dim]."
            )

        if (
            fingerprints.shape[1]
            != self.fingerprint_dim
        ):
            raise ValueError(
                "Unexpected fingerprint dimension: "
                f"{fingerprints.shape[1]} "
                f"(expected "
                f"{self.fingerprint_dim})."
            )

        fingerprints = (
            fingerprints.float()
        )

        embeddings = (
            self.network(
                fingerprints
            )
        )

        embeddings = F.normalize(
            embeddings,
            p=2,
            dim=-1,
            eps=1e-12,
        )

        return embeddings