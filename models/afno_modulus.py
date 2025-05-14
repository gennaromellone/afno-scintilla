import torch.nn as nn
from physicsnemo.models.afno import AFNO


class AFNOModel(nn.Module):
    def __init__(self, img_shape, in_channels, out_channels=1,
                 patch_size=(1, 1), embed_dim=64, depth=4,
                 num_blocks=8, sparsity_threshold=0.01, hard_thresholding_fraction=1.0):
        super().__init__()

        self.model = AFNO(
            inp_shape=list(img_shape),
            in_channels=in_channels,
            out_channels=out_channels,
            patch_size=list(patch_size),
            embed_dim=embed_dim,
            depth=depth,
            num_blocks=num_blocks,
            sparsity_threshold=sparsity_threshold,
            hard_thresholding_fraction=hard_thresholding_fraction
        )

    def forward(self, x):
        return self.model(x)
