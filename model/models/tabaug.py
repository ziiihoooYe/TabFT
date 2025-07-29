# tabaug_model.py

import torch
import torch.nn as nn
from torch import Tensor
import delu
from typing import Literal, List

# 假设这些辅助函数位于同一路径或可访问的路径
from model.lib.tabm.tabm import _init_scaling_by_sections
from model.lib.tabm.deep import ElementwiseAffineEnsemble, make_efficient_ensemble
from model.lib.tabr.utils import make_module1, MLP, ResNet

def _get_first_input_scaling(backbone):
    if isinstance(backbone, MLP):
        return backbone.blocks[0][0]
    elif isinstance(backbone, ResNet):
        return backbone.blocks[0][1] if backbone.proj is None else backbone.proj
    else:
        raise RuntimeError(f'Unsupported backbone: {backbone}')

class TabAugModel(nn.Module):
    def __init__(
        self,
        *,
        n_num_features: int,
        cat_cardinalities: List[int],
        n_classes: None | int,
        backbone: dict,
        arch_type: Literal['vanilla', 'tabm', 'tabm-mini', 'tabm-naive'],
        k: None | int = None,
        **kwargs
    ) -> None:
        super().__init__()

        if arch_type == 'vanilla':
            assert k is None
        else:
            assert k is not None and k > 0

        self.k = k
        self.arch_type = arch_type

        d_num = n_num_features
        d_cat = sum(cat_cardinalities)
        self.d_in = d_num + d_cat

        scaling_init_sections = []
        scaling_init_sections.extend(1 for _ in range(n_num_features))
        scaling_init_sections.extend(cat_cardinalities)
        
        # >>> 主干网络
        self.affine_ensemble = None
        self.backbone = make_module1(d_in=self.d_in, **backbone)

        # ... (与上次修改相同的 make_efficient_ensemble 和 _init_scaling_by_sections 逻辑)
        if arch_type != 'vanilla':
            # ... (tabm, tabm-mini, etc. logic)
            if arch_type == 'tabm':
                 make_efficient_ensemble(
                    self.backbone, k=k, ensemble_scaling_in=True,
                    ensemble_scaling_out=True, ensemble_bias=True, scaling_init='ones'
                )
                 _init_scaling_by_sections(
                    _get_first_input_scaling(self.backbone).r,
                    'random-signs',
                    scaling_init_sections,
                )
            # ... (其他 arch_type)

        # >>> 输出层
        d_block = backbone['d_block']

        # --- 这是关键的修改 ---
        # 移除对 n_classes == 2 的特殊处理，与 TabM 保持一致
        self.d_out = 1 if n_classes is None else n_classes
        # --- 修改结束 ---

        self.output = (
            nn.Linear(d_block, self.d_out)
            if arch_type == 'vanilla'
            else delu.nn.NLinear(k, d_block, self.d_out)
        )

    def forward(self, x: Tensor) -> Tensor:
        # forward 函数保持不变，它本身是正确的
        assert x.ndim == 3, "Input tensor must be 3-dimensional (B, n_aug, D_in)"
        B, n_aug, D_in = x.shape
        
        x = x.reshape(B * n_aug, D_in)

        if self.k is not None:
            x = x.unsqueeze(1).expand(-1, self.k, -1)
            if self.affine_ensemble is not None:
                x = self.affine_ensemble(x)
        else:
            x = x.unsqueeze(1)
            assert self.affine_ensemble is None

        x = self.backbone(x)
        x = self.output(x)

        x = x.view(B, n_aug, self.k or 1, self.d_out)

        # 这个 squeeze 只对回归或单输出二分类生效，但根据上面的修改，
        # self.d_out 在分类任务中将始终 >= 2。所以这条语句现在只对回归任务(n_classes=None)生效。
        if self.d_out == 1:
            x = x.squeeze(-1)
            
        return x.float()