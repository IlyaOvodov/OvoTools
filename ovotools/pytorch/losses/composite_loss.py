from typing import List
import torch
from torch.nn.modules.loss import _Loss


class SimpleLoss(_Loss):
    def __init__(self, loss_func: _Loss, dict_key: str = None):
        super(SimpleLoss, self).__init__()
        self.loss_func = loss_func
        self.dict_key = dict_key
        self.val = None

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if self.dict_key and isinstance(y_true, dict):
            y_true = y_true[self.dict_key]
        self.val = self.loss_func(y_pred, y_true)
        return self.val

    def __len__(self):
        return 0

    def get_val(self):
        def call(*kargs, **kwargs):
            return self.val
        return call


class CompositeLoss(_Loss):
    def __init__(self, loss_funcs: List):
        super(CompositeLoss, self).__init__()
        self.loss_funcs = loss_funcs
        self.val = None
        self.sub_vals = [None] * (len(self.loss_funcs) + 1)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        self.sub_vals = [loss_fn(y_pred, y_true) for (loss_fn, _,) in self.loss_funcs]
        self.val = sum([w * self.sub_vals[i] for i, (_, w,) in enumerate(self.loss_funcs)])
        return self.val

    def __len__(self):
        return len(self.loss_funcs)

    def get_val(self):
        def call(*kargs, **kwargs):
            return self.val
        return call

    def get_subval(self, index):
        def call(*kargs, **kwargs):
            return self.sub_vals[index]
        return call



