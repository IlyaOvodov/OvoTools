from typing import List
import torch
from torch.nn.modules.loss import _Loss


class SimpleLoss(_Loss):
    '''
    Wrapper over any usual loss function, that
    1) stores calculated loss as `val` attribute
    2) provides extra interfaces similar to CompositeLoss (`get_val`, `len`, `get_subval`)
    3) if `key` is defined, loss is calculated against y_true[key] instead of y_true
    '''
    def __init__(self, loss_func: _Loss, dict_key = None):
        super(SimpleLoss, self).__init__()
        self.loss_func = loss_func
        self.dict_key = dict_key
        self.val = None

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if self.dict_key:
            y_true = y_true[self.dict_key]
        self.val = self.loss_func(y_pred, y_true)
        return self.val

    def __len__(self):
        return 0

    def get_val(self):
        def call(*kargs, **kwargs):
            return self.val
        return call

    def get_subval(self, index):
        def call(*kargs, **kwargs):
            return None
        return call


class CompositeLoss(_Loss):
    '''
    Wrapper to calculate weighted sum of losses
    Also stores calculates value and calculated values of each loss being composed of
    '''
    def __init__(self, loss_funcs: List):
        '''
        :param loss_funcs: list of (loss_function, weight,)
        '''
        super(CompositeLoss, self).__init__()
        self.loss_funcs = loss_funcs
        self.val = None
        self.sub_vals = [None] * (len(self.loss_funcs) + 1)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        self.sub_vals = [loss_fn(y_pred, y_true) for (loss_fn, _,) in self.loss_funcs]
        self.val = sum([w * self.sub_vals[i] for i, (_, w,) in enumerate(self.loss_funcs)])
        return self.val

    def __len__(self):
        '''
        :return: number of component losses
        '''
        return len(self.loss_funcs)

    def get_val(self):
        '''
        :return: callable to get last calculated value
        '''
        def call(*kargs, **kwargs):
            return self.val
        return call

    def get_subval(self, index):
        '''
        :param index: index of component loss
        :return: callable to get last calculates value of component loss by index
        '''
        def call(*kargs, **kwargs):
            return self.sub_vals[index]
        return call



