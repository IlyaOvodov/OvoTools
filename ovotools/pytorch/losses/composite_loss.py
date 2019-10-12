import torch

class CompositeLoss(torch.nn.modules.loss._Loss):
    def __init__(self, loss_funcs):
        super(CompositeLoss, self).__init__()
        self.loss_funcs = loss_funcs
        self.loss_vals = [None] * (len(self.loss_funcs) + 1)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        self.loss_vals = [loss_fn(y_pred, y_true) for (loss_fn, _,) in self.loss_funcs]
        res = sum([w * self.loss_vals[i] for i, (_, w,) in enumerate(self.loss_funcs)])
        self.loss_vals.append(res)
        return res

    def __len__(self):
        return len(self.loss_funcs)

    def get_val(self, index):
        def call(*kargs, **kwargs):
            return self.loss_vals[index]
        return call



