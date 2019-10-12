import torch


class MeanLoss(torch.nn.modules.loss._Loss):
    '''
    New loss calculated as mean of base binary loss calculated for all channels separately
	loss values for individual channels are stored in get_val
    '''
    def __init__(self, base_binary_locc):
        super(MeanLoss, self).__init__()
        self.base_loss = base_binary_locc
        self.loss_vals = []

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        assert y_true.shape == y_pred.shape, (y_pred.shape, y_true.shape)
        self.loss_vals = [self.base_loss(y_pred[:, i, ...], y_true[:, i, ...]) for i in range(y_true.shape[1])]
        res = torch.stack(self.loss_vals).mean()
        self.loss_vals.append(res)
        return res

    def __len__(self):
        '''
        returns number of individual channel losses (not including the last value stored in self.loss_vals)
        '''
        return len(self.loss_funcs)

    def get_val(self, index):
        '''
		returns function that returns individual channel loss cor channel index
		valid indexes are 0..self.len(). The last index=self.len() or index=1 is mean value returned by forward()
        '''
        def call(*kargs, **kwargs):
            return self.loss_vals[index]
        return call



