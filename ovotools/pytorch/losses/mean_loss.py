import torch


class MeanLoss(torch.nn.modules.loss._Loss):
    '''
    New loss calculated as mean of base binary loss calculated for all channels separately
	loss values for individual channels are stored in sub_vals
    '''
    def __init__(self, base_binary_locc):
        super(MeanLoss, self).__init__()
        self.base_loss = base_binary_locc
        self.sub_vals = []

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        assert y_true.shape == y_pred.shape, (y_pred.shape, y_true.shape)
        self.sub_vals = [self.base_loss(y_pred[:, i, ...], y_true[:, i, ...]) for i in range(y_true.shape[1])]
        self.val = torch.stack(self.sub_vals).mean()
        return self.val

    def __len__(self):
        '''
        returns number of individual channel losses
        '''
        return len(self.sub_vals)

    def get_val(self):
        '''
        returns function to get last result
        '''
        def call(*kargs, **kwargs):
            return self.val
        return call

    def get_subval(self, index):
        '''
		returns function that returns individual channel loss cor channel index
        '''
        def call(*kargs, **kwargs):
            return self.sub_vals[index]
        return call