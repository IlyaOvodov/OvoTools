import torch
import math

class PseudoLabelingBCELoss(torch.nn.modules.loss._Loss):
    '''
    '''
    def __init__(self, confindence_thr = 0.2, **kwargs):
        super(PseudoLabelingBCELoss, self).__init__()
        self.base_loss = torch.nn.BCEWithLogitsLoss(reduction = 'none')
        self.logit_thr = math.log((1-confindence_thr)/confindence_thr)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            pseudo_labels = (y_pred.sign()+1)/2
            pseudo_labels_mask = (y_pred.abs()>self.logit_thr).float()
            pseudo_labels_cnt = pseudo_labels_mask.sum()
        self.val = (pseudo_labels_mask*self.base_loss(y_pred, pseudo_labels)).mean()
        if pseudo_labels_cnt:
            self.val *= torch.Tensor([y_pred.shape]).prod()/pseudo_labels_cnt
        return self.val

    def __len__(self):
        '''
        returns number of individual channel losses
        '''
        return 0

    def get_val(self):
        '''
        returns function to get last result
        '''
        def call(*kargs, **kwargs):
            return self.val
        return call
