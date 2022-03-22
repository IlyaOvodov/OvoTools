import torch
import torch.nn.functional as F

def LabelSmoothingBCEWithLogitsLoss(label_smoothing = 0.1, **kwargs):
    def loss(y, y_gt):
        '''
        s = 1/(1+exp(-x))
        L_smooth = -(y_gt*log(s) + (1-y_gt)*log(1-s)) = x(1-y_gt) - log(s)
        L_min = -(y_gt*log(y_gt) + (1-y_gt)*log(1-y_gt))
        L = L_smooth - L_min = KL distance
        '''
        y_gt = label_smoothing + (1-2*label_smoothing)*y_gt
        loss_val = y*(1-y_gt) - F.logsigmoid(y)
        if label_smoothing:
            loss_val += y_gt*torch.log(y_gt) + (1-y_gt)*torch.log(1-y_gt)
        loss_val_mean = loss_val.mean()
        return loss_val_mean
    return loss