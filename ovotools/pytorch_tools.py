import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DummyTimer:
    '''
    replacement for IgniteTimer if it is not provided
    '''

    class TimerWatch:
        def __init__(self, timer, name): pass
        def __enter__(self): return self
        def __exit__(self, *args): return False

    def __init__(self): pass
    def start(self, name): pass
    def end(self, name): pass
    def watch(self, name): return self.TimerWatch(self, name)


class DataSubset:
    '''
    subset of base_dataset defined by indexes
    '''
    def __init__(self, base_dataset, indexes):
        self.base_dataset = base_dataset
        self.indexes = indexes
    def __len__(self):
        return len(self.indexes)
    def __getitem__(self, item):
        return self.base_dataset[self.indexes[item]]


class MarginBaseLoss:
    '''
    L2-constrained Softmax Loss for Discriminative Face Verification https://arxiv.org/pdf/1703.09507
    margin based loss with distance weighted sampling https://arxiv.org/pdf/1706.07567.pdf
    '''
    ignore_index = -100
    def __init__(self, model, classes, device, params):
        assert params.data.samples_per_class >= 2
        self.model = model
        self.device = device
        self.params = params
        self.classes = sorted(classes)
        self.classes_dict = {v: i for i, v in enumerate(self.classes)}
        self.lambda_rev = 1/params.distance_weighted_sampling.lambda_
        self.timer = DummyTimer()
        print('classes: ', len(self.classes))

    def set_timer(self, timer):
        self.timer = timer

    def classes_to_ids(self, y_class, ignore_index = -100):
        return torch.tensor([self.classes_dict.get(int(c.item()), ignore_index) for c in y_class]).to(self.device)

    def l2_loss(self, net_output, y_class):
        with self.timer.watch('time.l2_loss'):
            pred_class = net_output[0]
            class_nos = self.classes_to_ids(y_class, ignore_index=self.ignore_index)
            self.l2_loss_val = torch.nn.CrossEntropyLoss(ignore_index=self.ignore_index)(pred_class, class_nos)
            return self.l2_loss_val

    def last_l2_loss(self, net_output, y_class):
        return self.l2_loss_val

    def mb_loss(self, net_output, y_class):
        with self.timer.watch('time.mb_loss'):
            pred_embeddings = net_output[1]
            loss = 0
            n = len(pred_embeddings) # samples in batch
            dim =  pred_embeddings[0].shape[0] # dimensionality
            self.true_pos = 0
            self.true_neg = 0
            self.false_pos = 0
            self.false_neg = 0

            alpha = self.model.mb_loss_alpha if self.params.mb_loss.train_alpha else self.model.mb_loss_alpha.detach()
            alpha2 = self.model.mb_loss_alpha if self.params.mb_loss.train_alpha else 0

            assert len(pred_embeddings.shape) == 2, pred_embeddings.shape
            norm = (pred_embeddings ** 2).sum(1)
            self.d_ij = norm.view(-1, 1) + norm.view(1, -1) - 2.0 * torch.mm(pred_embeddings, torch.transpose(pred_embeddings, 0, 1)) #https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/8
            self.d_ij = torch.sqrt(torch.clamp(self.d_ij, min=0.0) + 1.0e-8)

            for i_start in range(0, n, self.params.data.samples_per_class): # start of class block
                i_end = i_start + self.params.data.samples_per_class # start of class block
                for i in range(i_start, i_end):
                    d = self.d_ij[i,:].detach()
                    prob = torch.exp(-(d - 1.4142135623730951)**2 * dim) #https://arxiv.org/pdf/1706.07567.pdf
                    weights = (1/prob.clamp(min = self.lambda_rev)).cpu().numpy()
                    weights[i] = 0 # dont join with itself
                    # select positive pair
                    weights_same = weights[i_start: i_end] # i-th element already excluded
                    j = np.random.choice(range(i_start, i_end), p = weights_same/np.sum(weights_same), replace=False)
                    assert j != i
                    loss += (alpha + (self.d_ij[i,j] - self.model.mb_loss_beta)).clamp(min=0) - alpha2 #https://arxiv.org/pdf/1706.07567.pdf
                    # select neg. pair
                    weights = np.delete(weights, np.s_[i_start: i_end], axis=0)
                    with self.timer.watch('time.mb_loss_k'):
                        k = np.random.choice(range(0, n - self.params.data.samples_per_class), p = weights/np.sum(weights), replace=False)
                    if k >= i_start:
                        k += self.params.data.samples_per_class
                    loss += ((alpha - (self.d_ij[i,k] - self.model.mb_loss_beta)).clamp(min=0) - alpha2)*self.params.mb_loss.neg2pos_weight  #https://arxiv.org/pdf/1706.07567.pdf
                    self.mb_loss_val = loss[0] / len(pred_embeddings)
                    with self.timer.watch('time.mb_loss_acc1'):
                        '''
                        negative = (d > self.model.mb_loss_beta.detach()).float()
                        positive = (d <= self.model.mb_loss_beta.detach()).float()
                        '''
                        negative = (d > self.model.mb_loss_beta.detach())
                        positive = (~negative).float()
                        negative = negative.float()
                    with self.timer.watch('time.mb_loss_acc2'):
                        fn = (negative[i_start: i_end]).sum()
                        self.false_neg += fn
                        tp = (positive[i_start: i_end]).sum()
                        self.true_pos += tp
                        fp = (positive[: i_start]).sum() + (positive[i_end:]).sum()
                        self.false_pos += fp
                        fn = (negative[: i_start]).sum() + (negative[i_end:]).sum()
                        self.true_neg += fn
            self.true_pos /= n
            self.true_neg /= n
            self.false_pos /= n
            self.false_neg /= n
            return self.mb_loss_val

    def last_mb_loss(self, net_output, y_class):
        return self.mb_loss_val

    def last_false_pos(self, net_output, y_class):
        return self.false_pos

    def last_false_neg(self, net_output, y_class):
        return self.false_neg

    def last_true_pos(self, net_output, y_class):
        return self.true_pos

    def last_true_neg(self, net_output, y_class):
        return self.true_neg

    def loss(self, net_output, y_class):
        self.loss_val = self.l2_loss(net_output, y_class) + self.mb_loss(net_output, y_class)
        return self.loss_val



def save_model(model, params, rel_dir, filename):
    file_name = os.path.join(params.get_base_filename(), rel_dir, filename)
    dir_name = os.path.dirname(file_name)
    os.makedirs(dir_name, exist_ok=True)
    torch.save(model.state_dict(), file_name)


class FocalBceLoss(nn.Module):
    def __init__(self, weight=1, gamma=2, logits=False, reduce=True):
        super(FocalBceLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            bce_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-bce_loss)
        f_loss = self.weight * (1-pt)**self.gamma * bce_loss

        if self.reduce:
            return torch.mean(f_loss)
        else:
            return f_loss


class FocalCeLoss(nn.Module):
    def __init__(self, weight=None, gamma=2, logits=False, reduce=True):
        super(FocalCeLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        else:
            ce_loss = F.nll_loss(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        f_loss = (1 - pt) ** self.gamma * ce_loss
        if self.weight is not None:
            weights = torch.index_select(self.weight, 0, targets.view(-1)).view(targets.shape)
            f_loss *= weights
        if self.reduce:
            return torch.mean(f_loss)
        else:
            return f_loss

