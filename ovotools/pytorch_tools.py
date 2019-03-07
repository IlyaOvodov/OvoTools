import torch
import numpy as np

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
        print('classes: ', len(self.classes))

    def classes_to_ids(self, y_class, ignore_index = -100):
        return torch.tensor([self.classes_dict.get(int(c.item()), ignore_index) for c in y_class]).to(self.device)

    def l2_loss(self, net_output, y_class):
        pred_class = net_output[0]
        class_nos = self.classes_to_ids(y_class, ignore_index=self.ignore_index)
        return torch.nn.CrossEntropyLoss(ignore_index=self.ignore_index)(pred_class, class_nos)

    def D(self, pred_embeddings, i ,j):
        if i == j:
            return 0
        return torch.dist(pred_embeddings[i], pred_embeddings[j]).item()


    def mb_loss(self, net_output, y_class):
        pred_embeddings = net_output[1]
        loss = 0
        n = len(pred_embeddings) # samples in batch
        dim =  pred_embeddings[0].shape[0] # dimensionality
        for i_start in range(0, n, self.params.data.samples_per_class): # start of class block
            i_end = i_start + self.params.data.samples_per_class # start of class block
            for i in range(i_start, i_end -1):
                d_ij = [0 if i==j else self.D(pred_embeddings, i, j) for j in range(n)]
                weights = [1/max(self.lambda_rev, pow(d,dim-2)*pow(1-d*d/4, (dim-3)/2))  # https://arxiv.org/pdf/1706.07567.pdf
                           for id, d in enumerate(d_ij) if id != i] # dont join with itself
                weights_same = np.asarray(weights[i_start: i_end-1]) # i-th element already excluded
                j = np.random.choice(range(i_start, i_end-1), p = weights_same/np.sum(weights_same) )
                if j >= i:
                    j += 1
                # for j in range(i+1, i_end): # positive pair
                loss += (self.params.mb_loss.alpha + (d_ij[j] - self.model.mb_loss_beta)).clamp(min=0)
                # select neg. pait
                weights[i_start: i_end - 1] = []  # i-th element already excluded
                weights = np.asarray(weights)
                weights = weights/np.sum(weights)
                k = np.random.choice(range(0, n - self.params.data.samples_per_class), p = weights)
                if k >= i_start:
                    k += self.params.data.samples_per_class
                loss += (self.params.mb_loss.alpha - (d_ij[k] - self.model.mb_loss_beta)).clamp(min=0)
        return loss[0] / len(pred_embeddings)


    def loss(self, net_output, y_class):
        return self.l2_loss(net_output, y_class) + self.mb_loss(net_output, y_class)
