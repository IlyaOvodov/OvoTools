'''
DANN module. See  https://arxiv.org/abs/1505.07818, https://arxiv.org/abs/1409.7495
'''

import torch
import torch.nn as nn
from torch.autograd import Function

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class DANN_module(nn.Module):
    def __init__(self, gamma = 10., lambda_max = 1., **kwargs):
        super(DANN_module, self).__init__()
        self.gamma = gamma
        self.lambda_max = lambda_max
        self.progress = nn.Parameter(torch.Tensor([0,]), requires_grad = False) # must be updated from outside

    def set_progress(self, value):
        self.progress.data = torch.Tensor([value,]).to(self.progress.device)

    def forward(self, input_data):
        lambda_p = self.lambda_max * (2. / (1. + torch.exp(-self.gamma * self.progress.data)) - 1)
        reverse_feature = ReverseLayerF.apply(input_data, lambda_p)
        return reverse_feature


class Dann_Head(nn.Module):
    def __init__(self, input_dims, num_classes, **kwargs):
        super(Dann_Head, self).__init__()
        self.input_dims = input_dims
        self.pooling_depth = 100
        self.dann_module = DANN_module(**kwargs)
        self.pooling_modules = [nn.Sequential(
                                    nn.Conv2d(dim, self.pooling_depth, 3),
                                    nn.AdaptiveMaxPool2d(1)
                                )
                                for dim in input_dims]
        for i, m in enumerate(self.pooling_modules):
            self.add_module("pooling_module"+str(i), m)
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('dann_fc1', nn.Linear(self.pooling_depth*len(input_dims), 100))
        self.domain_classifier.add_module('dann_bn1', nn.BatchNorm1d(100)) # - dows not work for batch = 1
        #self.domain_classifier.add_module('dann_bn1', nn.GroupNorm(5, 100)) # - dows not work for batch = 1
        self.domain_classifier.add_module('dann_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('dann_fc2', nn.Linear(100, num_classes))
        self.domain_classifier.add_module('dann_softmax', nn.LogSoftmax(dim=1))
        self.loss = nn.NLLLoss()

    def set_progress(self, value):
            self.dann_module.set_progress(value)

    def forward(self, inputs, y_true: torch.Tensor):
        features = []
        for i, x in enumerate(inputs[:len(self.input_dims)]):
            assert x.shape[1] == self.input_dims[i]
            x = self.dann_module(x)
            x = self.pooling_modules[i](x)
            features.append(x)
        feature = torch.cat([f.flatten(start_dim=1) for f in features], dim=1)
        domain_output = self.domain_classifier(feature)
        loss = self.loss(domain_output, y_true.long())
        return loss


class DannEncDecNet(nn.Module):
    def __init__(self, base_net, input_dim, num_classes, **enc_dec_params):
        super(DannEncDecNet, self).__init__()
        self.net = base_net
        self.dann_head = Dann_Head(input_dim, num_classes, **enc_dec_params)
        self.bottleneck_data = None
        self.dann_loss = None

    def set_progress(self, value):
        self.dann_head.set_progress(value)

    def forward(self, x):
        """Sequentially pass `x` trough model`s `encoder` and `decoder` (return logits!)"""
        x = self.net.encoder(x)
        self.bottleneck_data = x
        x = self.net.decoder(x)
        return x

    def calc_dann_loss(self, y_pred, y_true):
        self.dann_loss = self.dann_head(self.bottleneck_data, y_true)
        return self.dann_loss

