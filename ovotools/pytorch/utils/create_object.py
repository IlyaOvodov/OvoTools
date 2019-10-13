from typing import Callable
import torch
from ovotools import AttrDict
from ..losses import SimpleLoss, CompositeLoss, MeanLoss


def create_object(params: dict, eval_func: Callable = eval, *args, **kwargs) -> object:
    '''
    Create object of type params['type'] using *args, **kwargs and parameters params['params'].
    params['params'] is optional.

    Example:
        create_object({'type': 'torch.nn.Conv2d', 'params': {'in_channels': 64, 'out_channels': 32, 'kernel_size': 3} })
        create_object({'type': 'torch.nn.BCELoss'})

    :param params: dict describing the object. Must contain ['type'] and optional ['params']
    :param eval_func: function to convert ['type'] string to object class. Usual usecase is calling eval(x)
        in a context of the calling module
    :param args: args to be passed to the constructor
    :param kwargs:
    :return: created object
    '''
    all_kwargs = kwargs.copy()
    p = params.get('params', dict())
    all_kwargs.update(p)
    print('creating: ', params['type'], repr(dict(p)))
    obj = eval_func(params['type'])(*args, **all_kwargs)
    return obj


def create_optional_object(params: dict, key: str, eval_func = eval, *args, **kwargs) -> object:
    '''
    Create object of type params[<key>]['type'] using *args, **kwargs and parameters params[<key>]['params'].
    If no params[<key>] or params[<key>]['type'] is defined, returns None.
    params[<key>]['params'] is optional

    Example:
        create_object({'type': 'torch.nn.Conv2d', 'params': {'in_channels': 64, 'out_channels': 32, 'kernel_size': 3} })
        create_object({'type': 'torch.nn.BCELoss'})

    :param params: dict containig params[<key>] describing the object. params[<key>] must contain ['type'] and optional ['params']
        for object to be created
    :param key: string, key in params dict
    :param eval_func: function to convert ['type'] string to object class. Usual usecase is calling eval(x)
        in a context of the calling module
    :param args: args to be passed to the constructor
    :param kwargs:
    :return: created object
    '''
    p = params.get(key)
    if not p:
        print('NO '+ key + ' is set')
        return None
    if not p.get('type'):
        print('NO '+ key + '["type"] is set')
        return None
    return create_object(p, eval_func, *args, **kwargs)


class Context(AttrDict):
    def __init__(self, settings: dict, params: dict, eval_func = eval):
        '''
        Creates standard context AttrDict with params, settings, net, loss, optimizer

        :param settings: parameters of program run, not of algorithm. Not expected to imfluence the result.
        :param params:   parameters of algorithm. To be stored with result.
        :return: context AttrDict
        '''
        super(Context, self).__init__(
            settings = settings,
            params = params,
            eval_func = eval_func,
            net = None,
            loss = None,
            optimizer = None,
        )

    def create_model(self, train=True):
        return create_model(self, train)

    def create_optim(self):
        return create_optim(self)

    def create_lr_scheduler(self):
        return create_lr_scheduler(self)

    def create_loss(self):
        return create_loss(self)


def create_model(context: Context, train=True) -> torch.nn.Module:
    '''
    Creates model using standard context structure (context.params.model)
    Stores model in context.net

    :return updated context
    '''
    context.net = create_object(context.params.model, context.eval_func)
    context.net.train(train)
    return context.net


def create_optim(context: Context) -> torch.optim.Optimizer:
    '''
    Creates optimizer using standard context structure (context.params.optimizer)
    Stores optimizer in context.optimizer
    self.net must be created before

    :return updated context
    '''
    context.optimizer = create_object(context.params.optimizer, context.eval_func, context.net.parameters())
    return context.optimizer


def create_lr_scheduler(context: Context) -> object:
    '''
    Creates lr_scheduler using standard context structure (context.params.lr_scheduler)
    Stores lr_scheduler in context.lr_scheduler
    self.optimizer must be created before

    :return updated context
    '''
    context.lr_scheduler = create_object(context.params.lr_scheduler, context.eval_func, optimizer=context.optimizer)
    return context.lr_scheduler


def create_loss(context: Context) -> torch.nn.modules.loss._Loss:
    '''
    Creates loss using standard context structure (context.params.loss)
    Stores loss in context.loss

    :return updated context
    '''
    context.loss = CreateCompositeLoss(context.params.loss, context.eval_func)
    return context.loss


def CreateCompositeLoss(loss_params: dict, eval_func=eval) -> torch.nn.modules.loss._Loss:
    '''
    creates loss using loss_params

    :param loss_params: dict can be dict (to create single loss or list of loss_params (to create composite loss)
        loss dict can contain optional 'weight' (for composite loss) and 'mean' values.
    :param eval_func:
    :return:
    '''
    if isinstance(loss_params, dict):
        loss = create_object(loss_params, eval_func)
        if loss_params.get('mean', False):
            loss = MeanLoss(loss)
        loss = SimpleLoss(loss, loss_params.get('key'))
        return loss
    else:
        loss_funcs = []
        for loss_param in loss_params:
            loss_i = CreateCompositeLoss(loss_param)
            loss_funcs.append((loss_i,  loss_param.get('weight', 1.),))
        return CompositeLoss(loss_funcs)
