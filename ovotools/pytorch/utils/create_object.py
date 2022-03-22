from typing import Callable
import torch
from ovotools import AttrDict
from ..losses import SimpleLoss, CompositeLoss, MeanLoss


def create_object(params: dict, eval_func: Callable = eval, *args, **kwargs) -> object:
    '''
    Create object of type params['type'] using *args, **kwargs and parameters params['params'].
    params['params'] is optional.
    Also other options available (see below)

    Examples:
        create_object({'type': 'torch.nn.Conv2d', 'params': {'in_channels': 64, 'out_channels': 32, 'kernel_size': 3} })
        create_object({'type': 'torch.nn.BCELoss'})
        create_object(['torch.nn.Conv2d', {'in_channels': 64, 'out_channels': 32, 'kernel_size': 3} },])
        create_object('torch.nn.BCELoss')
        create_object([['torch.nn.BCELoss',], ['torch.nn.Conv2d', {'in_channels': 64, 'out_channels': 32, 'kernel_size': 3} }]])

    :param params: options available
        1) dict describing the object. Must contain ['type']: str and optional ['params']: dict with constructor params
        2) tuple or list with 1 (type,) or 2 members (type, params) or more (other members are ignored)
        3) string contaning type
        4) list or tuple of options listed above. List of objects is created
    :param eval_func: function to convert ['type'] string to object class. Usual usecase is calling eval(x)
        in a context of the calling module
    :param args: args to be passed to the constructor
    :param kwargs: args to be passed to the constructor
    :return: created object or list of objects if params is list of params.
    '''
    if isinstance(params, dict):
        assert isinstance(params['type'], str)
        all_kwargs = kwargs.copy()
        p = params.get('params', dict())
        assert isinstance(p, dict)
        all_kwargs.update(p)
        print('creating: ', params['type'], repr(dict(p)))
        obj = eval_func(params['type'])(*args, **all_kwargs)
        return obj
    elif isinstance(params, (list, tuple)):
        if len(params) >= 1 and isinstance(params[0], str):
            if len(params) == 1:
                return create_object({'type': params[0]}, eval_func, *args, **kwargs)
            elif len(params) >= 2 and isinstance(params[1], dict):
                return create_object({'type': params[0], 'params': params[1]}, eval_func, *args, **kwargs)
        return [create_object(pi, eval_func, *args, **kwargs) for pi in params]
    elif isinstance(params, str):
        return create_object({'type': params}, eval_func, *args, **kwargs)
    else:
        raise Exception("Invalid call to create_object: params is {}".format(params))


def create_optional_object(params: dict, key: str, eval_func = eval, *args, **kwargs) -> object:
    '''
    Create object of type and with parameters defined in optional params[<key>]
    If no params[<key>] or params[<key>]['type'] is defined, returns None.

    See create_object() for details

    :param params: dict containig optional params[<key>] describing the object.
        if params[<key>] is defined, it must be valid parameter for create_object()
    :param key: string, key in params dict
    :param eval_func: function to convert ['type'] string to object class. Usual usecase is calling eval(x)
        in a context of the calling module
    :param args: args to be passed to the constructor
    :param kwargs: args to be passed to the constructor
    :return: created object
    '''
    p = params.get(key)
    if not p:
        print('NO '+ key + ' is set')
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

    def create_model(self, train=True, *args, **kwargs):
        return create_model(self, train, *args, **kwargs)

    def create_optim(self, *args, **kwargs):
        return create_optim(self, *args, **kwargs)

    def create_lr_scheduler(self, *args, **kwargs):
        return create_lr_scheduler(self, *args, **kwargs)

    def create_loss(self, *args, **kwargs):
        return create_loss(self, *args, **kwargs)


def create_model(context: Context, train=True, *args, **kwargs) -> torch.nn.Module:
    '''
    Creates model using standard context structure (context.params.model)
    Stores model in context.net

    :return updated context
    '''
    context.net = create_object(context.params.model, context.eval_func, *args, **kwargs)
    context.net.train(train)
    return context.net


def create_optim(context: Context, *args, **kwargs) -> torch.optim.Optimizer:
    '''
    Creates optimizer using standard context structure (context.params.optimizer)
    Stores optimizer in context.optimizer
    self.net must be created before

    :return updated context
    '''
    context.optimizer = create_object(context.params.optimizer, context.eval_func, context.net.parameters(), *args, **kwargs)
    return context.optimizer


def create_lr_scheduler(context: Context, *args, **kwargs) -> object:
    '''
    Creates lr_scheduler using standard context structure (context.params.lr_scheduler)
    Stores lr_scheduler in context.lr_scheduler
    self.optimizer must be created before

    :return updated context
    '''
    context.lr_scheduler = create_object(context.params.lr_scheduler, context.eval_func, *args, optimizer=context.optimizer, **kwargs)
    return context.lr_scheduler


def create_loss(context: Context, *args, **kwargs) -> torch.nn.modules.loss._Loss:
    '''
    Creates loss using standard context structure (context.params.loss)
    Stores loss in context.loss

    :return updated context
    '''
    context.loss = CreateCompositeLoss(context.params.loss, context.eval_func, *args, **kwargs)
    return context.loss


def CreateCompositeLoss(loss_params: dict, eval_func=eval, *args, **kwargs) -> torch.nn.modules.loss._Loss:
    '''
    creates loss using loss_params

    :param loss_params: params in any form accepted by to `creat_object()`
        if `loss_params` describes list of losses. `CompositeLoss` is created, otherwise `SimpleLoss`.
        if loss is described in loss_params as dict, it can contain keys:
        'key' - to calculate loss against `y_true[key]` instead of `y_true`
        'weight' - weight of loss in composite loss
        'mean' - to create `MeanLoss` (calculate loss over channels and then mean over channels).
    :param eval_func:
    :return: SimpleLoss or CompositeLoss object
    '''
    loss = create_object(loss_params, eval_func, *args, **kwargs)
    if not isinstance(loss, (list, tuple)):
        key = None
        if isinstance(loss_params, dict):
            if loss_params.get('mean', False):
                loss = MeanLoss(loss)
            key = loss_params.get('key')
        loss = SimpleLoss(loss, key)
        return loss
    else:
        loss_funcs = []
        for loss_param in loss_params:
            loss_i = CreateCompositeLoss(loss_param, eval_func=eval_func, *args, **kwargs)
            weight = loss_param.get('weight', 1.) if isinstance(loss_param, dict) else 1.
            loss_funcs.append((loss_i,  weight,))
        return CompositeLoss(loss_funcs)
