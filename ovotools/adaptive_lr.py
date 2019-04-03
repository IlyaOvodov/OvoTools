import copy
import torch
import ignite

def create_adaptive_supervised_trainer(model, optimizer, loss_fn, metrics={},
                              device=None, non_blocking=False,
                              prepare_batch=ignite.engine._prepare_batch, lr_scale = 2):
    """
    Factory function for creating a trainer for supervised models.

    Args:
        model (`torch.nn.Module`): the model to train.
        optimizer (`torch.optim.Optimizer`): the optimizer to use.
        loss_fn (torch.nn loss function): the loss function to use.
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
        non_blocking (bool, optional): if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch (callable, optional): function that receives `batch`, `device`, `non_blocking` and outputs
            tuple of tensors `(batch_x, batch_y)`.

    Note: `engine.state.output` for this engine is the loss of the processed batch.

    Returns:
        Engine: a trainer engine with supervised update function.
    """
    if device:
        model.to(device)

    def correct_model(prev_k, new_k):
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                p.data.add_(-group['lr'] * (new_k - prev_k), d_p)

    def _update1(engine, batch):
        model.train()
        optimizer.zero_grad()
        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

        prev_k = 1
        new_k = lr_scale
        multiply_k = True

        print('epoch', engine.state.epoch,'iter',engine.state.iteration, 'base', optimizer.param_groups[0]['lr'], 1, loss)

        if engine.state.epoch <= 1:
            return y_pred, y

        with torch.no_grad():
            while True:
                correct_model(prev_k, new_k)
                y_pred2 = model(x)
                loss2 = loss_fn(y_pred2, y)
                print('new ', optimizer.param_groups[0]['lr'], new_k, loss2)
                if loss2>=loss:
                    correct_model(new_k, prev_k)
                    if multiply_k and prev_k == 1:
                        multiply_k = False
                        new_k = prev_k/lr_scale
                    else:
                        break
                else:
                    y_pred = y_pred2
                    loss = loss2
                    prev_k = new_k
                    if multiply_k:
                        new_k *= lr_scale
                    else:
                        new_k /= lr_scale

            for group in optimizer.param_groups:
                group['lr'] *= prev_k
            print('fin ', optimizer.param_groups[0]['lr'], loss)

        return y_pred, y

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

        prev_k = 1
        new_k = lr_scale
        multiply_k = True

        print('epoch', engine.state.epoch,'iter',engine.state.iteration, 'base', optimizer.param_groups[0]['lr'], 1, loss.item())

        if engine.state.epoch <= 1:
            return y_pred, y

        with torch.no_grad():
            while True:
                correct_model(prev_k, new_k)
                y_pred2 = model(x)
                loss2 = loss_fn(y_pred2, y)
                print('new ', optimizer.param_groups[0]['lr'], new_k, loss2.item())
                if loss2>=loss:
                    correct_model(new_k, prev_k)
                    if multiply_k and prev_k == 1:
                        multiply_k = False
                        new_k = prev_k/lr_scale
                    else:
                        break
                else:
                    y_pred = y_pred2
                    loss = loss2
                    prev_k = new_k
                    break
                    '''
                    if multiply_k:
                        new_k *= lr_scale
                    else:
                        new_k /= lr_scale
                    '''

            for group in optimizer.param_groups:
                group['lr'] *= prev_k
            print('fin ', optimizer.param_groups[0]['lr'], loss)

        return y_pred, y

    engine = ignite.engine.Engine(_update)

    for name, metric in metrics.items():
        metric.attach(engine, 'train:' + name)

    return engine
