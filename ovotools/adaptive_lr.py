import copy
import torch
import ignite

def create_adaptive_supervised_trainer(model, optimizer, loss_fn, metrics={},
                              device=None, non_blocking=False,
                              prepare_batch=ignite.engine._prepare_batch, lr_scale = 1.1, warmup_iters = 50, ls_mult = 3):
    """
    Factory function for creating a trainer for supervised models.
l
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

    def _update(engine, batch):
        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)

        model.train()

        if engine.state.iteration > warmup_iters:
            if engine.state.iteration % 2:
                new_k = 1 / lr_scale
            else:
                new_k =  lr_scale
            for group in optimizer.param_groups:
                group['lr'] *= new_k
        else:
            prev_k = new_k = 1

        if engine.state.iteration > 1:
            optimizer.step()

        if engine.state.iteration > warmup_iters:
            with torch.no_grad():
                y_pred = model(x)
                loss0 = loss_fn(y_pred, y)
                print('iter\t{}.{}'.format(engine.state.epoch, engine.state.iteration), 'lr * {:5.3}'.format(new_k), 'loss', loss0.item())
                prev_k = new_k
                new_k = 1/new_k
                correct_model(prev_k, new_k)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()

        if engine.state.iteration > warmup_iters:
            with torch.no_grad():
                print('iter\t{}.{}'.format(engine.state.epoch, engine.state.iteration), 'lr * {:5.3}'.format(new_k), 'loss', loss.item())

                if loss < loss0 or (loss == loss0 and engine.state.iteration % 2):
                    for group in optimizer.param_groups:
                        group['lr'] *= new_k/prev_k

                print('iter\t{}.{}'.format(engine.state.epoch, engine.state.iteration), 'lr', optimizer.param_groups[0]['lr'], 'loss', loss.item())

        return y_pred, y

    engine = ignite.engine.Engine(_update)

    for name, metric in metrics.items():
        metric.attach(engine, 'train:' + name)

    return engine
