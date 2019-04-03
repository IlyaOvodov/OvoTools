import copy
import torch
import ignite

def create_adaptive_supervised_trainer(model, optimizer, loss_fn, metrics={},
                              device=None, non_blocking=False,
                              prepare_batch=ignite.engine._prepare_batch, lr_scale = 1.1, warmup_iters = 50):
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

    def _update(engine, batch):
        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)

        model.train()

        if engine.state.iteration > warmup_iters:
            prev_k = 1
            loss = None
            new_ks_list = (1/lr_scale, lr_scale,)
            with torch.no_grad():
                for new_k in new_ks_list:
                    correct_model(prev_k, new_k)
                    y_pred = model(x)
                    loss0 = loss
                    loss = loss_fn(y_pred, y)
                    prev_k = new_k
                    print('iter\t{}.{}'.format(engine.state.epoch, engine.state.iteration), 'lr',
                          optimizer.param_groups[0]['lr'], '*', new_k, 'loss', loss.item())
                if loss0 < loss or (loss0 == loss and engine.state.iteration % 2):
                    new_k = new_ks_list[0]
                    correct_model(prev_k, new_k)
                for group in optimizer.param_groups:
                    group['lr'] *= new_k

        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

        print('iter\t{}.{}'.format(engine.state.epoch, engine.state.iteration), 'lr', optimizer.param_groups[0]['lr'], 'loss', loss.item())

        return y_pred, y

    engine = ignite.engine.Engine(_update)

    for name, metric in metrics.items():
        metric.attach(engine, 'train:' + name)

    return engine
