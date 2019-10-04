import copy
import math
import os
import subprocess
import torch
import ignite
from ignite.engine import Events
import collections
import time
import tensorboardX



class IgniteTimes:
    class TimerWatch:
        def __init__(self, timer, name):
            self.name = name
            self.timer = timer

        def __enter__(self):
            self.timer.start(self.name)
            return self

        def __exit__(self, *args):
            self.timer.end(self.name)
            return False

    def __init__(self, engine, count_iters=False, measured_events={}):
        self.clocks = dict()
        self.sums = collections.defaultdict(float)
        self.counts = collections.defaultdict(int)
        for name, (event_engine, start_event, end_event) in measured_events.items():
            event_engine.add_event_handler(start_event, self.on_start, name)
            event_engine.add_event_handler(end_event, self.on_end, name)
        event = Events.ITERATION_COMPLETED if count_iters else Events.EPOCH_COMPLETED
        engine.add_event_handler(event, self.on_complete)

    def reset_all(self):
        self.clocks.clear()
        self.sums.clear()
        self.counts.clear()

    def start(self, name):
        assert not name in self.clocks
        self.clocks[name] = time.time()

    def end(self, name):
        assert name in self.clocks
        t = time.time() - self.clocks[name]
        self.counts[name] += 1
        self.sums[name] += t
        self.clocks.pop(name)

    def watch(self, name):
        return self.TimerWatch(self, name)

    def on_start(self, engine, name):
        self.start(name)

    def on_end(self, engine, name):
        self.end(name)

    def on_complete(self, engine):
        for n, v in self.sums.items():
            engine.state.metrics[n] = v
        self.reset_all()


class BestModelBuffer:
    def __init__(self, model, metric_name, params, minimize = True, save_to_file = True, verbose = 1):
        self.model = model
        assert metric_name
        self.metric_name = metric_name
        self.minimize = 1 if minimize else -1
        self.save_to_file = save_to_file
        self.verbose = verbose
        self.params = params
        self.reset()

    def reset(self):
        self.best_dict = None
        self.best_score = None
        self.best_epoch = None

    def __call__(self, engine):
        assert self.metric_name in engine.state.metrics.keys(), "{} {}".format(self.metric_name, engine.state.metrics.keys())
        if self.best_score is None or self.best_score*self.minimize > engine.state.metrics[self.metric_name]*self.minimize:
            self.best_score = engine.state.metrics[self.metric_name]
            self.best_dict  = copy.deepcopy(self.model.state_dict())
            self.best_epoch = engine.state.epoch
            if self.verbose:
                print('model for {}={} dumped'.format(self.metric_name, self.best_score))
            if self.save_to_file:
                self.save_model()

    def save_model(self, rel_dir = "models", filename = None):
        if filename is None:
            filename = "best.t7"
        file_name = os.path.join(self.params.get_base_filename(), rel_dir, filename)
        dir_name = os.path.dirname(file_name)
        os.makedirs(dir_name, exist_ok=True)
        torch.save(self.best_dict, file_name)

    def restore(self, model = None):
        assert self.best_dict is not None
        if model is None:
            model = self.model
        if self.verbose:
            print('model for {}={} on epoch {} restored'.format(self.metric_name, self.best_score, self.best_epoch))
        model.load_state_dict(self.best_dict)


class LogTrainingResults:
    def __init__(self, evaluator, loaders_dict, best_model_buffer, params, rel_dir = "", filename = None, duty_cycles = 1):
        '''
        evaluates metrics using evaluator and data loaders, adds them to caller engine metrics, logs and keeps track on best model
        :param evaluator: ignite engine to evaluate metrics
        :param loaders_dict: dict: {'name': data_loader}. Metrics are evaluated for all data_loaders, resulting metric named 'name:original_metric_name'
        :param best_model_buffer: optrional, tracks model with best mertric
        :param params: global params to get params.get_base_filename()
        :param duty_cycles: int or dict: {event: int}: if set, enables functioning only every n-th call fo each event
        '''
        self.evaluator = evaluator
        self.loaders_dict = loaders_dict
        self.best_model_buffer = best_model_buffer
        self.params = params
        if filename is None:
            filename = "log.log"
        self.file_name = os.path.join(self.params.get_base_filename(), rel_dir, filename)

        self.calls_count = collections.defaultdict(int)
        assert isinstance(duty_cycles, (int, dict,))
        self.duty_cycles = duty_cycles

    def __call__(self, engine, event):
        self.calls_count[event] += 1
        if isinstance(self.duty_cycles, int):
            duty_cycles = self.duty_cycles
        elif isinstance(self.duty_cycles, dict):
            duty_cycles = self.duty_cycles[event]
        if self.calls_count[event] % duty_cycles != 0 and self.calls_count[event] != 1: #always run 1st time to provide statistics
            return
        for key,loader in self.loaders_dict.items():
            self.evaluator.run(loader)
            for k,v in self.evaluator.state.metrics.items():
                engine.state.metrics[key+':'+k] = v
        if self.best_model_buffer:
            self.best_model_buffer(engine)
        if event == Events.ITERATION_COMPLETED:
            str = "Epoch:{}.{}\t".format(engine.state.epoch, engine.state.iteration)
        else:
            str = "Epoch:{}\t".format(engine.state.epoch)
        str += '\t'.join(['{}:{:.5f}'.format(k,v) for k,v in engine.state.metrics.items()])
        print(str)

        dir_name = os.path.dirname(self.file_name)
        os.makedirs(dir_name, exist_ok=True)
        with open(self.file_name, 'a') as f:
            f.write(str + '\n')


class TensorBoardLogger:
    SERIES_PLOT_SEPARATOR = ':'
    GROUP_PLOT_SEPARATOR = '.'

    def __init__(self, trainer_engine, params, count_iters=False, period=1, rel_dir = "tb_log"):
        self.log_dir = os.path.join(params.get_base_filename(), rel_dir)
        self.writer = tensorboardX.SummaryWriter(log_dir=self.log_dir, flush_secs = 10)
        event = Events.ITERATION_COMPLETED if count_iters else Events.EPOCH_COMPLETED
        trainer_engine.add_event_handler(event, self.on_event)
        self.period = period
        self.call_count = 0
        trainer_engine.add_event_handler(Events.COMPLETED, self.on_completed)

    def start_server(self, port, start_it = False):
        #cmd = r"tensorboard --host 127.0.0.1 --port {port} --logdir {dir}".format(port=port, dir=self.log_dir)
        #print(cmd)
        #os.popen(cmd)
        cmd = r'tensorboard --host 127.0.0.1 --logdir "" --port {port}'.format(port=port).split(' ')
        cmd[-3] = os.path.abspath(self.log_dir) # log_dir can contain spaces, so need to be set after split
        print(' '.join(cmd))
        if start_it:
            subprocess.Popen(cmd)

    def on_completed(self, engine):
        self.writer.close()

    def on_event(self, engine):
        '''
        engine.state.metrics with name
        *|* are interpreted as series(train,val).plot_name(metric)
        *|*.* are interpreted as series(train,val).group(metric class).plot_name
        '''
        self.call_count += 1
        if self.call_count % self.period != 0:
            return

        metrics = collections.defaultdict(dict)
        for name, value in engine.state.metrics.items():
            name_parts = name.split(self.SERIES_PLOT_SEPARATOR, 1)
            if len(name_parts) == 1:
                name_parts.append(name_parts[0])
            metrics[name_parts[1].replace(self.GROUP_PLOT_SEPARATOR, '/')][name_parts[0]] = value
        for n, d in metrics.items():
            if len(d) == 1:
                for k, v in d.items():
                    self.writer.add_scalar(n, v, self.call_count)
            else:
                self.writer.add_scalars(n, d, self.call_count)
        for path, writer in self.writer.all_writers.items():
            writer.flush()


class ClrScheduler:
    def __init__(self, train_loader, model, optimizer, metric_name, params, minimize = True, engine = None):
        self.optimizer = optimizer
        self.params = params
        self.cycle_index = 0
        self.iter_index = 0
        self.iterations_per_epoch = len(train_loader)
        self.min_lr = params.clr.min_lr
        self.max_lr = params.clr.max_lr
        self.best_model_buffer = BestModelBuffer(model, metric_name, params, minimize = minimize,
                                                 save_to_file = False, verbose = 0)
        if engine:
            self.attach(engine)

    def attach(self, engine):
        engine.add_event_handler(Events.EPOCH_STARTED, self.upd_lr_epoch)
        engine.add_event_handler(Events.ITERATION_STARTED, self.upd_lr)
        engine.add_event_handler(Events.EPOCH_COMPLETED, self.best_model_buffer)

    def upd_lr_epoch(self, engine):
        if (self.cycle_index == 0 and self.iter_index == self.params.clr.warmup_epochs*self.iterations_per_epoch
                                        or self.cycle_index > 0 and self.iter_index == self.params.clr.period_epochs*self.iterations_per_epoch):
            if self.cycle_index > 0:
                self.best_model_buffer.save_model(rel_dir = 'models', filename = 'clr.{:03}.t7'.format(self.cycle_index))
                self.best_model_buffer.restore()
                self.best_model_buffer.reset()
                self.min_lr *= self.params.clr.get('scale_min_lr', 1)
                self.max_lr *= self.params.clr.get('scale_max_lr', 1)
            self.cycle_index += 1
            self.iter_index = 0

    def upd_lr(self, engine):
        if self.cycle_index == 0:
            lr = self.min_lr + (self.max_lr - self.min_lr) * self.iter_index/(self.params.clr.warmup_epochs*self.iterations_per_epoch)
        else:
            cycle_progress = self.iter_index / (self.params.clr.period_epochs*self.iterations_per_epoch)
            lr = self.max_lr + ((self.min_lr - self.max_lr) / 2) * (1 - math.cos(math.pi * cycle_progress))
        self.optimizer.param_groups[0]['lr'] = lr
        engine.state.metrics['lr'] = self.optimizer.param_groups[0]['lr']
        self.iter_index += 1




def create_supervised_trainer(model, optimizer, loss_fn, metrics={},
                              device=None, non_blocking=False,
                              prepare_batch=ignite.engine._prepare_batch):
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
    engine = ignite.engine.create_supervised_trainer(model, optimizer, loss_fn, device, non_blocking, prepare_batch,
                                                     output_transform = lambda x, y, y_pred, loss: (y_pred, y,))

    @engine.on(Events.ITERATION_STARTED)
    def reset_output(engine):
        engine.state.output = None

    for name, metric in metrics.items():
        metric.attach(engine, 'train:' + name)

    return engine
