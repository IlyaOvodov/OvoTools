import copy
import torch
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
    def __init__(self, model, metric_name, params, minimize = True, save_to_file = True):
        self.model = model
        assert metric_name
        self.metric_name = metric_name
        assert minimize == True, "Not implemented"
        self.save_to_file = save_to_file
        self.params = params
        self.reset()

    def reset(self):
        self.best_dict = None
        self.best_score = None
        self.best_epoch = None

    def save_if_best(self, engine):
        assert self.metric_name in engine.state.metrics.keys(), "{} {}".format(self.metric_name, engine.state.metrics.keys())
        if self.best_score is None or self.best_score > engine.state.metrics[self.metric_name]:
            self.best_score = engine.state.metrics[self.metric_name]
            self.best_dict  = copy.deepcopy(self.model.state_dict())
            self.best_epoch = engine.state.epoch
            print('model for {}={} dumped'.format(self.metric_name, self.best_score))
            if self.save_to_file:
                torch.save(self.best_dict, self.params.get_base_filename() + '.t7')

    def restore(self, model = None):
        assert self.best_dict is not None
        if model is None:
            model = self.model
        print('model for {}={} on epoch {} restored'.format(self.metric_name, self.best_score, self.best_epoch))
        model.load_state_dict(self.best_dict)


class LogTrainingResults:
    def __init__(self, evaluator, loaders_dict, best_model_buffer, params):
        self.evaluator = evaluator
        self.loaders_dict = loaders_dict
        self.best_model_buffer = best_model_buffer
        self.params = params

    def __call__(self, engine, event):
        if event == Events.ITERATION_COMPLETED and engine.state.epoch != 1:
            return
        for key,loader in self.loaders_dict.items():
            self.evaluator.run(loader)
            for k,v in self.evaluator.state.metrics.items():
                engine.state.metrics[key+':'+k] = v
        self.best_model_buffer.save_if_best(engine)
        if event == Events.ITERATION_COMPLETED:
            str = "Epoch:{}.{}\t".format(engine.state.epoch, engine.state.iteration)
        else:
            str = "Epoch:{}\t".format(engine.state.epoch)
        str += '\t'.join(['{}:{:.5f}'.format(k,v) for k,v in engine.state.metrics.items()])
        print(str)
        with open(self.params.get_base_filename() + '.log', 'a') as f:
            f.write(str + '\n')


class TensorBoardLogger:
    SERIES_PLOT_SEPARATOR = ':'
    GROUP_PLOT_SEPARATOR = '.'

    def __init__(self, trainer_engine, params, count_iters=False, period=1):
        log_dir = params.get_base_filename()
        self.writer = tensorboardX.SummaryWriter(log_dir=log_dir, flush_secs = 10)
        event = Events.ITERATION_COMPLETED if count_iters else Events.EPOCH_COMPLETED
        trainer_engine.add_event_handler(event, self.on_event)
        self.period = period
        self.call_count = 0
        trainer_engine.add_event_handler(Events.COMPLETED, self.on_completed)

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
