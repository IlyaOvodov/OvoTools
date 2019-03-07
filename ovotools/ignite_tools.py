import copy
import torch
from ignite.engine import Events


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
                engine.state.metrics[key+'.'+k] = v
        self.best_model_buffer.save_if_best(engine)
        if event == Events.ITERATION_COMPLETED:
            str = "Epoch:{}.{}\t".format(engine.state.epoch, engine.state.iteration)
        else:
            str = "Epoch:{}\t".format(engine.state.epoch)
        str += '\t'.join(['{}:{:.3f}'.format(k,v) for k,v in engine.state.metrics.items()])
        print(str)
        with open(self.params.get_base_filename() + '.log', 'a') as f:
            f.write(str + '\n')
