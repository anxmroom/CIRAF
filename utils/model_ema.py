import torch
import math
import copy


class ModelEMA(object):
    def __init__(self,
                 model,
                 decay=0.9998
                 ):
        self.step = 0
        self.model = model
        self.decay = decay
        self.state_dict = dict()
        self.snapshot = dict()
        for k, v in model.state_dict().items():
            self.state_dict[k] = torch.zeros_like(v)

    def update(self, model=None):
        #decay = self.decay * (1 - math.exp(-(self.step + 1) / 6000))
        decay = min(self.decay, (1 + self.step) / (10 + self.step))
        model_dict = model.state_dict()
        for k, v in self.state_dict.items():
            # v = decay * v + (1 - decay) * model_dict[k]
            v = (1 - decay) * v + decay * model_dict[k]
            v.stop_gradient = True
            self.state_dict[k] = v
        self.step += 1

    def apply(self):
        self.snapshot = copy.deepcopy(self.model.state_dict())
        self.model.load_state_dict(self.state_dict)

    def restore(self):
        self.model.load_state_dict(self.snapshot)
