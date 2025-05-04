import torch

from torch import nn
from itertools import chain
    
class SkillObjectPipeline(nn.Module):
    def __init__(self, obs_key, option_key, object_key,
                 slot_extractor, slot_pooler, 
                 downstream_model, downstream_input_keys):
        super().__init__()
        self.obs_key, self.option_key, self.object_key = obs_key, option_key, object_key
        assert 'obs' in downstream_input_keys, 'Expected that obs will be always fed'
        self.downstream_input_keys = downstream_input_keys
        self.slot_extractor, self.slot_pooler, self.downstream_model = slot_extractor, slot_pooler, downstream_model

    def forward(self, inputs):
        obs = inputs[self.obs_key]
        slots = self.slot_extractor(obs)
        feature_vector = self.slot_pooler(slots, inputs[self.object_key])
        
        downstream_input = feature_vector
        for key in self.downstream_input_keys:
            if key == 'obs':
                continue
            downstream_input = torch.cat([downstream_input, inputs[key]], dim = -1)
        return self.downstream_model(downstream_input)
    
    def forward_mode(self, inputs):
        obs = inputs[self.obs_key]
        slots = self.slot_extractor(obs)
        feature_vector = self.slot_pooler(slots, inputs[self.object_key])
        
        downstream_input = feature_vector
        for key in self.downstream_input_keys:
            if key == 'obs':
                continue
            downstream_input = torch.cat([downstream_input, inputs[key]], dim = -1)
        return self.downstream_model.forward_mode(downstream_input)
    
    def get_policy_parameters_without_std(self):
        params = chain.from_iterable([self.slot_extractor.parameters(), self.slot_pooler.parameters(),
                                      self.downstream_model.get_parameters_without_std()])
        return params

    def get_policy_std_parameters(self):
        return self.downstream_model.get_std_parameters()
    
class DictPipeline(nn.Module):
    def __init__(self, downstream_model):
        super().__init__()
        self.downstream_model = downstream_model
    
    def forward(self, inputs):
        return self.downstream_model(inputs['obs'])
    
    def forward_mode(self, inputs):
        return self.downstream_model.forward_mode(inputs['obs'])
    
    def get_policy_parameters_without_std(self):
        return self.downstream_model.get_parameters_without_std()

    def get_policy_std_parameters(self):
        return self.downstream_model.get_std_parameters()
