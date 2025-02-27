import math
from torch.nn.init import _calculate_fan_in_and_fan_out, _no_grad_normal_

def xavier_normal(tensor, gain=1., multiplier=0.1):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    return _no_grad_normal_(tensor, 0., std * multiplier)