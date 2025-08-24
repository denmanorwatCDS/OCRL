import math

def linear_warmup(step: int, warmup_steps: int) -> float:
    """Linear warmup.

    Function maps current step or epoch to factor of learning rate schedules.
    """
    if warmup_steps > 0:
        return min(1.0, step / warmup_steps)
    else:
        return 1.0

def cosine_decay_with_warmup(
    step: int,
    warmup_steps: int,
    decay_steps: int,
) -> float:
    """Cosine decay to zero with linear learning rate warmup.

    Function maps current step or epoch to factor of learning rate schedules.
    """
    if step < warmup_steps:
        return linear_warmup(step, warmup_steps)
    else:
        step = step - warmup_steps
        decay_steps = decay_steps - warmup_steps
        step = min(step, decay_steps)
        return 0.5 * (1 + math.cos(math.pi * (step / decay_steps)))
    
def exponential_decay_with_warmup(
    step: int,
    warmup_steps: int,
    decay_rate: int,
    decay_steps: int,
) -> float:
    """Exponential decay with linear learning rate warmup.

    Function maps current step or epoch to factor of learning rate schedules.
    """
    if step < warmup_steps:
        return linear_warmup(step, warmup_steps)
    else:
        step = step - warmup_steps
        return decay_rate ** (step/decay_steps)
    
def constant_lr(step):
    return 1.