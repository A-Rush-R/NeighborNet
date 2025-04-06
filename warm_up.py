import math

def warmup_decay_cosine(warmup_steps, loop_steps):
    '''
    warmup_steps不被包含在第一个loop中
    loop_steps: 每个cosine周期的step数
    '''

    def fn(step):
        assert(loop_steps > 0 )
        if step < warmup_steps:
            return float(step) / float(max(1e-4, warmup_steps))
        step = step - warmup_steps
        rate = (step // loop_steps + 1)
        step = step % loop_steps
        progress = float(step) / float(max(1, loop_steps))
        lr = 0.5 * (1.0 + math.cos(math.pi * progress)) / rate
        return min(lr, 1)
    return fn

# import math

# def warmup_decay_cosine(warmup_steps, loop_steps):
#     """
#     Warmup and cosine decay scheduler.

#     Parameters:
#     warmup_steps: int
#         The number of steps for the warmup phase.
#     loop_steps: int
#         The number of steps in each cosine decay cycle.

#     Returns:
#     fn: callable
#         A function that computes the learning rate multiplier given the current step.
#     """

#     if warmup_steps < 0 or loop_steps <= 0:
#         raise ValueError("warmup_steps must be >= 0 and loop_steps must be > 0")

#     def fn(step):
#         if step < warmup_steps:
#             # Linear warmup phase
#             return float(step) / float(max(1, warmup_steps))

#         # Cosine decay phase
#         step -= warmup_steps
#         rate = (step // loop_steps + 1)  # Determine decay cycle
#         step %= loop_steps  # Progress within the current cycle
#         progress = float(step) / float(loop_steps)
#         lr = 0.5 * (1.0 + math.cos(math.pi * progress)) / rate
#         return min(lr, 1.0)

#     return fn

import math

def warmup_decay_cosine_tata(iter_num, total_steps, warmup_steps=1000):
    """
    Creates a warm-up followed by a cosine decay schedule.

    Args:
        iter_num (int): Current iteration number.
        total_steps (int): Total number of iterations (epochs * iterations_per_epoch).
        warmup_steps (int): Number of warm-up steps. Default is 1000.

    Returns:
        function: A lambda function that calculates the learning rate multiplier.
    """
    def fn(step):
        # Ensure warmup_steps and total_steps are valid
        if warmup_steps >= total_steps:
            raise ValueError("warmup_steps must be less than total_steps")

        # Compute warm-up and decay
        if step < warmup_steps:
            # Linear warm-up
            return step / warmup_steps
        else:
            # Cosine decay
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))

    return fn
