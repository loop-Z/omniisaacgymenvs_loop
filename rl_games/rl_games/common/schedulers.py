import math

class RLScheduler:
    def __init__(self):
        pass

    def update(self,current_lr, entropy_coef, epoch, frames, **kwargs):
        pass

class IdentityScheduler(RLScheduler):
    def __init__(self):
        super().__init__()

     
    def update(self, current_lr, entropy_coef, epoch, frames, kl_dist, **kwargs):
        return current_lr, entropy_coef  


class AdaptiveScheduler(RLScheduler):
    def __init__(self, kl_threshold = 0.008):
        super().__init__()
        self.min_lr = 1e-6
        self.max_lr = 1e-2
        self.kl_threshold = kl_threshold

    def update(self, current_lr, entropy_coef, epoch, frames, kl_dist, **kwargs):
        lr = current_lr
        if kl_dist > (2.0 * self.kl_threshold):
            lr = max(current_lr / 1.5, self.min_lr)
        if kl_dist < (0.5 * self.kl_threshold):
            lr = min(current_lr * 1.5, self.max_lr)
        return lr, entropy_coef         

class TransformerScheduler(RLScheduler):
    def __init__(self, min_lr=0, base_lr=3e-4, warmup_steps=10, warmup_factor=0.1, max_iter=1000, **kwargs):
        super().__init__()
        self.warmup_steps = warmup_steps
        self.warmup_factor = warmup_factor
        self.min_lr = min_lr
        self.base_lr = base_lr
        self.max_iter = max_iter

    def update(self, current_lr, entropy_coef, epoch, frames, kl_dist, **kwargs):
        lr = 0.5 * (1.0 + math.cos(math.pi * epoch / self.max_iter))
        lr = (1.0  - self.min_lr) * lr + self.min_lr
        lr *= self.base_lr
        if epoch < self.warmup_steps:
            alpha = epoch / self.warmup_steps
            warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            lr *= warmup_factor
        return lr, entropy_coef

class LinearScheduler(RLScheduler):
    def __init__(self, start_lr, min_lr=1e-6, max_steps=1000000, use_epochs=True, apply_to_entropy=False, **kwargs):
        super().__init__()

        self.start_lr = start_lr
        self.min_lr = min_lr
        self.max_steps = max_steps
        self.use_epochs = use_epochs
        self.apply_to_entropy = apply_to_entropy
        if apply_to_entropy:
            self.start_entropy_coef = kwargs.pop('start_entropy_coef', 0.01)
            self.min_entropy_coef = kwargs.pop('min_entropy_coef', 0.0001)

    def update(self, current_lr, entropy_coef, epoch, frames, kl_dist, **kwargs):
        if self.use_epochs:
            steps = epoch
        else:
            steps = frames
        mul = max(0, self.max_steps - steps)/self.max_steps 
        lr = self.min_lr + (self.start_lr - self.min_lr) * mul
        if self.apply_to_entropy:
            entropy_coef = self.min_entropy_coef + (self.start_entropy_coef - self.min_entropy_coef) * mul

        return lr, entropy_coef     