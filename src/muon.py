import torch

class MUON(torch.optim.Optimizer):
    def __init__(self, params, momentum_weight, learning_rate):
        self.momentum_weight = momentum_weight
        self.learning_rate = learning_rate
        super().__init__(params, dict(lr=learning_rate, momentum_weight=momentum_weight))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum_weight = group["momentum_weight"]
            for p in group["params"]:
                if p.grad is None: continue
                # --- MUON algorithm --- 
                self.algorithm(p=p, learning_rate=lr, momentum_weight=momentum_weight)

    def algorithm(self, p, momentum_weight, learning_rate):
        # --- Store momentum if not done yet for the parameter ---
        if len(self.state[p]) == 0: 
            self.state[p]["momentum_buffer"] = torch.zeros_like(p)
        
        # --- Update Momentum: m = weight * m + gradient ---
        self.state[p]["momentum_buffer"].mul_(momentum_weight).add_(p.grad)
        
        # --- Orthogonalize Momentum ---
        momentum = self.state[p]["momentum_buffer"]
        if momentum.ndim == 4:  # Special case handling of conv layer.
            momentum = momentum.reshape(len(momentum), -1)
        momentum /= (momentum.norm() + 1e-7) 
        p.data.mul_(len(p.data)**0.5 / p.data.norm()) # Helped with exploading gradients
        orthogonalized = self.newton_shultz(momentum).reshape(p.shape)
        
        # --- Update parameters: p = p - lr * p_orthoganlized ---
        p.add_(orthogonalized, alpha=-learning_rate)
    
    def newton_shultz(self, m, num_iters=5):
        """Batched Implementation - Iteratively apply: f(x) = ax + bx^3 +cx^5."""
        assert m.ndim == 2, f"dimension {m.ndim}"
        a, b, c = 3.4445, -4.7770, 2.0315
        
        transpose = m.shape[0] > m.shape[1]
        if transpose: m = m.T

        for i in range(num_iters):
            m = a * m \
                + b * (m @ m.T @ m) \
                + c * (m @ m.T @ m @ m.T @ m)
        
        if transpose: m = m.T
        return m        
