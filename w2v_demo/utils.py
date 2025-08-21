import torch


class NegativeSampler:
    def __init__(self, id_counts, K, power=0.75, device="cpu"):
        probs = id_counts**power
        probs /= probs.sum()
        self.probs = probs.to(device)
        self.K = K

    def sample(self, batch_size, K):
        """
        Returns tensor of shape [batch_size, K]
        """
        return torch.multinomial(self.probs, batch_size * K, replacement=True).view(
            batch_size, K
        )
