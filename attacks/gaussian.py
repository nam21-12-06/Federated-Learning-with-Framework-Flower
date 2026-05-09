from attacks.base import Attack
import numpy as np


class GaussianAttack(Attack):

    def __init__(self, mean=0.0, std=1.0):
        """
        Gaussian Noise Attack

        w_attack = w + N(mean, std^2)

        Args:
            mean: Mean of Gaussian noise
            std: Standard deviation of Gaussian noise
        """
        super().__init__()

        self.mean = mean
        self.std = std

    def apply(self, parameters, global_parameters=None):

        noisy_parameters = []

        for p in parameters:

            noise = np.random.normal(
                loc=self.mean,
                scale=self.std,
                size=p.shape
            ).astype(p.dtype)

            noisy_p = p + noise
            noisy_parameters.append(noisy_p)

        return noisy_parameters