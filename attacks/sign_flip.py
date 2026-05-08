from attacks.base import Attack

class SignFlipAttack(Attack):
    def __init__(self, scale=-1.0):
        """
        Initialize the attack with a scale factor.
        scale: float, default is -1.0 for a standard sign-flip attack.
        """
        super().__init__()
        self.scale = scale

    def apply(self, parameters, global_parameters=None):
        """
        Apply the sign flip by multiplying parameters by the scale factor.
        parameters: list[np.ndarray]
        return: manipulated parameters
        """
        # Multiply each layer's weights/gradients by the scale
        return [p * self.scale for p in parameters]