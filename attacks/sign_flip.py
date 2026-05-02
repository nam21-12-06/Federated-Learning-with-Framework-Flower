from attacks.base import Attack

class SignFlipAttack(Attack):
    def apply(self, parameters, global_parameters=None):
        return [-p for p in parameters]