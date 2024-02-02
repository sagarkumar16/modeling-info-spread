import numpy as np


class ModelOutput:

    def __init__(self,
                 infected: list[np.ndarray]) -> None:

        entropy: list[float] = list()

        for state in infected:
            d = state / sum(state)
            p = d[d > 0]
            h = -1 * np.sum(p * np.log2(p))
            entropy.append(h)

        self.H = entropy
        self.I = infected
