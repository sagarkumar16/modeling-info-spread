import numpy as np
#from .channel import Channel
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_nb
import random
from collections import Counter


"""
Noisy Susceptible Infected Model as implemented in the paper. 

-- Sagar Kumar, 2024
"""

#### Functions ####

def di(infected: np.ndarray,
        beta: float,
        k: float,
        P: np.ndarray
       ) -> np.ndarray:

    """
    dI/dt for a vector of infected population I.

    :param infected: infected population vector at time t (dim = # message strains)
    :param beta: transmissibility
    :param k: average degree
    :param P: communication channel
    :return: Infection vector at time t + 1
    """

    return beta * k * (1-sum(infected)) * P@infected


def error_message(m: int,
                  P: np.ndarray) -> int:

    """
    Replicates communcation over a discrete noisy channel by taking in the index for a messsage strain \
    and outputs a new one based on the channel.
    :param m: input message index
    :param P: communcation channel
    :return: random message to forward
    """
    m_out = np.random.choice(range(P.shape[0]), p=P[m])

    return m_out


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


class NSI:

    def __init__(self,
                 N: int,
                 T: int,
                 P: np.ndarray,
                 ) -> None:

        """
        Creates an instance of the Noisy Susceptible Infected Model
        :param N: populations size
        :param T: time to run model
        :param P: communication channel
        """

        self.P = P
        self.N = N
        self.T = T

    def homogeneous_analytic(self,
                             beta: float,
                             k: float,
                             seedI: np.ndarray = None
                             ) -> ModelOutput:

        """
        Runs the homogenous linear approximation of the NSI model as described in the paper.

        :param beta: transmissibility
        :param k: average degree
        :param seedI: seed distribution (default 1/N in message 0). To make output a vector of natural numbers, start \
        with 1 instead of 1/N.
        :return: infected at each time step and entropy at each time step
        """

        if seedI is None:
            seed = np.zeros(self.P.shape[0])
            seed[0] = 1 / self.N  # seed one node with message 0
            inf: list[np.ndarray] = [seed]

        else:
            inf: list[np.ndarray] = [seedI]

        entropy: list[float] = list()

        for t in range(1, self.T):
            infected = inf[t - 1]

            inf.append(inf[t - 1] + di(infected=infected, beta=beta, k=k, P=self.P))

        out = ModelOutput(infected=inf)

        return out

    def homogeneous_simulations(self,
                                beta: float,
                                k: int,
                                seedI: np.ndarray = None,
                                density = True,
                                pbar_on:bool = True,
                                notebook:bool = False) -> ModelOutput:

        """
        Runs the homogeneous population simulation of the NSI model as described in the paper.

        Note: -1 is the "susceptible" state, k must be in simulations

        :param beta: transmissibility
        :param k: average degree
        :param seedI: seed distribution (default 1 in message 0).
        :param density: Output infected vectors as fractions of the population
        :param pbar_on: turn on progress bar
        :param notebook: if pbar_on is True, indicates whether to use the notebook implementation of tqdm
        :return: infected vectors at each time step
        """

        # tracks the state/message received by each node
        population_dictionary: dict[int, int] = {n: -1 for n in range(self.N)}

        if seedI is None:
            seed: np.ndarray = np.zeros(self.P.shape[0])
            seed[0] = 1  # seed one node with message 0

        else:
            seed: np.ndarray = seedI

        inf: list[np.ndarray] = [seed]

        # setting seed(s)
        for d, seed_count in enumerate(seed):
            for _ in seed_count:
                node = random.randint(0, self.N - 1)
                population_dictionary[node] = d

        # progress bar
        if pbar_on:
            if notebook:
                pbar = tqdm(range(self.T))
            else:
                pbar = tqdm_nb(range(self.T))
        else:
            pbar = range(self.T)

        # running simulation
        for t in pbar:

            it = np.array([0]*self.P.shape[0])

            # time step updates
            valid_nodes = [node for node in population_dictionary.keys() if population_dictionary[node] != -1]

            while len(valid_nodes) > 0:

                n = random.choice(valid_nodes)
                valid_nodes.remove(n)

                n_state = population_dictionary[n]

                neighbors = np.random.randint(self.N, size=k)

                for ni in neighbors:

                    if random.random() < beta:

                        ni_state = population_dictionary[ni]

                        if ni_state < 0:
                            population_dictionary[ni] = error_message(n_state, e)

                        else:
                            pass

                    else:
                        pass

            cts = Counter(population_dictionary.values())

            for x in cts.keys():
                it[x] = cts[x]

            inf.append(it)

        out = ModelOutput(infected=inf)

        return out





