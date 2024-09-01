import numpy as np
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_nb
import random

from typing import List, Tuple, Dict

"""
Noisy Susceptible Infected Model as implemented in the paper. 

-- Sagar Kumar, 2024
"""

""" Helper Functions """


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

    return beta * k * (1 - sum(infected)) * P.T @ infected


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


""" Classes """


class ModelOutput:

    def __init__(self,
                 infected: List[np.ndarray]) -> None:
        entropy: List[float] = list()

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
        Creates an instance of the Discrete Noisy Susceptible Infected Model
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
            inf: List[np.ndarray] = [seed]

        else:
            inf: List[np.ndarray] = [seedI]

        for t in range(1, self.T):
            infected = inf[t - 1]

            inf.append(inf[t - 1] + di(infected=infected, beta=beta, k=k, P=self.P))

        out = ModelOutput(infected=inf)

        return out

    def homogeneous_simulation(self,
                               beta: float,
                               k: int,
                               filepath: str = None,
                               seedI: np.ndarray = None,
                               density: bool = True,
                               track_comm: bool = False,
                               pbar_on: bool = True,
                               notebook: bool = False) -> None or ModelOutput or Tuple[ModelOutput, dict]:

        """
        Runs the homogeneous population simulation of the NSI model as described in the paper.

        Note: -1 is the "susceptible" state, k must be in simulations

        :param beta: transmissibility
        :param k: average degree
        :param filepath: File to store simulation results. If none provided, no file is written. (Default = None)
        :param seedI: seed distribution (default 1 in message 0).
        :param density: Output infected vectors as fractions of the population
        :param track_comm: track successful communications
        :param pbar_on: turn on progress bar
        :param notebook: if pbar_on is True, indicates whether to use the notebook implementation of tqdm
        :return: infected vectors at each time step
        """

        # mutable dictionary of the state/message received by each node
        population_dictionary: Dict[int, int] = {n: -1 for n in range(self.N)}

        if seedI is None:
            seed: np.ndarray = np.zeros(self.P.shape[0])
            seed[0] = 1  # seed one node with message 0

        else:
            seed: np.ndarray = seedI

        inf: List[np.ndarray] = [seed]

        # initialize list to successful communications
        comms = {n: [] for n in range(self.N)}

        # setting seed(s)
        for d, seed_count in enumerate(seed):
            for _ in range(int(seed_count)):
                node = random.randint(0, self.N - 1)
                population_dictionary[node] = d
                comms[node].append((node, d))

        # progress bar
        if pbar_on:
            if notebook:
                pbar = tqdm_nb(range(1, self.T))
            else:
                pbar = tqdm(range(1, self.T))
        else:
            pbar = range(self.T)

        # running simulation
        for _ in pbar:

            # initialized array
            it = np.array([0] * self.P.shape[0])

            # infected nodes
            valid_nodes = np.array([node for node in population_dictionary.keys() if population_dictionary[node] != -1])

            # shuffle valid nodes
            np.random.shuffle(valid_nodes)

            # select k random neighbors for each node
            valid_neighbors = np.random.randint(self.N, size=(self.N, k))

            # shifting by one to allow boolean mask
            valid_neighbors = valid_neighbors + np.ones(valid_neighbors.shape, dtype=np.uint8)

            # defining connection probabilities
            valid_r = np.random.rand(self.N, k)

            # successful communication, shifted back down so that missed communication is flagged by value of -1
            valid_neighbors = ((valid_r < beta) * valid_neighbors) - np.ones(valid_neighbors.shape, dtype=np.uint8)

            for n, neighbors in zip(valid_nodes, valid_neighbors):
                n_state = population_dictionary[n]

                for ni in neighbors:

                    if not (ni < 0):
                        ni_state = population_dictionary[ni]

                        if ni_state < 0:
                            new_state = error_message(n_state, self.P)
                            population_dictionary[ni] = new_state

                            comms[ni] = comms[n].copy()
                            comms[ni].append((ni, new_state))

                        else:
                            pass
                    else:
                        pass

            # bincount doesn't like negative values, so we shift everything up by one
            shifted_states = np.array(list(population_dictionary.values())) + np.ones(len(population_dictionary),
                                                                                      dtype=np.uint8)
            cts = np.bincount(shifted_states, minlength=self.P.shape[0] + 1)

            for idx, x in enumerate(cts[0:]):
                it[idx - 1] = x  # ignoring 0 (which used to be -1), shifting back down, and assigning count

            inf.append(it)

        if density:
            inf = [i / self.N for i in inf]

        out = ModelOutput(infected=inf)

        if filepath is not None:
            np.savez_compressed(filepath + ".npz", inf)

        if track_comm:
            return out, comms

        else:
            return out


""" Class-Dependent Functions """


def load_sim(filepath) -> ModelOutput:
    """
    Loads compressed numpy array output from NSI.homogeneous_simulation() into a ModelOutput class.

    :param filepath: Compressed numpy file to read with infection densities.
    :return: ModelOutput object with standard attributes.
    """

    inf = np.load(filepath)

    return ModelOutput(infected=inf)



""" Class-Independent Functions"""

def mutual_info(list_of_arrays, px) -> float:
    
    """
    Calculates the mutual information for a given model output and input source distribution.
    
    :param list_of_arrays: Model output arrays of dimension the size of the alphabet. 
    :param px: Probability distribution over source states.
    
    :returns: Mutual Information 
    """
    
    # Ensure the inputs are normalized
    assert np.isclose(np.sum(X), 1.0), "X should be normalized"
    assert np.all(np.isclose(np.sum(M, axis=1), 1.0)), "Rows of M should sum to 1"

    num_x = M.shape[0]
    num_y = M.shape[1]

    # Calculate p(y)
    p_y = np.dot(X, M)

    # Calculate p(x, y)
    p_xy = np.zeros((num_x, num_y))
    for x in range(num_x):
        for y in range(num_y):
            p_xy[x, y] = X[x] * M[x, y]

    def log_term(x, y):
        if p_xy[x, y] > 0 and p_y[y] > 0:  # Ensure we don't divide by zero or take log of zero
            log_value = np.log2(p_xy[x, y] / (X[x] * p_y[y]))
            return log_value
        return 0

    I = np.sum([p_xy[x, y] * log_term(x, y) for x in range(num_x) for y in range(num_y)])
    
    return I
