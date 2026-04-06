import random
import numpy as np
import numpy.typing as npt
from typing import Tuple, Callable
from tqdm import tqdm
import networkx as nx

from .continuous_models import (
    continuous_time_random_walk,
    ODE,
    simulate_communication,
    mutual_info,
    bootstrap_mi_ci,
    simulate_mi,
    run_MI,
    objective,
    estimate_capacity,
    final_dist,
    estimate_spreading_capacity,
)


class NetworkGillespie:

    def __init__(self,
                 G: nx.Graph,
                 initial_infected: dict,
                 beta: float,
                 channel: np.ndarray):
        """
        :param G: networkx.Graph where nodes are individuals
        :param initial_infected: dict of {node: strain} initially infected nodes
        :param beta: base transmission rate per edge
        :param channel: noisy channel matrix (mutational probabilities)
        """
        self.G = G
        self.N = G.number_of_nodes()
        self.beta = beta
        self.channel = channel
        self.k = channel.shape[0]

        self.state = -np.ones(self.N, dtype=int)  # -1 = susceptible, otherwise strain index
        for node, strain in initial_infected.items():
            self.state[node] = strain

        self.time = 0.0
        self.t = [self.time]
        self.strain_counts = [np.bincount(self.state[self.state >= 0], minlength=self.k)]

    def get_possible_events(self):
        """Return all currently infectious edges with possible transmission"""
        events = []
        for u in range(self.N):
            if self.state[u] >= 0:  # u is infected
                for v in self.G.neighbors(u):
                    if self.state[v] == -1:  # v is susceptible
                        events.append((u, v))
        return events

    def step(self) -> bool:
        events = self.get_possible_events()
        if not events:
            return False

        rate = self.beta * len(events)
        tau = np.random.exponential(1. / rate)
        self.time += tau

        # Pick an event
        idx = np.random.randint(len(events))
        u, v = events[idx]

        # Mutate strain from u to v according to channel matrix
        source_strain = self.state[u]
        new_strain = np.random.choice(self.k, p=self.channel[source_strain])
        self.state[v] = new_strain

        # Record state
        self.t.append(self.time)
        strain_count = np.bincount(self.state[self.state >= 0], minlength=self.k)
        self.strain_counts.append(strain_count)
        return True

    def simulate(self,
                 max_time: float = 100.0,
                 max_steps: int = 10_000,
                 density: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        steps = 0
        while self.time < max_time and steps < max_steps:
            if not self.step():
                break
            steps += 1

        times = np.array(self.t)
        counts = np.array(self.strain_counts)
        if density:
            return times, counts / self.N
        else:
            return times, counts


class TwoStrainGillespie:

    S, I1, I2 = -1, 0, 1

    def __init__(self, G, Q, beta, n=None):
        """
        G : networkx Graph
        Q : 2x2 array-like with entries [[r11, r12],[r21, r22]]
        beta : transmissibility
        n : number of nodes (defaults to G.number_of_nodes())
        """
        self.G = G
        self.Q = np.asarray(Q, dtype=float)
        self.beta = float(beta)
        self.n = int(n if n is not None else G.number_of_nodes())
        assert self.Q.shape == (2, 2), "Q must be 2x2"

    def _infected_neighbor_counts(self, state_array, i):
        n1 = n2 = 0
        for j in self.G.neighbors(i):
            if state_array[j] == self.I1:
                n1 += 1
            elif state_array[j] == self.I2:
                n2 += 1
        return n1, n2

    def _compute_node_rates(self, state_array, i):
        # Zero if not susceptible
        if state_array[i] != self.S:
            return 0.0, 0.0
        n1, n2 = self._infected_neighbor_counts(state_array, i)
        r11, r12 = self.Q[0, 0], self.Q[0, 1]
        r21, r22 = self.Q[1, 0], self.Q[1, 1]
        return (
            self.beta * (r11 * n1 + r12 * n2),   # rate -> I1
            self.beta * (r21 * n1 + r22 * n2),   # rate -> I2
        )

    def _refresh_rates_for_node_and_neighbors(self, state_array, rate_matrix, i):
        # update node i
        r1, r2 = self._compute_node_rates(state_array, i)
        rate_matrix[0, i] = r1
        rate_matrix[1, i] = r2
        # update its neighbors
        for j in self.G.neighbors(i):
            r1, r2 = self._compute_node_rates(state_array, j)
            rate_matrix[0, j] = r1
            rate_matrix[1, j] = r2

    def _choose_event(self, rate_matrix, rng=np.random):
        rflat = rate_matrix.reshape(-1)
        T = rflat.sum()
        if T <= 0.0:
            return np.inf, None, None
        tau = rng.exponential(1.0 / T)
        u = rng.random() * T
        idx = np.searchsorted(np.cumsum(rflat), u, side="right")
        N = rate_matrix.shape[1]
        strain = idx // N      # 0 for I1, 1 for I2
        node = idx % N
        return tau, strain, node

    @staticmethod
    def _counts(state_array, S, I1, I2):
        return (
            np.sum(state_array == S),
            np.sum(state_array == I1),
            np.sum(state_array == I2),
        )

    @staticmethod
    def _resample_step_series(t_vec, y_vec, t_grid):
        idx = np.searchsorted(t_vec, t_grid, side="right") - 1
        idx = np.clip(idx, 0, len(y_vec) - 1)
        return y_vec[idx]

    def simulate_once(
        self,
        t_max: float = np.inf,
        seed_node: int = None,
        initial_strain: str = "I1",
        rng: Callable = np.random,
        return_events: bool = False,
    ):
        """
        Run one Gillespie simulation.

        :param t_max: max time
        :param seed_node:
        :param t_grid: discretized common time array
        :param initial_strain: Must be "I1" or "I2"
        :param rng: function to generate random number
        :param return_events: whether to return list of infection events

        Returns dict with arrays: t, S, I1, I2 and (optional) events.
        """
        # state & rates
        state_array = np.full(self.n, self.S, dtype=int)
        rate_matrix = np.zeros((2, self.n), dtype=float)

        # choose seed
        if seed_node is None:
            seed_node = random.randint(0, self.n - 1)
        seed_strain = self.I1 if str(initial_strain).upper() == "I1" else self.I2
        state_array[seed_node] = seed_strain
        rate_matrix[:, seed_node] = 0.0

        # initialize all rates
        for i in range(self.n):
            r1, r2 = self._compute_node_rates(state_array, i)
            rate_matrix[0, i] = r1
            rate_matrix[1, i] = r2

        # time series containers
        t = 0.0
        t_vec = [0.0]
        S0, I10, I20 = self._counts(state_array, self.S, self.I1, self.I2)
        S_vec = [S0]; I1_vec = [I10]; I2_vec = [I20]

        # optional event log
        if return_events:
            events = []  # list of (t, strain, node)

        # main loop
        while True:
            tau, strain, node = self._choose_event(rate_matrix, rng=rng)
            if not np.isfinite(tau) or t + tau > t_max or strain is None:
                break
            t += tau

            # execute infection if node is susceptible
            if state_array[node] == self.S:
                state_array[node] = self.I1 if strain == 0 else self.I2
                rate_matrix[:, node] = 0.0
                self._refresh_rates_for_node_and_neighbors(state_array, rate_matrix, node)

            # record snapshot (right-continuous step)
            t_vec.append(t)
            S_t, I1_t, I2_t = self._counts(state_array, self.S, self.I1, self.I2)
            S_vec.append(S_t); I1_vec.append(I1_t); I2_vec.append(I2_t)

            if return_events:
                events.append((t, int(strain), int(node)))

        out = {
            "t": np.array(t_vec),
            "S": np.array(S_vec),
            "I1": np.array(I1_vec),
            "I2": np.array(I2_vec),
        }
        if return_events:
            out["events"] = events
        return out

    def run_many(
        self,
        K: int,
        t_max: float = np.inf,
        t_grid: np.ndarray = None,
        initial_strain: str = "I1",
        rng_seed: int = None,
    ) -> dict:
        """
        Run K simulations, resample curves onto a common time grid, and return means and 95% CIs for I1, I2, and total infected.

        :param K: num simulations
        :param t_max: max time
        :param t_grid: discretized common time array
        :param initial_strain: Must be "I1" or "I2"
        :rng_seed: random seed

        :returns: a dict containing
          t_grid,
          I1_mean, I1_lo, I1_hi,
          I2_mean, I2_lo, I2_hi,
          It_mean, It_lo, It_hi
        """
        # optional seeding for reproducibility
        if rng_seed is not None:
            np.random.seed(rng_seed)
            random.seed(rng_seed)

        # First run (to define grid if not provided)
        first = self.simulate_once(t_max=t_max, initial_strain=initial_strain)
        if t_grid is None:
            t_stop = first["t"][-1] if len(first["t"]) > 0 else 0.0
            # keep at least 2 points if trivial run
            t_grid = np.linspace(0.0, max(t_stop, 1e-12), 200)

        # allocate storage
        I1_runs = []
        I2_runs = []
        It_runs = []

        # resample first run
        I1_runs.append(self._resample_step_series(first["t"], first["I1"], t_grid))
        I2_runs.append(self._resample_step_series(first["t"], first["I2"], t_grid))
        It_runs.append(self._resample_step_series(first["t"], first["I1"] + first["I2"], t_grid))

        # remaining runs
        for _ in tqdm(range(K - 1), desc='Simulating'):
            sim = self.simulate_once(t_max=t_max, initial_strain=initial_strain)
            I1_runs.append(self._resample_step_series(sim["t"], sim["I1"], t_grid))
            I2_runs.append(self._resample_step_series(sim["t"], sim["I2"], t_grid))
            It_runs.append(self._resample_step_series(sim["t"], sim["I1"] + sim["I2"], t_grid))

        I1_runs = np.vstack(I1_runs)
        I2_runs = np.vstack(I2_runs)
        It_runs = np.vstack(It_runs)

        def summarize(mat):
            mean = mat.mean(axis=0)
            lo = np.percentile(mat, 2.5, axis=0)
            hi = np.percentile(mat, 97.5, axis=0)
            return mean, lo, hi

        I1_mean, I1_lo, I1_hi = summarize(I1_runs)
        I2_mean, I2_lo, I2_hi = summarize(I2_runs)
        It_mean, It_lo, It_hi = summarize(It_runs)

        return {
            "t_grid": t_grid,
            "I1_mean": I1_mean, "I1_lo": I1_lo, "I1_hi": I1_hi,
            "I2_mean": I2_mean, "I2_lo": I2_lo, "I2_hi": I2_hi,
            "It_mean": It_mean, "It_lo": It_lo, "It_hi": It_hi,
        }
