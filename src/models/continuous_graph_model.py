import numpy as np
import numpy.typing as npt
from typing import Union, List, Tuple, Callable
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from tqdm import tqdm
import networkx as nx

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

        
def continuous_time_random_walk(Q: npt.NDArray,
                                rate: float,
                                initial_state:npt.NDArray,
                                max_time: int):
    """
    Perform a continuous-time random walk on a graph with weighted adjacency matrix Q,
    with random time steps based on an exponential distribution.

    :param Q: Weighted adjacency matrix of the graph.
    :param rate: Rate parameter for the exponential distribution of time steps.
    :param initial_state: Initial distribution of walkers.
    :param max_time: Maximum simulation time.

    Returns:
    numpy.ndarray: State of walkers after each randomly selected time step.
    list: List of cumulative times at each step.
    """
    # Normalize Q to create a probability transition matrix
    row_sums = Q.sum(axis=1, keepdims=True)
    transition_matrix = Q / row_sums

    # Initialize the state of walkers
    state = np.array(initial_state, dtype=float)
    states_over_time = [state.copy()]
    times = [0]

    # Continuous-time evolution until max_time is reached
    current_time = 0
    while current_time < max_time:
        # Sample a time step from an exponential distribution with mean = rate
        delta_t = np.random.exponential(rate)

        current_time += delta_t

        # select random node based on current state
        r1 = np.random.uniform(0, np.sum(state))
        outgoing = np.searchsorted(np.cumsum(state), r1)
        # select what happens based on Q
        a = Q @ state
        r2 = np.random.uniform(0, np.sum(a))

        incoming = np.searchsorted(np.cumsum(a), r2)

        state[incoming] += 1
        state[outgoing] -= 1

        # Ensure state distribution is non-negative
        #state[state < 0] = 0
        states_over_time.append(state.copy())
        times.append(current_time)

    return times, np.array(states_over_time)


class ODE:

    def __init__(self,
                 beta,
                 k,
                 channel,
                 T,
                 t_eval,
                 integration_method: str = 'RK45'
                 ):

        """
        Wrapper for scipy integration of the ODE.

        :param beta: transmissibility
        :param k: average number of contacts
        :param channel: noisy channel
        :param T: time steps
        :param integration_method: scipy.integrate.solve_ivp method (Default RK45)
        """

        self.beta = beta
        self.k = k
        self.channel = channel
        self.T = T
        self.integration_method = integration_method
        self.t_eval=t_eval

    def __call__(self, initial_state, function='NSI') -> Tuple[np.ndarray[float], np.ndarray[float]]:
        
        if function=='NSI':
            return solve_ivp(fun=self.differential, t_span=[0,self.T], y0=initial_state, t_eval=self.t_eval)
        elif function=='RW':
            return solve_ivp(fun=self.diffusion, t_span=[0,self.T], y0=initial_state, t_eval=self.t_eval)
        elif function=='SI':
            return solve_ivp(fun=self.spread, t_span=[0,self.T], y0=initial_state, t_eval=self.t_eval)
        else:
            raise ValueError('function must be either NSI, SI, or RW')

    def differential(self, t, phi) -> np.ndarray[float]:

        """
        Mean field d(phi)/d(t)
        :param t: time (not relevant because this is autonomous, but necessary for solve_ivp()
        :param phi: message concentrations
        :return: differential at time t
        """

        return self.beta * self.k * (1-np.sum(phi)) * (self.channel @ phi)
    
    def diffusion(self, t, phi) -> np.ndarray[float]:
        
        """
        Random Walk d(phi)/d(t) = Q*phi
        :param t: time
        :param phi: message concentrations
        :return: diffusion equation at time t
        """
        
        return self.beta * self.k * (self.channel - np.eye(self.channel.shape[0])) @ phi
    
    def spread(self, t, phi) -> np.ndarray[float]:
        
        """
        SI Model
        :param t: time
        :param phi: message concentrations
        :return: spread at time t
        """
        
        return self.beta * self.k * (1-np.sum(phi)) * phi


def simulate_communication(runs: int,
                           T: int,
                           N: int,
                           beta: float,
                           k: Union[float, int],
                           channel: np.ndarray[float],
                           encoding: np.ndarray[float],
                           pW: np.ndarray[float],
                           function: str):

    results = []

    for _ in tqdm(range(runs), desc=f'Simulating {function}'):
        # Random true state
        rw = np.random.uniform(0,1)
        w = np.searchsorted(np.cumsum(pW), rw)

        # what is the message encoded?
        re = np.random.uniform(0,1)
        e = np.searchsorted(np.cumsum(encoding[w]), re)

        # initial state
        initial_state = np.zeros(channel.shape[0])
        initial_state[e] = 1

        if function == 'NSI':
            # run Gillespie
            sim = NoisyGillespie(N=N, initial_state=initial_state, beta=beta, k=k,
                                 channel=channel)
            run = sim.simulate(max_time=T, density=True)

        elif function == 'SI':
            # run Gillespie
            sim = NoisyGillespie(N=N, initial_state=initial_state, beta=beta, k=k,
                                 channel=channel)
            run = sim.simulate(max_time=T, density=True, function='SI')

        elif function == 'RW':
            run = continuous_time_random_walk(Q=channel, rate=1/(beta*k),
                                              initial_state=initial_state, max_time=T)

        else:
            raise ValueError("Must specify a function, either 'NSI', 'SI' or 'RW'")

        # Append source state, simulation time, simulated values
        results.append((w, run[0], run[1]))

    return results

    
def mutual_info(M, X):

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


def bootstrap_mi_ci(common_time: npt.NDArray,
                    run_dict: dict[int,npt.NDArray],
                    pW: npt.NDArray,
                    n_bootstrap: int = 100):

    """
    Estimates the 95% Confidence Interval using a bootstrap resampling method.

    :param common_time: Common time that values are interpolated on
    :param run_dict: Dictionary of runs keyed by intial state (see: interpolated_curves dict in run_MI())
    :return: (lower interval for each point, upper interval for each point)
    """

    ci_lower = []
    ci_upper = []

    for n,t in tqdm(enumerate(common_time), desc='Estimating CIs'):
        MIs = []
        for _ in range(n_bootstrap):
            means = []
            for key, val in run_dict.items():
                arr = np.array(val)[:,n]
                # Resample with replacement from each seed's results in this time step
                num_rows = arr.shape[0]
                bootstrap = arr[np.random.choice(num_rows, size=num_rows, replace=True)]
                mean = np.mean(bootstrap, axis=0)
                means.append(mean)
            means = np.array(means)
            norm = np.array(means)/np.sum(means, axis=1, keepdims=True)
            MIs.append(mutual_info(norm, pW))
        MIs = np.array(MIs)
        ci_lower_t = np.percentile(MIs, 2.5)
        ci_upper_t = np.percentile(MIs, 97.5)

        ci_lower.append(ci_lower_t)
        ci_upper.append(ci_upper_t)

    return ci_lower, ci_upper


def simulate_mi(function: str,
                Q: npt.NDArray[np.float64],
                E: npt.NDArray[np.float64],
                pW: npt.NDArray[np.float64],
                N: int,
                beta: float,
                k: int,
                num_runs: int,
                T: int,
                common_time: npt.NDArray,
                run_ci: bool = True,
                n_bootstrap: int = None
                ):

    sim_runs = simulate_communication(runs=num_runs,
                                      T=T,
                                      N=N,
                                      beta=beta,
                                      k=k,
                                      channel=Q,
                                      encoding=E,
                                      pW=pW,
                                      function=function)

    # Interpolated curves for each source state
    interpolated_curves = {state: [] for state in range(Q.shape[0])}
    source_states = []

    for w, run_t, run_y in sim_runs:
        source_states.append(w)
        interp_func = interp1d(run_t, run_y, kind='linear', fill_value='extrapolate', axis=0)
        interpolated_curves[w].append(interp_func(common_time))

    empirical_source_dist = np.bincount(source_states) / num_runs
    mean_curves = np.array([np.mean(np.array(val), axis=0) for val in interpolated_curves.values()])
    norm_curves = mean_curves / np.sum(mean_curves, axis=2, keepdims=True)
    MI_sim_means = [mutual_info(np.array(i), empirical_source_dist) for i in list(zip(*norm_curves))]

    if run_ci:
        MI_lower, MI_upper = bootstrap_mi_ci(common_time=common_time,
                                             run_dict=interpolated_curves,
                                             pW=empirical_source_dist,
                                             n_bootstrap=n_bootstrap)

        return MI_sim_means, MI_lower, MI_upper


def run_MI(Q: npt.NDArray[np.float64],
           E: npt.NDArray[np.float64],
           pW: npt.NDArray[np.float64] = None,
           N: int = 10e3,
           beta: float = 0.01,
           k: int = 4,
           num_runs: int =100,
           T = 1000,
           run_simulations: bool = True,
           run_rw_sim: bool = True,
           run_si_sim: bool = True,
           run_ci: bool = True,
           run_mean_field: bool = True,
           run_random_walk: bool = True,
           run_si: bool = True,
           resolution: int = 10,
           n_bootstrap: int = 100):

    """

    :param Q:
    :param E:
    :param pW:
    :param N:
    :param beta:
    :param k:
    :param num_runs:
    :param T:
    :param run_simulations:
    :param run_rw_sim:
    :param run_si_sim:
    :param run_ci:
    :param run_mean_field:
    :param run_random_walk:
    :param run_si:
    :param resolution: time steps in between interpolated points
    :param n_bootstrap: iterations of the bootstrap
    :return:
    """
    
    output = {}

    common_time = np.linspace(0, T, int(T/resolution) + 1)
    output['time'] = common_time

    # default uniform encoding
    if pW is None:
        pW = np.array([1/Q.shape[0]]*Q.shape[0])
    else:
        pW = pW
    
    # array true states
    Omega = np.eye(Q.shape[0])

    # Initial States (Row Vectors)
    phi0_arr = E @ Omega

    # Noisy Gillespie Algorithm
    if run_simulations:
        model_sim_means, model_lower, model_upper = simulate_mi(function='NSI',
                                                                Q=Q,
                                                                E=E,
                                                                pW=pW,
                                                                N=N,
                                                                beta=beta,
                                                                k=k,
                                                                T=T,
                                                                common_time=common_time,
                                                                run_ci=run_ci,
                                                                n_bootstrap=n_bootstrap,
                                                                num_runs=num_runs)

        output['mc_means'] = model_sim_means
        output['mc_lower'] = model_lower
        output['mc_upper'] = model_upper

    if run_rw_sim:
        rw_sim_means, rw_lower, rw_upper = simulate_mi(function='RW',
                                                                Q=Q,
                                                                E=E,
                                                                pW=pW,
                                                                N=N,
                                                                beta=beta,
                                                                k=k,
                                                                T=T,
                                                                common_time=common_time,
                                                                run_ci=run_ci,
                                                                n_bootstrap=n_bootstrap,
                                                                num_runs=num_runs)

        output['rw_means'] = rw_sim_means
        output['rw_lower'] = rw_lower
        output['rw_upper'] = rw_upper

    if run_si_sim:
        si_sim_means, si_lower, si_upper = simulate_mi(function='SI',
                                                                Q=Q,
                                                                E=E,
                                                                pW=pW,
                                                                N=N,
                                                                beta=beta,
                                                                k=k,
                                                                T=T,
                                                                common_time=common_time,
                                                                run_ci=run_ci,
                                                                n_bootstrap=n_bootstrap,
                                                                num_runs=num_runs)

        output['si_means'] = si_sim_means
        output['si_lower'] = si_lower
        output['si_upper'] = si_upper

    # Numerical Approximations    
    mean_field = ODE(beta=beta, k=k, channel=Q, T=T, t_eval=common_time)

    # Accumulate MI for each analytical model
    analytic_MI = []
    rw_MI = []
    si_MI = []

    # Accumulate runs of each analytical model
    analytic_curves = []
    random_walks = []
    SIs = []

    for row in phi0_arr:
        if run_mean_field:
            analytic_curve = mean_field(initial_state=row/N)
            analytic_curves.append(analytic_curve)

        if run_random_walk:
            random_walk = mean_field(initial_state=row/N, function='RW')
            random_walks.append(random_walk)

        if run_si:
            SI = mean_field(initial_state=row/N, function='SI')
            SIs.append(SI)

    for n, t in enumerate(common_time):
        if run_mean_field:
            phi_t = []
            for curve in analytic_curves:
                message = curve.y.T
                phi_n = message[n] / np.sum(message[n])
                phi_t.append(phi_n)
            analytic_MI.append(mutual_info(np.stack(phi_t), pW))


        if run_random_walk:
            phi_t_rw = []
            for rw in random_walks:
                message = rw.y.T
                phi_n = message[n] / np.sum(message[n])
                phi_t_rw.append(phi_n)
            rw_MI.append(mutual_info(np.stack(phi_t_rw), pW))

        if run_si:
            phi_t_si = []
            for si in SIs:
                message = si.y.T
                phi_n = message[n] / np.sum(message[n])
                phi_t_si.append(phi_n)
            si_MI.append(mutual_info(np.stack(phi_t_si), pW))
    
    if run_mean_field:
        output['mean_field'] = analytic_MI
    
    if run_random_walk:
        output['markov_chain'] = rw_MI
        
    if run_si:
        output['SI'] = si_MI

    return output


def objective(px, Q):
    
    """
    Objective function for scipy.optimize.minimize() used in estimate_capacity()
    """
    
    return -mutual_info(Q, px)


def estimate_capacity(Q):
    
    """
    Estimate capacity of a channel Q using least squares.
    """
    
    # Initial guess for input distribution P(X)
    px0 = np.array([1/Q.shape[0]]*Q.shape[0])  

    # Constraints: probabilities must sum to 1
    constraints = ({'type': 'eq', 'fun': lambda px: np.sum(px) - 1})

    # Bounds: each probability must be between 0 and 1
    bounds = [(0, 1)]*Q.shape[0]

    # Optimize to find the input distribution that maximizes mutual information
    result = minimize(objective, px0, args=(Q,), bounds=bounds, constraints=constraints, method='SLSQP')

    # Channel capacity is the negative of the minimized result
    channel_capacity = -result.fun
    optimal_px = result.x

    return channel_capacity, optimal_px


def final_dist(pW: npt.NDArray[np.float64],
               Q: Callable,
               E: Callable,
               T: int = 1000,
               N: int = 10e3,
               beta: float = 0.01,
               k: int = 4):
    """
    Returns MI value of a communication process occurring over the channels specified.
    """
    if pW is None:
        pW = np.array([1 / Q.shape[0]] * Q.shape[0])
    else:
        pW = pW

    # array true states
    Omega = np.eye(Q.shape[0])

    # Initial States (Row Vectors)
    phi0_arr = E @ Omega

    mean_field = ODE(beta=beta, k=k, channel=Q, T=T, t_eval=np.linspace(0, T, int(T / 10) + 1))

    analytic_curves = []

    for row in phi0_arr:
        analytic_curve = mean_field(initial_state=row / N)
        analytic_curves.append(analytic_curve)

    phi_t = []
    for curve in analytic_curves:
        final_message = curve.y.T[-1]
        phi_n = final_message / np.sum(final_message)
        phi_t.append(phi_n)

    return -1 * mutual_info(np.stack(phi_t), pW)


def estimate_spreading_capacity(
        Q: Callable,
        E: Callable,
        T: int = 1000,
        N: int = 10e3,
        beta: float = 0.01,
        k: int = 4):

    # Initial guess for input distribution P(X)
    px0 = np.array([1 / Q.shape[0]] * Q.shape[0])

    # Constraints: probabilities must sum to 1
    constraints = ({'type': 'eq', 'fun': lambda px: np.sum(px) - 1})

    # Bounds: each probability must be between 0 and 1
    bounds = [(0, 1)] * Q.shape[0]

    # Optimize to find the input distribution that maximizes mutual information
    result = minimize(final_dist, px0, args=(Q, E, T, N, beta, k), bounds=bounds, constraints=constraints,
                      method='SLSQP')

    # Channel capacity is the negative of the minimized result
    channel_capacity = -result.fun
    optimal_px = result.x

    return channel_capacity, optimal_px

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
        t_grid:np.ndarray = None,
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
