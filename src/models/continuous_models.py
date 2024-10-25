import numpy as np
import numpy.typing as npt
from typing import Union, List, Tuple, Callable
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from tqdm import tqdm


class NoisyGillespie:

    def __init__(self,
                 N: int,
                 initial_state: np.ndarray[float],
                 beta: float,
                 k: Union[float, int],
                 channel: np.ndarray[float]):

        """
        Instantiate the Noisy Gillespie Stochastic Model.

        :param N: Population size
        :param initial_state: phi * N at time t=0
        :param beta: transmissibility
        :param k: average number of contacts
        :param channel: noisy channel
        """
        self.N = N
        self.state = np.array(initial_state.copy(), dtype=float)
        self.beta = beta
        self.k = k
        self.channel = channel
        self.time = 0.0
        self.t = [self.time]
        self.y = [initial_state.copy()]

    def propensities(self) -> np.ndarray[float]:

        """
        Calculates the differential d(phi)/d(t) * N
        :return: Array of propensities for each phi_i * N
        """

        return self.beta * self.k * (self.N - np.sum(self.state)) * (self.channel @ self.state)

    def step(self):

        """
        Take a step of random (exponentially distributed) time in the Gillespie algorithm and update the model state.
        :return: boolean of if saturation has NOT occurred yet
        """

        a = self.propensities()
        a0 = a.sum()
        if a0 == 0:
            return False  # No more events can happen

        # Time to next event
        tau = np.random.exponential(self.N / a0)

        # Determine which event occurs
        r = np.random.uniform(0, a0)

        event = np.searchsorted(np.cumsum(a), r)

        for e in range(a.shape[0]):
            if event == e:
                self.state[e] += 1  # increase the value of the infection corresponding to the selected event

        # Update time
        self.time += tau
        self.t.append(self.time)
        self.y.append(self.state.copy())
        #self.history.append((self.time, self.state.copy()))
        return True

    def simulate(self,
                 max_time: float,
                 max_steps: int = int(1e6),
                 density: bool = True) -> Tuple[np.ndarray[float], np.ndarray[float]]:

        """
        Simulate Noisy Gillespie Algorithm for maximum number of time steps or until saturation.
        :param max_time: Maximum tau
        :param max_steps: max number of algorithm iterations.
        :param density: Whether to return density (if False, outputs integer # of individuals the received each message)
        :return: Tuple of 1D x [time steps] array of model time and [dim_phi] x [time steps] array of states full full
        history
        """

        steps = 0
        while (self.time < max_time and steps < max_steps and
               np.sum(self.state) <= self.N):  # As long as not everybody has been infected
            if not self.step():
                break
            steps += 1

        if density:
            return np.array(self.t), np.array(self.y)/self.N

        else:
            return np.array(self.t), np.array(self.y)

        
def continuous_markov_chain(Q: npt.NDArray[np.float64],
                            initial_state: npt.NDArray[np.float64],
                            beta: float,
                            k: int,
                            T: int):
    """
    Simulates a Continuous-Time Markov Chain.
    """
    num_states = Q.shape[0]
    states = [initial_state]
    times = [0]
    current_state = initial_state
    current_time = 0
    
    while current_time < T:
        # Calculate the rate of leaving the current state (diagonal of Q)
        rate = -Q[current_state, current_state] / tau
        
        # Sample the time to the next transition (exponentially distributed)
        wait_time = np.random.exponential(1 / rate)
        current_time += wait_time
        
        if current_time > T:
            break
        
        # Choose the next state based on the transition probabilities
        # Normalize transition rates (row of Q corresponding to current_state)
        transition_probs = Q[current_state] / -Q[current_state, current_state]
        
        next_state = np.random.choice(num_states, p=transition_probs)
        states.append(next_state)
        times.append(current_time)
        current_state = next_state
    
    return times, states


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
        
        return (self.channel - np.eye(self.channel.shape[0])) @ phi
    
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
                           pW: np.ndarray[float]):

    results = []

    for _ in tqdm(range(runs), desc='Simulating'):
        # Random true state
        rw = np.random.uniform(0,1)
        w = np.searchsorted(np.cumsum(pW), rw)

        # what is the message encoded?
        re = np.random.uniform(0,1)
        e = np.searchsorted(np.cumsum(encoding[w]), re)

        # initial state
        initial_state = np.zeros(channel.shape[0])
        initial_state[e] = 1

        # run Gillespie
        sim = NoisyGillespie(N=N, initial_state=initial_state, beta=beta, k=k,
                             channel=channel)
        run = sim.simulate(max_time=T, density=True)

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


def run_MI(Q: npt.NDArray[np.float64],
           E: npt.NDArray[np.float64],
           pW: npt.NDArray[np.float64] = None,
           N: int = 10e3,
           beta: float = 0.01,
           k: int = 4,
           num_runs: int =100,
           T = 1000,
           run_simulations=True,
           run_mean_field=True,
           run_random_walk=True,
           run_si=True):
    
    """
    Runs Gillespie and Mean Field 
    """
    
    output = {}
    
    MI_sim_runs = []
    analytic_MI = []
    rw_MI = []
    si_MI = []
    MI_sim_lower = []
    MI_sim_higher = []
    
    
    common_time = np.linspace(0, T, int(T/10) + 1)
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
        sim_runs = simulate_communication(runs=num_runs,
                                      T=T,
                                      N=N,
                                      beta=beta,
                                      k=k,
                                      channel=Q,
                                      encoding=E,
                                      pW=pW)

        # Interpolated curves for each source state
        interpolated_curves = {state: [] for state in range(Q.shape[0])}
        source_states = []

        for w, run_t, run_y in sim_runs:
            source_states.append(w)
            interp_func = interp1d(run_t, run_y, kind='linear', fill_value='extrapolate', axis=0)
            interpolated_curves[w].append(interp_func(common_time))

        empirical_source_dist = np.bincount(source_states)/num_runs
        mean_curves = np.array([np.mean(np.array(val), axis=0) for val in interpolated_curves.values()])
        norm_curves = mean_curves/np.sum(mean_curves, axis=2, keepdims=True)
        MI_sim_means = [mutual_info(np.array(i), empirical_source_dist) for i in list(zip(*norm_curves))]

        output['mc_means'] = MI_sim_means

    # Numerical Approximations    
    mean_field = ODE(beta=beta, k=k, channel=Q, T=T, t_eval=common_time)
    
    analytic_curves = []
    random_walks = []
    SIs = []

    time_stamps = None
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
    
    for n,t in enumerate(common_time):
        if run_mean_field:
            phi_t = []
            for curve in analytic_curves:
                message = curve.y.T
                phi_n = message[n]/np.sum(message[n])
                phi_t.append(phi_n)
            analytic_MI.append(mutual_info(np.stack(phi_t), pW))
            
        
        if run_random_walk:
            phi_t_rw = []
            for rw in random_walks:
                message = rw.y.T
                phi_n = message[n]/np.sum(message[n])
                phi_t_rw.append(phi_n)
            rw_MI.append(mutual_info(np.stack(phi_t_rw), pW))
        
        if run_si:
            phi_t_si = []
            for si in SIs:
                message = si.y.T
                phi_n = message[n]/np.sum(message[n])
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
