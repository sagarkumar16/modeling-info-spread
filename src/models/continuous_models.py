import numpy as np
from typing import Union, List, Tuple, Callable
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult

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
        #self.history = [(self.time, self.state.copy())]

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



class ODE:

    def __init__(self,
                 beta,
                 k,
                 channel,
                 T,
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

    def __call__(self, initial_state) -> Tuple[np.ndarray[float], np.ndarray[float]]:

        return solve_ivp(fun=self.differential, t_span=[0,self.T], y0=initial_state)

    def differential(self, t, phi) -> np.ndarray[float]:

        """
        Mean field d(phi)/d(t)
        :param t: time (not relevant because this is autonomous, but necessary for solve_ivp()
        :param phi: message concentrations
        :return: differential at time t
        """

        return self.beta * self.k * (1-np.sum(phi)) * (self.channel @ phi)






