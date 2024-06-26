import numpy as np
from scipy.special import comb
from typing import List

"""
Channels to use in the Noisy Susceptible-Infected Model. Still under construction. 

- Sagar Kumar, 2024 
"""


def poly_gen(m: int) -> np.ndarray:

    """
    Generates a hypercube of dimension m

    :param m: dimension
    :return: hypercube adjacency matrix
    """

    if m <= 0:
        raise ValueError("The dimension of the cube must be a positive integer.")

    # Calculate the total number of vertices in the n-cube
    num_vertices = 2 ** m

    # Initialize an n x n matrix with all zeros
    adjacency_matrix = [[0] * num_vertices for _ in range(num_vertices)]

    # Iterate through all vertices
    for i in range(num_vertices):
        # Iterate through all dimensions
        for j in range(m):
            # Toggle the j-th bit to find the adjacent vertex
            neighbor = i ^ (1 << j)
            adjacency_matrix[i][neighbor] = 1

    return np.array(adjacency_matrix)


def single_flip_channel(m: int, 
                        e: float) -> np.ndarray:
    """
    Generates an m-bit binary symmetric channel with error e. Only single bit flips allowed, so error
    is e/m for all messages with Hamming Distance = 1.

    :param m: hypercube dim/message length
    :param e: error (probability of departing from original message)

    :return: Numpy array of transition probabilities
    """

    selfloops = np.eye(2 ** m) * (1 - e)
    polygon = poly_gen(m) * e / m

    return selfloops + polygon


def bin_asym_channel(e0: float,
                     e1: float) -> np.ndarray:
    
    """
    Creates a 1 bit Binary Asymmetric Channel 
    
    :param e0: error (probability of departing from original message) of 0 state
    :param e1: error (probability of departing from original message) of 1 state

    :return: Numpy array of transition probabilities
    """
    
    M = np.array([[1-e0, e0], [e1, 1-e1]])
    
    return M


def n_flip_channel(m: int, 
                   e: float) -> np.ndarray:
    
    """
    Generates an m-bit binary symmetric channel with error e. N bit flips are allowed, so error
    is the likelihood of going from one message to another is a function of its Hamming Distance. The error value
    input, however, is still the probabiltiy of departing from the input message. Correct weights are solved as
    a polynomial.

    :param m: hypercube dim/message length
    :param e: error (probability of departing from original message)

    :return: Numpy array of transition probabilities
    """

    def generate_ith_row_pascal(i: int) -> list:
        """
        Generates the ith row of Pascal's Triangle. Written by GPT4 after user prompt: 'generate the ith row of Pascal's
        Triangle using inbuilt functions in python.'

        :param i: row number
        :return: list with number of messages d steps away for each index d
        """
        return [int(comb(i, j)) for j in range(i + 1)]

    mcube: np.ndarray = poly_gen(m)

    arrays: List[np.ndarray] = list()

    # number of message distance d away for an m-dim hypercube corresponds to the mth row in pascals triangle
    errors: list = np.roots(generate_ith_row_pascal(m)[:-1] + [-1*e])
    errors: list = [root.real for root in errors if root.imag == 0 and root.real > 0] # selecting the positive real root

    if len(errors) == 0:
        raise ValueError('Error does not render positive, real transition probabilities.')
    else:
        error = errors[0]

    for d in range(m):
        mpower = np.linalg.matrix_power(mcube, d + 1)
        binary_arr = (mpower != 0)*1
        arrays.append(error ** (d + 1) * binary_arr)

    Q: np.ndarray = np.eye(2 ** m) * (1 - e)

    # applying errors to each value based on distance by raising the adjacency to some power and
    for arr in arrays:
        mask = (Q == 0)*1
        Q = mask * arr + Q

    return Q


def star_channel(n: int,
                 e: float) -> np.ndarray:

    """
    Generate a star-shaped channel (one node with n-1 degree) where the hub node is always message 0.
    Error of departing is the same for hub and spokes.

    :param n: number of nodes/messages
    :param e: error (probability of departing from original message)

    :return: Numpy array of transition probabilities
    """

    err = e/(n-1)

    hub_probs: np.ndarray = np.array([1-e] + [err]*(n-1))
    spoke_probs: list = list()

    for i in range(1,n):
        i_probs: np.ndarray = np.array([e] + [0]*(n-1))
        i_probs[i] = 1 - e
        spoke_probs.append(i_probs)

    all_probs = [hub_probs] + spoke_probs
    Q: np.ndarray = np.array(all_probs)

    return Q


def simplex_channel(n: int,
                    e: float) -> np.ndarray:

    """
    Generates a symmetric n-simplex-shaped channel.

    :param n: number of nodes/messages
    :param e: error (probability of departing from original message)

    :return: Numpy array of transition probabilities
    """

    err = e / (n - 1)

    adj: np.ndarray = np.ones((n,n))
    Q: np.ndarray = err * adj

    np.fill_diagonal(Q, 1-e)

    return Q


def paw_channel(e: float) -> np.ndarray:

    """
    Generates a symmetric "paw graph" channel in which 3 nodes/messages for a triangle, and an additional node/message is attached
    to node 2

    :param e: error (probability of departing from original message)
    :return: Numpy array of transition probabilities
    """

    adj = np.array([[0, 1, 1, 0],
                    [1, 0, 1, 0],
                    [1, 1, 0, 1],
                    [0, 0, 1, 0]])

    degrees = np.sum(adj, axis=1)

    errors = e / degrees

    Q = adj * errors.reshape(-1,1)

    np.fill_diagonal(Q, 1-e)

    return Q


def symmetric_graphical_channel(a: np.ndarray,
                                e: float):

    """
    Create a valid transition matrix from an arbitrary adjacency matrix, a

    :param a: adjacency matrix
    :param e: error (probability of departing from original message)
    :return: Numpy array of transition probabilities
    """

    degrees = np.sum(a, axis=1)

    errors = e / degrees

    Q = a * errors.reshape(-1,1)

    np.fill_diagonal(Q, 1-e)

    return Q
