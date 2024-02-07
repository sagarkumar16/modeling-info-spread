import numpy as np
from scipy.special import comb


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
    """

    selfloops = np.eye(2 ** m) * (1 - e)
    polygon = poly_gen(m) * e / m

    return selfloops + polygon


def n_flip_channel(m: int, 
                   e: float) -> np.ndarray:
    
    """
    Generates an m-bit binary symmetric channel with error e. N bit flips are allowed, so error
    is the likelihood of going from one message to another is a function of its Hamming Distance. The error value
    input, however, is still the probabiltiy of departing from the input message. Correct weights are solved as
    a polynomial.

    :param m: hypercube dim/message length
    :param e: error (probability of departing from original message)
    """

    def generate_ith_row_pascal(i: int) -> list:
        """
        Generates the ith row of Pascal's Triangle. Written by GPT4 after user prompt: 'generate the ith row of Pascal's
        Triangle using inbuilt functions in python.'

        :param i: row number
        :return: list
        """
        return [int(comb(i, j)) for j in range(i + 1)]

    mcube: np.ndarray = poly_gen(m)

    arrays: list[np.ndarray] = list()

    # number of message distance d away for an m-dim hypercube corresponds to the mth row in pascals triangle
    errors: list = np.roots(generate_ith_row_pascal(m)[:-1] + [-1*e])
    errors: list = [root.real for root in errors if root.imag == 0 and root.real > 0] # selecting the positive real root

    if len(errors) == 0:
        raise ValueError('Error does not render positive, real transition probabilities.')
    else:
        error = errors[0]

    for d in range(m):
        mpower = np.linalg.matrix_power(mcube, d + 1)
        binary_arr = (mpower != 0).astype(int)
        arrays.append(error ** (d + 1) * binary_arr)

    Q: np.ndarray = np.eye(2 ** m) * (1 - e)

    # applying errors to each value based on distance by raising the adjacency to some power and
    for arr in arrays:
        mask = (Q == 0).astype(int)
        Q = mask * arr + Q

    return Q



        
        

# class Channel:
#
#     def __init__(self,
#                  name: str):

