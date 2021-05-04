import numpy as np
import math
from shapely.geometry import Polygon, Point
import utm


# H = np.array().T

def fspl_model(TX_coords, RX_coords, m, n, l, h_TX, h_RX, h_prime, wavelength, G_TX, G_RX):

    # Model is indexed from 1 ~ m or 1 ~ n, but due to Python syntax, here it is indexed from 0 ~ m-1 or 0 ~ n-1

    # How to calculate these values?
    d_11, theta = d11_and_theta(TX_coords, RX_coords)

    h = h_RX - h_TX  # Difference of RX and TX heights, is negative when h_RX < h_TX (which is intentional)

    # Initialize arrays to store 2D distance from first row of MIMO, and channel matrix elements
    d_1j = []
    H_elements = []

    # Initialize array to store (amp, phase) of each antenna pair
    phase_and_amp = np.empty(shape=(m, n))

    # Compute the d_(1,j), the distance between RX (first row of antennas) and TX in 2D
    for j in range(n):
        d = d_11 - j * l * math.cos(theta)
        d_1j.append(d)

    # Iterate through each row of RX MIMO antennas
    for i in range(m):

        # Iterate through each antenna on row of RX MIMO
        for j in range(n):
            # Compute p_(i,j), the true distance between RX and TX antennas in 3D
            p_ij = abs(math.sqrt((d_1j[j]) ** 2 + (h + i * h_prime) ** 2))

            # Calculate the phase
            phase = 2 * math.pi * p_ij / wavelength

            # Calculate the gain amplitude
            amp = G_TX * G_RX * (wavelength / (4 * math.pi * p_ij)) ** 2

            # Fill in phase and amp values for this pair of antennas
            phase_and_amp[i][j] = (phase, amp)

            # Compute H matrix element corresponding to the (i, j)th RX antenna and TX antenna pair
            H_element = amp * math.exp(1j * phase)
            H_elements.append(H_element)

    # Save H matrix as an array of size mn x 1
    H = np.array(H_elements).T

    return H, phase_and_amp


def d11_and_theta(TX_coords, RX_coords):
    # Convert lat/long coordinates of TX's first antenna and RX into utm
    TX_coord = utm.from_latlon(TX_coords[0], TX_coords[1])
    RX_11_coord = utm.from_latlon(RX_coords[0], RX_coords[1])

    # Compute the 2D distance between RX and TX's first antenna (used in model as base 2D distance)
    d_11 = abs(math.sqrt((TX_coord[0] - RX_11_coord[0]) ** 2 + (TX_coord[1] - RX_11_coord[1]) ** 2))

    # Compute the theta angle (used in model for computing other 2D distances)
    tangent = abs((TX_coord[0] - RX_11_coord[0]) / (TX_coord[1] - RX_11_coord[1]))
    theta = math.degrees(math.atan(tangent))

    return d_11, theta

# tangent = math.sqrt(3)
# print(math.degrees(math.atan(tangent)))
