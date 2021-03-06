import numpy as np
import math
import utm
import cmath


def trgr_model(TX_coords, RX_coords, m, n, l, h_TX, h_RX, h_prime, wavelength, G_TX, G_RX, e_r):
    """
    Find the channel matrix H and phase/amplitude of two-ray ground reflection model for MIMO

    :param TX_coords: Latitude/longitude coordinates of transmitter (i.e. drone or pole)
    :param RX_coords: Latitude/longitude coordinates of one RX antenna (first row, rightmost)
    :param m: Number of RX antenna rows in MIMO
    :param n: Number of RX antenna in each row
    :param l: Horizontal distance between each RX antenna in same row (MIMO)
    :param h_TX: Height of transmitter
    :param h_RX: Height of receiver
    :param h_prime: Vertical distance between each row of RX antennas (MIMO)
    :param wavelength: wavelength of signal (assumed equal for each pair of TX and RX antennas)
    :param G_TX: Gain of transmitter
    :param G_RX: Gain of receiver
    :param e_r: relative permittivity (ex: ground, metal bleachers)
    :return:
    """

    # Model is indexed from 1 ~ m or 1 ~ n, but due to Python syntax, here it is indexed from 0 ~ m-1 or 0 ~ n-1

    # How to calculate these values?
    d_11, phi = d11_and_phi(TX_coords, RX_coords)

    h = h_RX - h_TX  # Difference of RX and TX heights, is negative when h_RX < h_TX (which is intentional)

    # Initialize arrays to store 2D distance from first row of MIMO, and channel matrix elements
    d_1j = []
    H_elements = []

    # Initialize array to store (amp, phase) of each antenna pair
    amps = np.empty(shape=(m, n))
    phases = np.empty(shape=(m, n))

    # Compute the d_(1,j), the distance between RX (first row of antennas) and TX in 2D
    for j in range(n):
        d = d_11 - j * l * math.cos(phi)
        d_1j.append(d)

    # Iterate through each row of RX MIMO antennas
    for i in range(m):

        # Iterate through each antenna on row of RX MIMO
        for j in range(n):
            # Compute p_LOS_(i,j), the true line-of-sight distance between TX and RX antennas in 3D
            p_LOS_ij = abs(math.sqrt((d_1j[j]) ** 2 + (h + i * h_prime) ** 2))

            # Compute p_REF_(i,j), the true reflected distance between TX and RX antennas in 3D
            p_REF_ij = abs(math.sqrt((d_1j[j]) ** 2 + (h_TX + h_RX + i * h_prime) ** 2))

            # Compute the reflection coefficient, and the theta along with it
            theta_ij = math.atan((h_TX + h_RX + i * h_prime) / d_1j[j])

            c = math.sqrt(e_r - (math.cos(theta_ij)) ** 2)

            r_ij = (math.sin(theta_ij) - c) / (math.sin(theta_ij) + c)

            # Calculate the phase difference
            phase = math.degrees(2 * math.pi * (p_REF_ij - p_LOS_ij) / wavelength) % 360

            # Calculate the gain amplitude
            amp = G_TX * G_RX * ((wavelength / (4 * math.pi)) ** 2) * (
                        abs(1 / p_LOS_ij + r_ij * cmath.exp(-1j * phase) / p_REF_ij) ** 2)

            # Fill in phase and amp values for this pair of antennas
            amps[i][j] = amp
            phases[i][j] = phase

            # Compute H matrix element corresponding to the (i, j)th RX antenna and TX antenna pair
            H_element = amp * cmath.exp(1j * phase)
            H_elements.append(H_element)

    # Save H matrix as an array of size mn x 1
    H = np.array(H_elements).T

    return H, amps, phases


def d11_and_phi(TX_coords, RX_coords):
    # Convert lat/long coordinates of TX's first antenna and RX into utm
    TX_coord = utm.from_latlon(TX_coords[0], TX_coords[1])
    RX_11_coord = utm.from_latlon(RX_coords[0], RX_coords[1])

    # Compute the 2D distance between RX and TX's first antenna (used in model as base 2D distance)
    d_11 = abs(math.sqrt((TX_coord[0] - RX_11_coord[0]) ** 2 + (TX_coord[1] - RX_11_coord[1]) ** 2))

    # Compute the phi angle (used in model for computing other 2D distances)
    tangent = abs((TX_coord[0] - RX_11_coord[0]) / (TX_coord[1] - RX_11_coord[1]))
    phi = math.degrees(math.atan(tangent))  # Convert to degrees

    return d_11, phi
