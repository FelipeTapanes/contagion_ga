'''
Docstring for sir_model
Implements a simple SIR model on a network of counties, 
where the edges represent travel times between county towns. 
The model includes local infections, imported infections from neighboring counties, and recoveries. 
The parameters beta, gamma, and tau control the rates of these processes.

'''

import pandas as pd
import numpy as np


def build_adjacency(edges_csv_path, use_time, time_scale):
    """
    Reads the edges CSV file and builds an adjacency list with weights based on travel times.
    The weights are computed as w_ij = exp(-t_ij / time_scale), where t_ij is the travel time between counties i and j. 
    This means that shorter travel times correspond to stronger connections (higher weights) in the network.
    Returns:
    - nodes: a list of county names (nodes in the graph)
    - neighbors: a list of lists, where neighbors[i] is a list of (j, w_ij) pairs representing the neighbors of node i and the corresponding weights.
    """

# Load the edges from the CSV file and extract the unique nodes
    edges = pd.read_csv(edges_csv_path)
    nodes = sorted(set(edges["node_a"]).union(set(edges["node_b"])))

# Create a mapping from node names to indices for easy lookup
    idx = {name: i for i, name in enumerate(nodes)}

    
# Build the empty adjacency list with weights based on travel times
    neighbors = [[] for i in range(len(nodes))]

# Iterate over the edges and fill the adjacency list with neighbors and weights
    for _, row in edges.iterrows():
        a = row["node_a"]
        b = row["node_b"]
        t = float(row[use_time])

    # Convert travel time to weight using an exponential decay function
        w = np.exp(-t / time_scale)

        i = idx[a]
        j = idx[b]

        neighbors[i].append((j, w))
        neighbors[j].append((i, w))

    return nodes, neighbors




def init_sir(nodes, counties_csv_path, seed_county, seed_infected=50):
    """
    Initializes the SIR model with the population of each county and the initial number of infected individuals in the seed county.
    Returns:
    - S: an array of susceptible individuals in each county
    - I: an array of infected individuals in each county
    - R: an array of recovered individuals in each county
    - N: an array of total population in each county

    """
    # Load county data into a DataFrame and create a mapping from county names to population
    counties = pd.read_csv(counties_csv_path)
    county_population = dict(zip(counties["county_town"], counties["population"]))

    # Initialize S, I, R, and N arrays based on the population of each county 
    # and the initial number of infected individuals in the seed county
    N = np.array([county_population[name] for name in nodes], dtype=float)
    S = N.copy()
    I = np.zeros(len(N), dtype=float)
    R = np.zeros(len(N), dtype=float)
 
    # Set the initial number of infected individuals in the seed county 
    # and update the susceptible population accordingly
    seed_index = nodes.index(seed_county)
    I[seed_index] = float(seed_infected)
    S[seed_index] -= float(seed_infected)

    return S, I, R, N




def sir_step(S, I, R, N, neighbors, beta, gamma, tau):
    """
    Performs one step of the SIR model on the network.
    - S: array of susceptible individuals in each county
    - I: array of infected individuals in each county
    - R: array of recovered individuals in each county
    - N: array of total population in each county
    - neighbors: adjacency list with weights representing the connections between counties
    - beta: infection rate for local infections
    - gamma: recovery rate
    - tau: scaling factor for imported infections from neighboring counties
    Returns:
    - S2: updated array of susceptible individuals in each county
    - I2: updated array of infected individuals in each county
    - R2: updated array of recovered individuals in each county
    """

    # Local infections in each county based on the SIR model formula
    new_local = beta * S * I / N

    # Imported infections from neighboring counties
    new_import = np.zeros_like(S)

    for i in range(len(S)):

        # Calculate the pressure from neighboring counties based on the number of infected individuals 
        # and the weights of the connections
        pressure = 0.0
        for (j, w_ij) in neighbors[i]:
            pressure += w_ij * (I[j] / N[j])

        # The number of new imported infections in county i is proportional to the pressure from neighboring counties and the number of susceptible individuals in county i
        new_import[i] = tau[i] * S[i] * pressure

    # Recoveries
    new_rec = gamma * I

    # Update
    S2 = S - new_local - new_import
    I2 = I + new_local + new_import - new_rec
    R2 = R + new_rec

    # Safety: avoid negative values from floating point due to discrete modeling
    S2 = np.clip(S2, 0, None)
    I2 = np.clip(I2, 0, None)
    R2 = np.clip(R2, 0, None)

    return S2, I2, R2


def run_sir(S, I, R, N, neighbors, beta, gamma, tau, T=120, z_beta = None, z_gamma = None, z_tau = None):
    """
    Runs the SIR model for T time steps and returns the total number of infected individuals at each step.
    - S: array of susceptible individuals in each county
    - I: array of infected individuals in each county
    - R: array of recovered individuals in each county
    - N: array of total population in each county
    - neighbors: adjacency list with weights representing the connections between counties
    - beta: infection rate for local infections
    - gamma: recovery rate
    - tau: scaling factor for imported infections from neighboring counties
    - T: number of time steps to run the simulation
    - z_beta: optional array of intervention parameters that reduce the infection rate
    - z_gamma: optional array of intervention parameters that increase the recovery rate 
    - z_tau: optional array of intervention parameters that reduce the imported infection rate
    Returns:- total_I: an array of the total number of infected individuals at each time step
    """

    # If the intervention parameters are not provided, initialize them as zero arrays (no intervention)
    n = len(S)

    if z_beta is None:
        z_beta = np.zeros(n)

    if z_gamma is None:
        z_gamma = np.zeros(n)

    if z_tau is None:
        z_tau = np.zeros(n)

    total_I = []

    # set the effective parameters for the SIR model based on the intervention parameters.
    beta_intervention  = beta  * (1 - z_beta)
    gamma_intervention = gamma * (1 + z_gamma)
    tau_intervention   = tau   * (1 - z_tau)

    # Run the SIR model for T time steps and record the total number of infected individuals at each step
    for _ in range(T):
        total_I.append(I.sum())
        S, I, R = sir_step(S, I, R, N, neighbors, beta_intervention, gamma_intervention, tau_intervention)

    # Return the total number of infected individuals at each time step as a numpy array 
    # because it is more convenient for plotting and analysis
    return np.array(total_I)
