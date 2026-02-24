import random
import numpy as np
from deap import base, creator, tools, algorithms
from sir_model import build_adjacency, init_sir, run_sir


# --------- HELPER FUNCTIONS ---------

def decode(individual, n):
    '''
    Decodes the individual's genome into intervention parameters for beta, gamma, and tau.
    The genome is expected to be a list of length 3*n, where the first n values correspond to z_beta,
    the next n to z_gamma, and the last n to z_tau.
    '''
    z_beta = np.array(individual[0:n], dtype=float)
    z_gamma = np.array(individual[n:2*n], dtype=float)
    z_tau = np.array(individual[2*n:3*n], dtype=float)
    return z_beta, z_gamma, z_tau


def intervention_cost(z_beta, z_gamma, z_tau, c_beta=1.0, c_gamma=1.0, c_tau=1.0):
    '''
    Calculates the cost of the interventions based on the parameters and their respective costs.
    The cost is a weighted sum of the intervention parameters, where c_beta, c_gamma, 
    and c_tau are the weights for the respective parameters.
    '''
    return float(c_beta * z_beta.sum() + c_gamma * z_gamma.sum() + c_tau * z_tau.sum())


def make_evaluator(nodes, neighbors, counties_csv_path,
                   seed_county, seed_infected,
                   beta, gamma, tau, T,
                   c_beta=1.0, c_gamma=1.0, c_tau=1.0):
    
    '''
    Creates an evaluation function for the genetic algorithm that simulates the SIR model with interventions 
    and calculates the peak number of infected individuals and the cost of the interventions.
    - nodes: list of county names
    - neighbors: adjacency list with weights representing the connections between counties
    - counties_csv_path: path to the CSV file containing county population data 
    - seed_county: the county where the initial infections are seeded
    - seed_infected: the initial number of infected individuals in the seed county
    - beta: base infection rate for local infections
    - gamma: base recovery rate
    - tau: base scaling factor for imported infections from neighboring counties
    - T: number of time steps to run the simulation (days)
    - c_beta: cost weight for interventions that reduce the infection rate
    - c_gamma: cost weight for interventions that increase the recovery rate
    - c_tau: cost weight for interventions that reduce the imported infection rate
    Returns: a function that takes an individual's genome as input and returns a tuple
    (peak number of infected individuals, cost of interventions)
    '''
    n = len(nodes)

    def evaluate(individual):
        '''
        Evaluates an individual's fitness by simulating the SIR model with the decoded intervention parameters
        and calculating the peak number of infected individuals and the cost of the interventions.
        - individual: a list representing the individual's genome, where the first n values correspond to z_beta,
          the next n to z_gamma, and the last n to z_tau.
        Returns: a tuple (peak number of infected individuals, cost of interventions)   
        '''
        z_beta, z_gamma, z_tau = decode(individual, n)
        S0, I0, R0, N0 = init_sir(
            nodes,
            counties_csv_path,
            seed_county,
            seed_infected
        )

        total_I = run_sir(
            S0, I0, R0, N0, neighbors,
            beta, gamma, tau,
            T,
            z_beta, z_gamma, z_tau
        )

        peak_I = float(total_I.max())
        cost = intervention_cost(z_beta, z_gamma, z_tau, c_beta, c_gamma, c_tau)
        return (peak_I, cost)

    return evaluate


def clip01(individual):
    '''
    Clips the values of the individual's genome to be within the range [0, 1].
    '''
    for i in range(len(individual)):
        if individual[i] < 0.0:
            individual[i] = 0.0
        elif individual[i] > 1.0:
            individual[i] = 1.0
    return individual


def mut_poly_bounded_and_clip(individual, eta, indpb):
    '''
    Applies polynomial mutation to the individual's genome with bounds [0, 1] and then clips the values to ensure they are within the range.
    - individual: the individual's genome to mutate
    - eta: the distribution for the polynomial mutation (higher means less mutation)
    - indpb: independent probability for each gene to be mutated
    Returns: a tuple containing the mutated individual
    '''
    tools.mutPolynomialBounded(individual, eta=eta, low=0.0, up=1.0, indpb=indpb)
    clip01(individual)
    return (individual,)



# --------- MAIN NSGA-II ---------

def run_nsga2(
    # data / model
    edges_csv_path="data/edges_k4.csv",
    counties_csv_path="data/counties.csv",
    use_time="minutes",
    time_scale=60.0,
    seed_county="Dublin",
    seed_infected=50,
    beta=0.43,
    gamma=0.10,
    tau=0.10,
    T=120,
    # cost weights
    c_beta=1.0,
    c_gamma=1.0,
    c_tau=1.0,
    # nsga2 settings
    pop_size=150,
    # - number of generations: you can adjust this based on how long you want the optimization to run 
    # - and how good you want the solutions to be. More generations typically lead to better solutions but take more time.
    ngen=150,
    # - crossover probability: this controls how often crossover (recombination) happens between individuals.
    # - A common choice is around 0.9, which means that 90% of the time, offspring are produced by crossing over two parents,
    # -  and 10% of the time, offspring are produced by copying a parent (without crossover).
    cxpb=0.9,
    # - mutation probability: this controls how often mutation happens to the offspring.
    # - A common choice is around 0.1 to 0.2, which means that 10% to 20% of the time, 
    # - an offspring will undergo mutation after crossover
    mutpb=0.2,
    # - random seed for reproducibility
    seed=42,
    # - crossover strength 
    cx_eta=20.0,
    # - mutation strength and probability
    mut_eta=20.0,
    # - per-gene mutation probability
    mut_indpb=0.05
):
    '''
    Runs the NSGA-II algorithm to optimize intervention parameters for the SIR model.
    - data / model parameters: used to set up the SIR model and the evaluation function
    - cost weights: used to calculate the cost of interventions in the evaluation function
    - nsga2 settings: parameters for the genetic algorithm, including population size, number of generations, crossover probability, mutation probability, and random seed for reproducibility
    Returns: a tuple containing the list of county names and the Pareto front of solutions (list of individuals) after running the NSGA-II algorithm
    '''
    random.seed(seed)
    np.random.seed(seed)

    # Build network
    nodes, neighbors = build_adjacency(edges_csv_path, use_time, time_scale)
    n = len(nodes)
    genome_len = 3 * n

    # DEAP class setup for NSGA-II (2 objectives: minimize peak infections and minimize cost)  
    if not hasattr(creator, "SirFitness"):
        creator.create("SirFitness", base.Fitness, weights=(-1.0, -1.0))
    if not hasattr(creator, "SirIndividual"):
        creator.create("SirIndividual", list, fitness=creator.SirFitness)

    toolbox = base.Toolbox()

    # Attribute generator: random float in [0, 1]
    toolbox.register("attr_float", random.random)
    # Structure initializers: define 'individual' to be a list of floats and 'population' to be a list of individuals
    toolbox.register("individual", tools.initRepeat, creator.SirIndividual, toolbox.attr_float, n=genome_len)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Fitness evaluator. make_evaluator is a closure that captures the model parameters 
    # and returns a function that evaluates an individual's fitness by simulating the SIR model with the decoded intervention parameters
    # and calculating the peak number of infected individuals and the cost of the interventions.
    toolbox.register(
        "evaluate",
        make_evaluator(nodes, neighbors, counties_csv_path,
                       seed_county, seed_infected,
                       beta, gamma, tau, T,
                       c_beta, c_gamma, c_tau)
    )

    # NSGA-II operators
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=0.0, up=1.0, eta=cx_eta)
    toolbox.register("mutate", mut_poly_bounded_and_clip, eta=mut_eta, indpb=mut_indpb)
    toolbox.register("select", tools.selNSGA2)

    # Init population
    pop = toolbox.population(n=pop_size)

    # Evaluate initial population and assign fitness values
    invalid = [ind for ind in pop if not ind.fitness.valid]
    for ind, fit in zip(invalid, map(toolbox.evaluate, invalid)):
        ind.fitness.values = fit

    # as the first step of NSGA-II, we sort the population and select the best individuals to form the first generation
    pop = toolbox.select(pop, len(pop))

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Generate offspring through crossover and mutation
        offspring = algorithms.varAnd(pop, toolbox, cxpb=cxpb, mutpb=mutpb)

    # Evaluate the offspring and assign fitness values
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind, fit in zip(invalid, map(toolbox.evaluate, invalid)):
            ind.fitness.values = fit

    # Select the next generation population from the combined parent and offspring population
        pop = toolbox.select(pop + offspring, pop_size)

        # quick progress print
        peaks = [ind.fitness.values[0] for ind in pop]
        costs = [ind.fitness.values[1] for ind in pop]
        print(f"Gen {gen:03d} | best peak={min(peaks):.0f} | best cost={min(costs):.2f}")

    # First Pareto front
    front = tools.sortNondominated(pop, k=len(pop), first_front_only=True)[0]
    return nodes, front