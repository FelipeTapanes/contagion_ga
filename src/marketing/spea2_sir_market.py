import random
import numpy as np
from deap import base, creator, tools, algorithms
from .sir_model_market import build_adjacency, init_sir, run_sir


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

        spread = float(total_I.sum())
        cost = intervention_cost(z_beta, z_gamma, z_tau, c_beta, c_gamma, c_tau)
        return (spread, cost)

    return evaluate


def clip08(individual):
    '''
    Clips the values of the individual's genome to be within the range [0.0, 0.8].
    '''
    for i in range(len(individual)):
        if individual[i] < 0.0:
            individual[i] = 0.0
        elif individual[i] > 0.8:
            individual[i] = 0.8
    return individual


def mut_poly_bounded_and_clip(individual, eta, indpb):
    '''
    Applies polynomial mutation to the individual's genome with bounds [0.0, 0.8]
    and then clips the values to ensure they are within the range.
    - individual: the individual's genome to mutate
    - eta: the distribution for the polynomial mutation (higher means less mutation)
    - indpb: independent probability for each gene to be mutated
    Returns: a tuple containing the mutated individual
    '''
    tools.mutPolynomialBounded(individual, eta=eta, low=0.0, up=0.8, indpb=indpb)
    clip08(individual)
    return (individual,)


def evaluate_invalid(toolbox, individuals):
    '''
    Evaluates only individuals whose fitness is not valid yet.
    '''
    invalid = [ind for ind in individuals if not ind.fitness.valid]
    for ind, fit in zip(invalid, map(toolbox.evaluate, invalid)):
        ind.fitness.values = fit


# --------- MAIN SPEA2 ---------

def run_spea2(
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

    # spea2 settings
    pop_size=150,
    archive_size=75,
    ngen=105,
    cxpb=0.9,
    mutpb=0.2,
    seed=42,
    cx_eta=20.0,
    mut_eta=20.0,
    mut_indpb=0.05,

    # optional callback function to report progress after each generation
    progress_callback=None
):
    '''
    Runs the SPEA2 algorithm to optimize intervention parameters for the SIR model.

    Key difference from NSGA-II:
    - SPEA2 maintains an external archive of elite non-dominated solutions.
    - Each generation combines the current population and archive,
      then selects a new archive using SPEA2 fitness assignment.

    Returns:
    - nodes: list of county names
    - front: first Pareto front extracted from the final archive
    '''
    random.seed(seed)
    np.random.seed(seed)

    # Build network
    nodes, neighbors = build_adjacency(edges_csv_path, use_time, time_scale)
    n = len(nodes)
    genome_len = 3 * n

    # DEAP class setup for SPEA2
    if not hasattr(creator, "MarketFitnessSPEA2"):
        creator.create("MarketFitnessSPEA2", base.Fitness, weights=(1.0, -1.0))
    if not hasattr(creator, "MarketIndividualSPEA2"):
        creator.create("MarketIndividualSPEA2", list, fitness=creator.MarketFitnessSPEA2)

    toolbox = base.Toolbox()

    # Attribute generator: random float in [0.0, 0.8]
    toolbox.register("attr_float", random.uniform, 0.0, 0.8)

    # Structure initializers
    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.MarketIndividualSPEA2,
        toolbox.attr_float,
        n=genome_len
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Fitness evaluator
    toolbox.register(
        "evaluate",
        make_evaluator(
            nodes, neighbors, counties_csv_path,
            seed_county, seed_infected,
            beta, gamma, tau, T,
            c_beta, c_gamma, c_tau
        )
    )

    # Variation operators
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=0.0, up=0.8, eta=cx_eta)
    toolbox.register("mutate", mut_poly_bounded_and_clip, eta=mut_eta, indpb=mut_indpb)

    # SPEA2 environmental selection
    toolbox.register("select", tools.selSPEA2)

    # Init population and empty archive
    pop = toolbox.population(n=pop_size)
    archive = []

    # Evaluate initial population
    evaluate_invalid(toolbox, pop)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Step 1: combine population and archive
        union = pop + archive

        # Step 2: choose next archive using SPEA2
        archive = toolbox.select(union, archive_size)

        # Step 3: create mating pool from archive
        if len(archive) == 0:
            mating_pool = pop
        else:
            mating_pool = archive

        # Step 4: clone enough individuals to make the next population
        offspring = [toolbox.clone(ind) for ind in mating_pool]

        while len(offspring) < pop_size:
            offspring.append(toolbox.clone(random.choice(mating_pool)))

        offspring = offspring[:pop_size]

        # Step 5: apply crossover and mutation
        offspring = algorithms.varAnd(offspring, toolbox, cxpb=cxpb, mutpb=mutpb)

        # Step 6: evaluate offspring
        evaluate_invalid(toolbox, offspring)

        # Step 7: offspring becomes the next working population
        pop = offspring

        # Optional progress reporting
        if progress_callback and len(archive) > 0:
            best_peak = min(ind.fitness.values[0] for ind in archive)
            best_cost = min(ind.fitness.values[1] for ind in archive)
            progress_callback(gen, ngen, best_peak, best_cost)

    # Final archive update
    union = pop + archive
    archive = toolbox.select(union, archive_size)

    # First Pareto front from final archive
    front = tools.sortNondominated(archive, k=len(archive), first_front_only=True)[0]

    # Sort front by cost for nicer plotting later
    front = sorted(front, key=lambda ind: ind.fitness.values[1])

    return nodes, front