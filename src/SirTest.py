from sir_model import build_adjacency, init_sir, run_sir
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def test():
    # build network from your edges file
    nodes, neighbors = build_adjacency(
        edges_csv_path="data/edges_k4.csv",  
        use_time="minutes",                      
        time_scale=60.0
    )

    # initialise SIR state
    S, I, R, N = init_sir(
        nodes=nodes,
        counties_csv_path="data/counties.csv",
        seed_county="Dublin",
        seed_infected=50
    )

    # parameters
    beta = 0.43
    gamma = 0.1
    tau = 0.1
    T = 120

    n = len(S)

    # run sim
    total_I = run_sir(S, I, R, N, neighbors, beta, gamma, tau, T=T)

    # quick checks
    print("Counties:", len(nodes))
    print("Initial total infected:", total_I[0])
    print("Peak total infected:", total_I.max())
    print("Final total infected:", total_I[-1])

    # plot total infected over time
    plt.plot(total_I)
    plt.title("Total infected over time")
    plt.xlabel("Time step")
    plt.ylabel("Total infected")
    plt.ylim(0, N.sum())
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x/1_000_000:.1f}M"))
    plt.show()


    print("Total pop:", N.sum())
    print("End pop check:", (S + I + R).sum())




if __name__ == "__main__":
    test()
