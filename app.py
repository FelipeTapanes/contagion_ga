# app.py
from matplotlib import ticker
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import time
from src.sir_model import build_adjacency, init_sir, run_sir
from src.nsga2_sir import decode, run_nsga2

if "pareto_front" not in st.session_state:
    st.session_state.pareto_front = None
if "counties" not in st.session_state:
    st.session_state.counties = None

@st.cache_data
def load_counties():
    df = pd.read_csv("data/counties.csv")
    df.sort_values("county_town", inplace=True)
    return df["county_town"].tolist()


st.set_page_config(page_title="Ireland Contagion Optimiser", layout="wide")

st.title("Ireland Contagion Optimiser (NSGA-II)")
st.write("Run the optimiser and inspect the Pareto front + interventions.")

with st.sidebar.form("params_form"):
    
    st.header("SIR model parameters")
    beta = st.slider("beta (local spread)", 0.01, 1.0, 0.43, step=0.01)
    gamma = st.slider("gamma (recovery)", 0.01, 1.0, 0.1, step=0.01)
    tau = st.slider("tau (network spread)", 0.01, 1.0, 0.1, step=0.01)
    days = st.slider("Simulation days", 30, 365, 120, step=5)

    counties = load_counties()
    seed_county = st.selectbox(
    "Seed county",
    options= counties,
    index=counties.index("Dublin")        
    )
    seed_infected = st.number_input("Starting infected number", value=50, step=1)


    st.header("Model / GA parameters")
    pop_size = st.slider("Population size", 20, 400, 120, step=10)
    ngen = st.slider("Generations", 5, 300, 80, step=5)
    seed = st.number_input("Random seed", value=42, step=1)

    run_btn = st.form_submit_button("Run NSGA-II")



def plot_pareto(front):
    costs = []
    infected = []

    for s in front:
        costs.append(s.fitness.values[1])
        infected.append(s.fitness.values[0])


    fig, ax = plt.subplots()
    
    ax.scatter(costs, infected)
    ax.set_xlabel("Intervention Cost")
    ax.set_ylabel("Peak Infected")
    ax.set_title("NSGA-II Pareto Front")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x/1_000_000:.1f}M"))
    st.pyplot(fig)

    return costs, infected

def plot_SIR():


    nodes, neighbors = build_adjacency(
        edges_csv_path="data/edges_k4.csv",  
        use_time="minutes",                      
        time_scale=60.0
    )

   
    S, I, R, N = init_sir(
        nodes=nodes,
        counties_csv_path="data/counties.csv",
        seed_county=seed_county,
        seed_infected=seed_infected
    )

    n = len(S)
    total_I = run_sir(S, I, R, N, neighbors, beta, gamma, tau, T=days)


    fig, ax = plt.subplots()

    ax.plot(total_I)
    ax.set_title("Total infected over time")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Total infected")
    ax.set_ylim(0, N.sum())
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x/1_000_000:.1f}M"))
    st.pyplot(fig)

    return total_I.max()




if run_btn:

    status = st.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()

    status.info("Running optimisation… (this may take a bit depending on generations/population)")

    def progress_callback(gen, ngen, best_peak=None, best_cost=None):
        progress_bar.progress(gen / ngen)
        if best_peak is not None and best_cost is not None:
            status_text.info(f"Gen {gen}/{ngen} | best peak={best_peak:,.0f} | best cost={best_cost:.2f}")
        else:
            status_text.info(f"Generation {gen}/{ngen}")
        time.sleep(0.01)  # <-- tiny yield so UI actually refreshes

    counties, pareto_front = run_nsga2(
        T=days,
        pop_size=pop_size,
        ngen=ngen,
        seed=int(seed),
        beta=beta,
        gamma=gamma,
        tau=tau,
        seed_county=str(seed_county),
        seed_infected=int(seed_infected),
        progress_callback=progress_callback
    )

    progress_bar.progress(1.0)
    status_text.empty()
    status.success(f"Pareto front complete: {len(pareto_front)} solutions found")
    st.session_state.counties = counties
    st.session_state.pareto_front = pareto_front

    status.info(f"Pareto front size: {len(pareto_front)} solutions found.")


if st.session_state.pareto_front is not None:
    nodes = st.session_state.counties
    pareto_front = st.session_state.pareto_front

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("SIR contagion (no interventions)")
        peak0 = plot_SIR()

        st.metric("Peak infected (baseline)", f"{peak0:,.0f}")

        st.subheader("Pareto front")
        costs, infected = plot_pareto(pareto_front)

    with col2:

        st.subheader("Inerventions for a selected solution and its SIR curve")
        pareto_front = sorted(pareto_front, key=lambda ind: ind.fitness.values[1])


        idx = st.slider("Solution index", 1, len(pareto_front), 1) - 1
        sol = pareto_front[idx]

        n = len(nodes)
        z_beta, z_gamma, z_tau = decode(sol, n)

        df = pd.DataFrame(
            [z_beta, z_gamma, z_tau],
            index=["beta", "gamma", "tau"],
            columns=nodes
        )

        st.dataframe(df)

        colA, colB, colC = st.columns(3)

        peak_I, cost = sol.fitness.values

        colA.metric("Peak infected", f"{peak_I:,.0f}")
        colB.metric("Cost", f"{cost:,.2f}")
        reduction = 1.0 - (peak_I / peak0)
        colC.metric("Reduction vs baseline", f"{reduction*100:.1f}%")


        nodes, neighbors = build_adjacency(
        edges_csv_path="data/edges_k4.csv",  
        use_time="minutes",                      
        time_scale=60.0
        )

        S, I, R, N = init_sir(
        nodes=nodes,
        counties_csv_path="data/counties.csv",
        seed_county=seed_county,
        seed_infected=seed_infected
        )

        n = len(S)
        total_I = run_sir(S, I, R, N, neighbors=neighbors, beta=beta, gamma=gamma, tau=tau, T=120, z_beta = z_beta, z_gamma = z_gamma, z_tau = z_tau)

        fig, ax = plt.subplots()
        ax.plot(total_I)
        ax.set_title("Total infected over time")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Total infected")
        ax.set_ylim(0, N.sum())
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x/1_000_000:.1f}M"))
        st.pyplot(fig)

        peak_I, cost = sol.fitness.values
