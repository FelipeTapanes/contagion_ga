# app.py
from matplotlib import ticker
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import time

from src.disease.spea2_sir import run_spea2
from src.disease.sir_model import build_adjacency, init_sir, run_sir
from src.disease.nsga2_sir import decode, run_nsga2

from src.marketing.spea2_sir_market import run_spea2 as run_spea2_marketing
from src.marketing.sir_model_market import (
    build_adjacency as build_adjacency_marketing,
    init_sir as init_sir_marketing,
    run_sir as run_sir_marketing
)
from src.marketing.nsga2_sir_market import decode as decode_marketing, run_nsga2 as run_nsga2_marketing

if "pareto_front" not in st.session_state:
    st.session_state.pareto_front = None
if "counties" not in st.session_state:
    st.session_state.counties = None
if "baseline_curve" not in st.session_state:
    st.session_state.baseline_curve = None
if "baseline_objective" not in st.session_state:
    st.session_state.baseline_objective = None

@st.cache_data
def load_counties():
    df = pd.read_csv("data/counties.csv")
    df.sort_values("county_town", inplace=True)
    return df["county_town"].tolist()




st.set_page_config(page_title="Ireland Contagion Optimiser", layout="wide")

st.title("Ireland Contagion Optimiser (NSGA-II + SPEA2)")
st.write("Run the optimiser and inspect the Pareto front + interventions.")

st.header("Model mode")
mode = st.radio(
    "Contagion type",
    ["Disease", "Marketing"],
    index=0
)


if "last_mode" not in st.session_state:
    st.session_state.last_mode = mode

if st.session_state.last_mode != mode:
    st.session_state.pareto_front = None
    st.session_state.counties = None
    st.session_state.baseline_curve = None
    st.session_state.baseline_objective = None
    st.session_state.last_mode = mode

if mode == "Disease":
    objective_name = "Peak Infected"
    curve_title = "Total infected over time"
    section_title = "SIR contagion: baseline vs selected solution"
    policy_title = "Policy for selected solution"
    baseline_label = "Baseline"
    selected_label = "Selected solution"
    change_label = "Reduction vs baseline"
else:
    objective_name = "Cumulative Spread"
    curve_title = "Total marketing spread over time"
    section_title = "Marketing contagion: baseline vs selected solution"
    policy_title = "Campaign for selected solution"
    baseline_label = "Baseline"
    selected_label = "Selected campaign"
    change_label = "Increase vs baseline"


if mode == "Disease":
    run_nsga2 = run_nsga2
    run_spea2 = run_spea2
    build_adjacency = build_adjacency
    init_sir = init_sir
    run_sir = run_sir
    decode = decode
else:
    run_nsga2 = run_nsga2_marketing
    run_spea2 = run_spea2_marketing
    build_adjacency = build_adjacency_marketing
    init_sir = init_sir_marketing
    run_sir = run_sir_marketing
    decode = decode_marketing



with st.sidebar.form("params_form"):
    
    st.header("SIR model parameters")

    if mode == "Disease":
        beta = st.slider("beta (local spread)", 0.01, 1.0, 0.43, step=0.01)
        gamma = st.slider("gamma (recovery)", 0.01, 1.0, 0.10, step=0.01)
        tau = st.slider("tau (network spread)", 0.01, 1.0, 0.10, step=0.01)
    else:
        beta = st.slider("beta (local adoption)", 0.01, 1.0, 0.43, step=0.01)
        gamma = st.slider("gamma (drop-off)", 0.01, 1.0, 0.10, step=0.01)
        tau = st.slider("tau (network diffusion)", 0.01, 1.0, 0.10, step=0.01)
    days = st.slider("Simulation days", 30, 365, 120, step=5)

    counties = load_counties()
    seed_county = st.selectbox(
        "Seed county",
        options= counties,
        index=counties.index("Dublin")        
    )

    seed_label = "Starting infected number" if mode == "Disease" else "Starting adopters"
    seed_infected = st.number_input(seed_label, value=50, step=1)


    st.header("Model / GA parameters")
    algorithm = st.radio(
        "Optimisation algorithm",
        ["NSGA-II", "SPEA2"],
        index=0
    )
    pop_size = st.slider("Population size", 20, 400, 120, step=10)
    ngen = st.slider("Generations", 5, 300, 80, step=5)
    seed = st.number_input("Random seed", value=42, step=1)

    run_btn = st.form_submit_button("Run GA")


def plot_pareto(front, selected_idx=None):
    costs = []
    objective_values = []

    for s in front:
        costs.append(s.fitness.values[1])
        objective_values.append(s.fitness.values[0])

    fig, ax = plt.subplots()
    ax.scatter(costs, objective_values)

    if selected_idx is not None:
        ax.scatter(
            costs[selected_idx],
            objective_values[selected_idx],
            color="red",
            s=180,
            label="Selected solution"
        )

    ax.set_xlabel("Intervention Cost")
    ax.set_ylabel(objective_name)
    ax.set_title("GA Pareto Front")
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: f"{x/1_000_000:.1f}M")
    )
    ax.legend()
    st.pyplot(fig)

    return costs, objective_values


def run_baseline_simulation():
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

    total_I = run_sir(S, I, R, N, neighbors, beta, gamma, tau, T=days)

    if mode == "Disease":
        baseline_objective = total_I.max()
    else:
        baseline_objective = total_I.sum()

    return total_I, baseline_objective, N.sum()


if run_btn:

    status = st.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()

    status.info("Running optimisation… (this may take a bit depending on generations/population)")

    def progress_callback(gen, ngen, best_obj=None, best_cost=None):
        progress_bar.progress(gen / ngen)
        if best_obj is not None and best_cost is not None:
            status_text.info(
                f"Gen {gen}/{ngen} | best {objective_name.lower()}={best_obj:,.0f} | best cost={best_cost:.2f}"
            )
        else:
            status_text.info(f"Generation {gen}/{ngen}")
        time.sleep(0.01)

    if algorithm == "NSGA-II":
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
        
    else:
        counties, pareto_front = run_spea2(
            T=days,
            pop_size=pop_size,
            archive_size=max(10, pop_size // 2),
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
    baseline_curve, baseline_objective, pop_sum = run_baseline_simulation()
    st.session_state.baseline_curve = baseline_curve
    st.session_state.baseline_objective = baseline_objective
    st.session_state.pop_sum = pop_sum

    status.info(f"Pareto front size: {len(pareto_front)} solutions found.")


if st.session_state.pareto_front is not None:
    nodes = st.session_state.counties
    pareto_front = st.session_state.pareto_front

    pareto_front = sorted(pareto_front, key=lambda ind: ind.fitness.values[1])
    idx = st.slider("Solution index", 1, len(pareto_front), 1) - 1
    sol = pareto_front[idx]

    n = len(nodes)
    z_beta, z_gamma, z_tau = decode(sol, n)

    nodes2, neighbors = build_adjacency("data/edges_k4.csv", "minutes", 60.0)
    S, I, R, N = init_sir(nodes2, "data/counties.csv", seed_county, seed_infected)

    selected_curve = run_sir(
        S, I, R, N, neighbors,
        beta=beta, gamma=gamma, tau=tau,
        T=days,
        z_beta=z_beta, z_gamma=z_gamma, z_tau=z_tau
    )

    st.subheader(section_title)

    baseline_curve = st.session_state.baseline_curve
    pop_sum = st.session_state.pop_sum

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(baseline_curve, label=baseline_label)
    ax.plot(selected_curve, label=selected_label)
    ax.set_title(curve_title)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Total infected" if mode == "Disease" else "Total active spread")
    ax.set_ylim(0, pop_sum)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x/1_000_000:.1f}M"))
    ax.legend()
    st.pyplot(fig)
    

    colA, colB, colC, colD = st.columns(4)

    cost = sol.fitness.values[1]
    baseline_value = st.session_state.baseline_objective

    if mode == "Disease":
        objective_value = sol.fitness.values[0]
    else:
        objective_value = float(selected_curve.sum())

    colA.metric(f"{objective_name} (baseline)", f"{baseline_value:,.0f}")
    colB.metric(f"{objective_name} (selected)", f"{objective_value:,.0f}")
    colC.metric("Cost", f"{cost:,.2f}")

    if mode == "Disease":
        change = 1.0 - (objective_value / baseline_value)
        colD.metric("Reduction vs baseline", f"{change*100:.1f}%")
    else:
        absolute_gain = objective_value - baseline_value
        pct_gain = (objective_value / baseline_value) - 1.0

        if absolute_gain >= 0:
            colD.metric("Spread gain", f"{absolute_gain:,.0f}", f"+{pct_gain*100:.1f}%")
        else:
            colD.metric("Spread loss", f"{abs(absolute_gain):,.0f}", f"{pct_gain*100:.1f}%")


    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Pareto front")
        plot_pareto(pareto_front, selected_idx=idx)

    with col2:
        n = len(nodes)
        z_beta, z_gamma, z_tau = decode(sol, n)

        st.subheader(policy_title)

        if mode == "Disease":
            row_names = ["beta", "gamma", "tau"]
        else:
            row_names = ["beta boost", "retention", "tau boost"]

        df = pd.DataFrame(
            [z_beta, z_gamma, z_tau],
            index=row_names,
            columns=nodes
            )
        st.dataframe(df)