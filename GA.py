import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st


# -------------------- Problem Definitions --------------------
@dataclass
class GAProblem:
    name: str
    chromosome_type: str  # 'bit' or 'real'
    dim: int
    bounds: Tuple[float, float] | None
    fitness_fn: Callable[[np.ndarray], float]


def make_onemax(dim: int) -> GAProblem:
    """
    Custom version:
    - Chromosome length = dim
    - Max fitness occurs when ones == 50
    - Fitness = 80 when ones == 50
    - Otherwise drop according to abs difference
    """
    def fitness(x: np.ndarray) -> float:
        ones = np.sum(x)
        if ones == 50:
            return 80.0
        return 80.0 - abs(ones - 50)

    return GAProblem(
        name=f"OneMax ({dim} bits, peak @50)",
        chromosome_type="bit",
        dim=dim,
        bounds=None,
        fitness_fn=fitness,
    )


# -------------------- GA Operators --------------------
def init_population(problem: GAProblem, pop_size: int, rng: np.random.Generator) -> np.ndarray:
    if problem.chromosome_type == "bit":
        return rng.integers(0, 2, size=(pop_size, problem.dim), dtype=np.int8)
    else:
        assert problem.bounds is not None
        lo, hi = problem.bounds
        return rng.uniform(lo, hi, size=(pop_size, problem.dim))


def tournament_selection(fitness: np.ndarray, k: int, rng: np.random.Generator) -> int:
    idxs = rng.integers(0, fitness.size, size=k)
    best = idxs[np.argmax(fitness[idxs])]
    return int(best)


def one_point_crossover(a: np.ndarray, b: np.ndarray, rng: np.random.Generator):
    if a.size <= 1:
        return a.copy(), b.copy()
    point = int(rng.integers(1, a.size))
    c1 = np.concatenate([a[:point], b[point:]])
    c2 = np.concatenate([b[:point], a[point:]])
    return c1, c2


def bit_mutation(x: np.ndarray, mut_rate: float, rng: np.random.Generator) -> np.ndarray:
    mask = rng.random(x.shape) < mut_rate
    y = x.copy()
    y[mask] = 1 - y[mask]
    return y


def evaluate(pop: np.ndarray, problem: GAProblem) -> np.ndarray:
    return np.array([problem.fitness_fn(ind) for ind in pop], dtype=float)


def run_ga(
    problem: GAProblem,
    pop_size: int,
    generations: int,
    crossover_rate: float,
    mutation_rate: float,
    tournament_k: int,
    elitism: int,
    seed: int | None,
    stream_live: bool = True,
):
    rng = np.random.default_rng(seed)
    pop = init_population(problem, pop_size, rng)
    fit = evaluate(pop, problem)

    chart_area = st.empty()
    best_area = st.empty()

    history_best = []
    history_avg = []
    history_worst = []

    for gen in range(generations):
        best_idx = int(np.argmax(fit))
        best_fit = float(fit[best_idx])
        avg_fit = float(np.mean(fit))
        worst_fit = float(np.min(fit))

        history_best.append(best_fit)
        history_avg.append(avg_fit)
        history_worst.append(worst_fit)

        if stream_live:
            df = pd.DataFrame({"Best": history_best, "Average": history_avg, "Worst": history_worst})
            chart_area.line_chart(df)
            best_area.markdown(
                f"Generation {gen+1}/{generations} — Best fitness: **{best_fit:.6f}**"
            )

        # Elitism
        E = max(0, min(elitism, pop_size))
        elite_idx = np.argpartition(fit, -E)[-E:] if E > 0 else np.array([], dtype=int)
        elites = pop[elite_idx].copy() if E > 0 else np.empty((0, pop.shape[1]))

        next_pop = []
        while len(next_pop) < pop_size - E:
            i1 = tournament_selection(fit, tournament_k, rng)
            i2 = tournament_selection(fit, tournament_k, rng)
            p1, p2 = pop[i1], pop[i2]

            if rng.random() < crossover_rate:
                c1, c2 = one_point_crossover(p1, p2, rng)
            else:
                c1, c2 = p1.copy(), p2.copy()

            c1 = bit_mutation(c1, mutation_rate, rng)
            c2 = bit_mutation(c2, mutation_rate, rng)

            next_pop.append(c1)
            if len(next_pop) < pop_size - E:
                next_pop.append(c2)

        pop = np.vstack([np.array(next_pop), elites]) if E > 0 else np.array(next_pop)
        fit = evaluate(pop, problem)

    best_idx = int(np.argmax(fit))
    best = pop[best_idx].copy()
    best_fit = float(fit[best_idx])

    df = pd.DataFrame({"Best": history_best, "Average": history_avg, "Worst": history_worst})

    return {
        "best": best,
        "best_fitness": best_fit,
        "history": df,
        "final_population": pop,
        "final_fitness": fit,
    }


# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Genetic Algorithm", page_icon="🧬", layout="wide")
st.title("Genetic Algorithm (GA)")
st.caption("Custom OneMax with max fitness at 50 ones.")


# -------------- FIXED PARAMETERS --------------
POP_SIZE = 300
DIM = 80
GENERATIONS = 50
CROSSOVER_RATE = 0.9
MUTATION_RATE = 0.01
TOURNAMENT_K = 3
ELITISM = 2


with st.sidebar:
    st.header("Problem")
    st.write("Type: OneMax (bits)")
    st.number_input("Chromosome length (bits)", value=DIM, disabled=True)

    st.header("GA Parameters")
    st.number_input("Population size", value=POP_SIZE, disabled=True)
    st.number_input("Generations", value=GENERATIONS, disabled=True)
    st.write(f"Crossover rate = {CROSSOVER_RATE}")
    st.write(f"Mutation rate = {MUTATION_RATE}")
    st.write(f"Tournament size = {TOURNAMENT_K}")
    st.write(f"Elites per generation = {ELITISM}")

    seed = st.number_input("Random seed", min_value=0, max_value=2**32 - 1, value=42)
    live = st.checkbox("Live chart while running", value=True)

problem = make_onemax(DIM)

left, right = st.columns([1, 1])

with left:
    if st.button("Run GA", type="primary"):

        result = run_ga(
            problem=problem,
            pop_size=POP_SIZE,
            generations=GENERATIONS,
            crossover_rate=CROSSOVER_RATE,
            mutation_rate=MUTATION_RATE,
            tournament_k=TOURNAMENT_K,
            elitism=ELITISM,
            seed=int(seed),
            stream_live=bool(live),
        )

        st.subheader("Fitness Over Generations")
        st.line_chart(result["history"])

        st.subheader("Best Solution")
        st.write(f"Best fitness: {result['best_fitness']:.6f}")

        bitstring = ''.join(map(str, result["best"].astype(int).tolist()))
        st.code(bitstring, language="text")
        st.write(f"Number of ones: {int(np.sum(result['best']))} / {problem.dim}")


with right:
    st.subheader("Population Snapshot (final)")
    st.caption("Shows first 20 individuals with fitness")
    if st.button("Show final population table"):
        pop = result["final_population"]
        fit = result["final_fitness"]
        nshow = min(20, pop.shape[0])
        df = pd.DataFrame(pop[:nshow])
        df["fitness"] = fit[:nshow]
        st.dataframe(df, use_container_width=True)
