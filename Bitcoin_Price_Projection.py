#MonteCarlo analysis of BTC price in 5 years (as pictured), variables can be altered
#to suit other companies or markets
#Also has an excel .csv export file to support regression modeling given simulations
#of running the code.
import math
import random
import statistics
from typing import List, Tuple

def simulate_gbm_path(
    s0: float,
    mu: float,
    sigma: float,
    years: float,
    steps_per_year: int = 252,
    rng: random.Random = random
) -> List[float]:
    
    dt = 1.0 / steps_per_year
    n_steps = int(years * steps_per_year)
    path = [s0]
    drift = (mu - 0.5 * sigma * sigma) * dt
    vol_scale = sigma * math.sqrt(dt)

    for _ in range(n_steps):
        z = rng.gauss(0.0, 1.0)          # standard normal
        s_next = path[-1] * math.exp(drift + vol_scale * z)
        path.append(s_next)

    return path

def percentile(data: List[float], p: float) -> float:

    if not data:
        raise ValueError("Empty data for percentile")
    if p <= 0: 
        return min(data)
    if p >= 100: 
        return max(data)
    xs = sorted(data)
    k = (len(xs) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs[int(k)]
    return xs[f] + (xs[c] - xs[f]) * (k - f)

def simulate_terminal_prices(
    s0: float = 113000.0,   # current BTC price
    mu: float = 0.15,       # expected annual return (15%)
    sigma: float = 0.70,    # annual volatility (70%)
    years: float = 5.0,
    steps_per_year: int = 252,
    n_sims: int = 10000,
    seed: int = 42
) -> Tuple[List[float], dict]:
  
    rng = random.Random(seed)
    terminals: List[float] = []

    for _ in range(n_sims):
        path = simulate_gbm_path(s0, mu, sigma, years, steps_per_year, rng)
        terminals.append(path[-1])

    # Summary statistics
    stats = {
        "start_price": s0,
        "years": years,
        "n_sims": n_sims,
        "mean_terminal": statistics.fmean(terminals),
        "median_terminal": percentile(terminals, 50),
        "stdev_terminal": statistics.pstdev(terminals),  # population stdev
        "p5": percentile(terminals, 5),
        "p25": percentile(terminals, 25),
        "p75": percentile(terminals, 75),
        "p95": percentile(terminals, 95),
        "prob_above_start": sum(1 for x in terminals if x > s0) / n_sims,
        "expected_log_return_annualized": (math.log(statistics.fmean(terminals) / s0) / years),
    }
    return terminals, stats

if __name__ == "__main__":
    # --- Configure your assumptions here ---
    S0     = 113000.0   # current price
    MU     = 0.15       # expected annual return
    SIGMA  = 0.70       # annual vol
    YEARS  = 5.0
    STEPS  = 252
    NSIMS  = 10000
    SEED   = 1337

    terminals, stats = simulate_terminal_prices(
        s0=S0, mu=MU, sigma=SIGMA, years=YEARS,
        steps_per_year=STEPS, n_sims=NSIMS, seed=SEED
    )

    print("=== Monte Carlo Summary =")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"{k:>28s}: {v:,.4f}")
        else:
            print(f"{k:>28s}: {v}")
    print("\nSample path snippets:")
    preview_rng = random.Random(SEED)
    for i in range(3):
        path = simulate_gbm_path(S0, MU, SIGMA, years=1.0, steps_per_year=60, rng=preview_rng)
        print(f"Path {i+1} (first 10 pts): " + ", ".join(f"{p:,.0f}" for p in path[:10]))

import csv

# CSV
with open("btc_terminals.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["TerminalPrice"])
    for price in terminals:
        writer.writerow([price])


