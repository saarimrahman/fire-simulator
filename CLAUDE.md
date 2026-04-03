# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the app

```bash
# Activate venv first
source venv/bin/activate

# Run the Streamlit dashboard
streamlit run fire_dashboard.py
# or directly:
./venv/bin/streamlit run fire_dashboard.py
```

## Architecture

This is a two-file project:

- **`fire.py`** — Simulation engine. Contains all financial logic, dataclasses, and the core `run_vectorized()` function.
- **`fire_dashboard.py`** — Streamlit UI. Imports from `fire.py`, renders sidebar controls, and builds all charts.

### Simulation design

The simulation is **fully vectorized with numpy**: it runs `N_SIMS` (default 10,000) independent simulations simultaneously using `(N_YEARS, N_SIMS)` shaped arrays. There are no Python loops over simulations — only over years.

### Key dataclasses in `fire.py`

| Class | Purpose |
|---|---|
| `CityConfig` | Housing costs, mortgage parameters, state tax rate per city |
| `SeedAmounts` | Starting account balances (taxable, 401k, Roth, HSA) |
| `FamilyConfig` | Marriage age, kid ages, spouse work pattern and salary |
| `CareerConfig` | Salary trajectory, promotion windows, 401k employer match |
| `SocialSecurityConfig` | SS benefit amounts and claiming age |
| `SimulationResults` | All output arrays from `run_vectorized()` |

### Key functions in `fire.py`

- `run_vectorized()` — Main entry point. Takes all config objects + city name, returns `SimulationResults`.
- `calc_taxes_vec()` — Vectorized federal+state+FICA tax calculation across all simulations.
- `calc_spouse_income()` — Spouse work-pattern logic (full-time → gap for kids → part-time).
- `simulate_career_growth()` — Stochastic promotion model generating salary trajectories.
- `find_min_tc()` — Binary searches for minimum TC to achieve a target FIRE probability by a target age.
- `calc_swr()` — Dynamic safe withdrawal rate based on retirement horizon (lower rate for earlier retirement).

### Global constants in `fire.py`

`CURRENT_AGE`, `FIRE_HORIZON`, `LIFE_EXPECTANCY`, `N_SIMS`, and contribution limits (`FOUR01K_LIMIT`, `ROTH_IRA_LIMIT`, `HSA_FAMILY_LIMIT`) are module-level constants. The dashboard overrides `CURRENT_AGE` and `N_SIMS` by passing them as arguments to `run_vectorized()`.

### City data

Built-in cities (Sacramento, Dublin, San Francisco) are defined as `CityConfig` instances in the `CITIES` dict. The dashboard allows users to add custom cities via `add_custom_cities()`, which updates the global dict.

## Dependencies

```
numpy==2.4.2
streamlit==1.54.0
plotly==6.5.2
tabulate==0.9.0
```

Install: `pip install -r requirements.txt`
