# FIRE Monte Carlo Simulator

A Monte Carlo simulator for **FIRE** (Financial Independence, Retire Early). Model your income, savings, housing, family, and market randomness to see when you could reach financial independence—and how likely you are to get there.

---

## What it does

- **Runs thousands of simulations** with random market returns, job changes, and health shocks
- **Tracks real-life details**: 401(k), Roth IRA, HSA, taxable accounts, home equity, taxes, and spending by category
- **Models family life**: marriage age, kids, spouse income (including part-time after kids), college savings (529s)
- **City-specific costs**: rent, home prices, property tax, and state taxes (Sacramento, Dublin, SF built-in; add your own)
- **Dynamic safe withdrawal**: retirement length drives withdrawal rate (e.g. ~2.5% for very long retirements, ~4% for 30 years)

You get **FIRE probability by age**, **net worth fan charts**, **cash flow Sankey diagrams**, and a **minimum income table** for “90% chance to FIRE by age X” by city.

---

## Quick start

```bash
# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run fire_dashboard.py
```

Then open the URL in your browser (usually `http://localhost:8501`).

---

## Project layout

| File | Purpose |
|------|--------|
| `fire.py` | Simulation engine: income, taxes, spending, accounts, FIRE logic |
| `fire_dashboard.py` | Streamlit UI: sliders, charts, tables, city editor |
| `requirements.txt` | Python dependencies |

---

## Dashboard at a glance

- **Net Worth Fan** — Percentile bands (P5–P95, P10–P90, P25–P75) and median over time; hover for dollar ranges
- **Cash Flow Sankey** — Where money goes (income → taxes, spending, savings) by life stage
- **Spending Deep Dive** — Housing, discretionary, kids, healthcare, education, one-time over time
- **Account Composition** — Taxable, 401(k), Roth, HSA, home equity over time
- **Income vs Spending** — Household income and total spending (and the gap = savings)
- **FIRE Histogram** — Distribution of ages at which you hit FIRE
- **Outcome Distributions** — Ending net worth, net worth at FIRE, spending at FIRE, years to FIRE
- **FIRE Probability** — CDF: “% FIRE’d by this age” (with vs without starting savings)
- **Sensitivity** — How ±20% in income/seed/spouse pay changes median FIRE age
- **Minimum TC table** — “What income do I need for 90% chance to FIRE by 40/42/…/55?” by city
- **City editor** — Add or edit cities (rent, home price, taxes, etc.)

---

## Requirements

- Python 3.x
- `numpy`, `streamlit`, `plotly`, `tabulate` (see `requirements.txt` for versions)

---

## Live app

**[fire-monte-carlo-sim.streamlit.app](https://fire-monte-carlo-sim.streamlit.app/)**

---

## License

Use and modify as you like. No warranty—this is for exploration and planning, not financial advice.
