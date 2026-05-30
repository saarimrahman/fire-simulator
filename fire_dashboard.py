"""
./venv/bin/streamlit run fire_dashboard.py
"""


import streamlit as st
import numpy as np
import plotly.graph_objects as go
from datetime import date

from fire import (
    run_vectorized, find_min_tc, SeedAmounts, FamilyConfig, CareerConfig, SocialSecurityConfig,
    SimulationResults, CITIES, CURRENT_AGE, FIRE_HORIZON, LIFE_EXPECTANCY, INFLATION, CityConfig, add_custom_cities,
    FOUR01K_LIMIT, ROTH_IRA_LIMIT, HSA_INDIVIDUAL_LIMIT, HSA_FAMILY_LIMIT,
    HEALTH_SHOCK_PROB, HEALTH_SHOCK_COST,
    OT_CAR_MOVE, OT_BABY_SETUP, OT_MID_UPGRADE, HOME_CLOSING_COSTS,
    HC_YOUNG, HC_OLDER, HC_RET_BASE, HC_RET_AGE_STEP, HC_RET_REAL_GROWTH, HC_MEDICARE,
    DISC_YOUNG, DISC_MID, DISC_FAMILY, DISC_STEP_35, DISC_STEP_40,
    JL_ANNUAL_PROB, JL_MONTHS_MIN, JL_MONTHS_MAX, JL_REENTRY_MIN, JL_REENTRY_MAX, JL_SEARCH_COST,
    MR_NORMAL_MEAN, MR_NORMAL_STD, MR_RECESSION_MEAN, MR_RECESSION_STD, RECESSION_PROB,
)

def _amt(n): return f"\\${n:,}"

def _qp(key, default, type_fn=int):
    """Read a query param, returning *default* if missing or unparseable."""
    raw = st.query_params.get(key)
    if raw is None:
        return default
    try:
        return type_fn(raw)
    except (ValueError, TypeError):
        return default

st.set_page_config(page_title="FIRE Simulator", page_icon="🔥", layout="wide")

st.title("FIRE Simulation Dashboard")

# =============================================================================
# SIDEBAR: Controls Panel
# =============================================================================
with st.sidebar:
    st.header("Simulation Settings")


    # Calculate age of someone born July 17, 2000
    birth_date = date(2000, 7, 17)
    today = date.today()
    default_age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))

    start_age = st.slider("Current Age", min_value=18, max_value=40, value=_qp("age", default_age))

    starting_tc = st.slider(
        "Starting Total Compensation ($)",
        min_value=100000, max_value=500000, value=_qp("tc", 200000), step=10000,
        format="$%d"
    )

    _sim_options = [1000, 2500, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000]
    _sim_default = _qp("sims", 10000)
    if _sim_default not in _sim_options:
        _sim_default = 10000
    n_sims = st.select_slider(
        "Number of Simulations",
        options=_sim_options,
        value=_sim_default,
        help="More simulations = more accurate but slower"
    )

    # Merge built-in cities with custom cities
    if 'custom_cities' not in st.session_state:
        st.session_state.custom_cities = {}

    # Add custom cities to the global CITIES dict
    if st.session_state.custom_cities:
        add_custom_cities(st.session_state.custom_cities)

    all_cities = {**CITIES, **st.session_state.custom_cities}

    _city_keys = list(all_cities.keys())
    _qp_city = st.query_params.get("city")
    _city_default = _city_keys.index(_qp_city) if _qp_city in _city_keys else (
        _city_keys.index("New York City") if "New York City" in _city_keys else 0
    )
    city = st.selectbox("City", _city_keys, index=_city_default)

    rent_override_enabled = st.checkbox("Override city rent", value=False)
    rent_override = None
    if rent_override_enabled:
        rent_override = st.number_input(
            "Monthly Rent ($)", min_value=0, max_value=20000, value=3000, step=100,
            help="Overrides all city rent tiers with this monthly amount"
        )

    with st.expander("Career Progression", expanded=False):
        _career_options = ["conservative", "moderate", "aggressive"]
        _qp_career = st.query_params.get("career", "moderate")
        if _qp_career not in _career_options:
            _qp_career = "moderate"
        career_trajectory = st.select_slider(
            "Career Trajectory",
            options=_career_options,
            value=_qp_career,
            help="Affects promotion probability and salary growth rate"
        )

        tc_soft_cap = st.slider(
            "TC Soft Cap ($)",
            min_value=400000, max_value=1200000, value=_qp("cap", 600000), step=50000,
            format="$%d",
            help="Salary growth tapers as you approach this ceiling (most ICs plateau $500-700k)"
        )

        st.caption(f"""
        **{career_trajectory.title()} trajectory:**
        - {'High' if career_trajectory == 'aggressive' else 'Moderate' if career_trajectory == 'moderate' else 'Lower'} promotion probability
        - Growth tapers near {_amt(int(tc_soft_cap/1000))}k cap
        """)

    with st.expander("Starting Balances", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            seed_taxable = st.number_input("Taxable ($)", 0, 1000000, _qp("seed_tax", 0), step=10000)
            seed_401k = st.number_input("401(k) ($)", 0, 1000000, _qp("seed_401k", 0), step=10000)
        with col2:
            seed_roth = st.number_input("Roth IRA ($)", 0, 500000, _qp("seed_roth", 0), step=1000)
            seed_hsa = st.number_input("HSA ($)", 0, 100000, _qp("seed_hsa", 0), step=1000)

        seed_total = seed_taxable + seed_401k + seed_roth + seed_hsa
        st.metric("Total Starting Seed", f"${seed_total:,.0f}")

    with st.expander("Account Contributions", expanded=False):
        st.caption("Toggle accounts on/off. Disabled contributions flow to taxable brokerage instead.")

        use_401k = st.toggle("401(k)",   value=bool(_qp("use_401k", 1)), help="Max employee contribution + employer match each year")
        if use_401k:
            employer_match_pct = st.slider(
                "Employer Match Rate (%)",
                min_value=0, max_value=100, value=50, step=5,
                format="%d%%",
                help="Employer contributes this % of your contribution (e.g. 50% = 50¢ per $1 you contribute)"
            ) / 100.0
            employer_match_limit = st.slider(
                "Employer Match Limit (% of IRS limit)",
                min_value=0, max_value=100, value=50, step=1,
                format="%d%%",
                help="Employer matches contributions up to this % of the IRS 401(k) employee limit. 50% ≈ $11,500/yr."
            ) / 100.0
        else:
            employer_match_pct = 0.0
            employer_match_limit = 0.0
        use_mega_backdoor = st.toggle(
            "Mega Backdoor Roth",
            value=bool(_qp("mega_backdoor", 0)) if use_401k else False,
            disabled=not use_401k,
            help="After-tax 401k contributions converted to Roth (up to the annual 401k total-additions limit minus pre-tax and employer match). Requires employer plan support and an enabled 401(k)."
        )
        mega_backdoor_cap = None
        if use_mega_backdoor:
            _mbc_default = int(_qp("mbc", 0))
            _mbc_val = st.slider(
                "Mega Backdoor Cap ($/yr)",
                min_value=0, max_value=46000, value=_mbc_default, step=1000,
                format="$%d",
                help="Cap your annual after-tax 401k contribution. $0 = no cap (use all available room, ~$34–46k depending on match). Set lower if cash flow is tight."
            )
            mega_backdoor_cap = int(_mbc_val) if _mbc_val > 0 else None
        use_roth = st.toggle("Roth IRA", value=bool(_qp("use_roth", 0)), help="Annual direct Roth IRA contributions, with MAGI phaseout now modeled.")
        use_hsa  = st.toggle("HSA",      value=bool(_qp("use_hsa", 1)),  help="Health Savings Account contributions (requires HDHP)")

        if use_hsa:
            hsa_annual_contrib = st.slider(
                "Annual HSA Contribution ($)",
                min_value=0, max_value=8550, value=8550, step=50,
                format="$%d",
                help="2025 IRS limits: $4,300 individual / $8,550 family."
            )
            hsa_employer_contrib = st.slider(
                "Employer HSA Contribution ($/yr)",
                min_value=0, max_value=2000, value=750, step=50,
                format="$%d",
                help="Annual employer HSA contribution (free money). Typical range $500–$1,500/yr."
            )
        else:
            hsa_annual_contrib = 0
            hsa_employer_contrib = 0

    with st.expander("Family", expanded=False):
        marriage_age = st.slider("Marriage Age", 25, 40, _qp("marriage", 29))

        num_kids = st.radio("Number of Kids", [0, 1, 2, 3], index=_qp("kids", 2), horizontal=True)
        kid_ages = ()
        wedding_budget = st.number_input(
            "Wedding Budget ($)", min_value=0, max_value=500000, value=60000, step=5000,
            help="One-time cost in the year after your marriage age"
        )

        if num_kids >= 1:
            kid1_age = st.slider("First Kid Born (Your Age)", 26, 45, 31)
            kid_ages = (kid1_age,)
            if num_kids >= 2:
                kid2_age = st.slider("Second Kid Born (Your Age)", kid1_age, 47, max(33, kid1_age + 2))
                kid_ages = (kid1_age, kid2_age)
                if num_kids == 3:
                    kid3_age = st.slider("Third Kid Born (Your Age)", kid2_age, 49, max(35, kid2_age + 2))
                    kid_ages = (kid1_age, kid2_age, kid3_age)

    with st.expander("Spouse Income", expanded=False):
        spouse_works = st.toggle("Spouse Works", value=bool(_qp("spouse", 1)))

        if spouse_works:
            spouse_salary = st.slider("Spouse Salary ($)", 0, 200000, _qp("spouse_sal", 80000), step=5000, format="$%d")
            spouse_soft_cap = st.slider("Spouse Salary Cap ($)", 100000, 300000, 150000, step=10000,
                                        format="$%d", help="Salary ceiling (growth tapers near this)")
            part_time_fraction = st.slider("Part-time Fraction", 0.0, 1.0, 0.5, step=0.1,
                                           help="After kids start school")
        else:
            spouse_salary = 0
            spouse_soft_cap = 150000
            part_time_fraction = 0.5

    with st.expander("Housing", expanded=False):
        cfg = all_cities[city]
        buy_home = st.toggle("Buy Home", value=cfg.home_price is not None,
                             disabled=cfg.home_price is None,
                             help="Not available in SF (no home price configured)")

    with st.expander("Social Security", expanded=False):
        ss_enabled = st.toggle("Include Social Security", value=True,
                               help="Model Social Security benefits starting at claiming age")

        if ss_enabled:
            ss_claiming_age = st.slider(
                "Your Claiming Age", min_value=62, max_value=70, value=67,
                help="FRA is 67. Earlier = reduced benefits (~6.7%/yr). Later = increased (~8%/yr)."
            )
            ss_monthly = st.slider(
                "Your Monthly Benefit at FRA ($)",
                min_value=0, max_value=5000, value=2500, step=100,
                format="$%d",
                help="Estimated monthly benefit at full retirement age (67). Check ssa.gov/myaccount."
            )
            ss_spouse_monthly = st.slider(
                "Spouse Monthly Benefit at FRA ($)",
                min_value=0, max_value=5000, value=1250, step=100,
                format="$%d",
                help="Spouse benefit (often 50% of primary earner's benefit)"
            )
            ss_spouse_claiming = st.slider(
                "Spouse Claiming Age", min_value=62, max_value=70, value=67
            )

            # Show estimated annual benefits
            ss_temp = SocialSecurityConfig(
                claiming_age=ss_claiming_age,
                monthly_benefit_today=ss_monthly,
                spouse_monthly_benefit=ss_spouse_monthly,
                spouse_claiming_age=ss_spouse_claiming
            )
            primary_annual = ss_temp.get_benefit_at_age(ss_claiming_age)
            spouse_annual = ss_temp.get_spouse_benefit_at_age(ss_spouse_claiming)
            st.caption(f"Estimated annual: **{_amt(int(primary_annual))}** (you) + **{_amt(int(spouse_annual))}** (spouse) = **{_amt(int(primary_annual + spouse_annual))}** total")
        else:
            ss_claiming_age = 67
            ss_monthly = 2500
            ss_spouse_monthly = 1250
            ss_spouse_claiming = 67

    with st.expander("Spending & Lifestyle", expanded=False):
        lifestyle_creep_pct = st.slider(
            "Lifestyle Creep (% of income)",
            min_value=0.0, max_value=20.0, value=2.5, step=0.5,
            format="%.1f%%",
            help="Extra discretionary spending as % of gross income. Tapers down as you approach your retirement horizon."
        ) / 100.0

    with st.expander("Retirement Planning", expanded=False):
        fire_horizon = st.slider(
            "Latest Retirement Age",
            min_value=50, max_value=75, value=_qp("fire_age", FIRE_HORIZON), step=1,
            help="If you haven't voluntarily FIRE'd by this age, the simulation forces retirement and you start drawing down"
        )
        life_expectancy = st.slider(
            "Life Expectancy",
            min_value=75, max_value=100, value=_qp("life_exp", LIFE_EXPECTANCY), step=1,
            help="Simulation runs through this age to check if portfolio survives"
        )
        custom_swr_enabled = st.toggle(
            "Override Safe Withdrawal Rate",
            value=st.query_params.get("swr") is not None,
            help="Use a fixed SWR everywhere instead of the simulator's age-based default."
        )
        if custom_swr_enabled:
            swr_override = st.slider(
                "Custom SWR (%)",
                min_value=2.5, max_value=6.0, value=float(_qp("swr", 4.0, float)), step=0.1,
                format="%.1f%%",
                help="Applies a fixed withdrawal rate to the FIRE target instead of the age-based auto SWR."
            ) / 100.0
        else:
            swr_override = None

    with st.expander("Tax Settings", expanded=False):
        tax_adjust = st.toggle(
            "Show after-tax portfolio values",
            value=True,
            help="Discounts 401k by an estimated effective income-tax rate and taxable gains by LTCG rate. Roth and HSA are already tax-free."
        )
        if tax_adjust:
            eff_ret_tax_rate = st.slider(
                "Effective 401k withdrawal tax rate (%)",
                min_value=5, max_value=40, value=20, step=1,
                format="%d%%",
                help="Estimated blended federal + state income-tax rate on 401k withdrawals in retirement. 20% is a reasonable middle estimate for a large 401k."
            ) / 100.0
            ltcg_rate = st.slider(
                "LTCG rate on taxable gains (%)",
                min_value=0, max_value=25, value=15, step=1,
                format="%d%%",
                help="Federal long-term capital gains rate (15% for most; 20% if income is high in retirement)."
            ) / 100.0
        else:
            eff_ret_tax_rate = 0.20
            ltcg_rate = 0.15

# =============================================================================
# RUN SIMULATION
# =============================================================================
seed_amounts = SeedAmounts(
    taxable=seed_taxable,
    t401k=seed_401k,
    roth=seed_roth,
    hsa=seed_hsa
)

family_config = FamilyConfig(
    marriage_age=marriage_age,
    kid_ages=kid_ages,
    spouse_works=spouse_works,
    spouse_salary=spouse_salary,
    spouse_soft_cap=spouse_soft_cap,
    part_time_fraction=part_time_fraction,
)

@st.cache_data
def run_sim(starting_tc, city, n_sims, seed_taxable, seed_401k, seed_roth, seed_hsa,
            marriage_age, kid_ages, spouse_works, spouse_salary, spouse_soft_cap, part_time_fraction, life_exp,
            career_trajectory, tc_soft_cap, employer_match_pct, employer_match_limit, current_age,
            ss_enabled, ss_claiming_age, ss_monthly, ss_spouse_monthly, ss_spouse_claiming,
            hsa_annual_contrib=None, hsa_employer_contrib=0, rent_override=None,
            lifestyle_creep_pct=0.025, wedding_budget=60000, fire_horizon=60,
            use_401k=True, use_roth=True, use_hsa=True, use_mega_backdoor=False,
            mega_backdoor_cap=None, swr_override=None):
    seed = SeedAmounts(taxable=seed_taxable, t401k=seed_401k, roth=seed_roth, hsa=seed_hsa)
    family = FamilyConfig(
        marriage_age=marriage_age, kid_ages=kid_ages, spouse_works=spouse_works,
        spouse_salary=spouse_salary, spouse_soft_cap=spouse_soft_cap, part_time_fraction=part_time_fraction,
        wedding_budget=wedding_budget,
    )
    career = CareerConfig(
        soft_cap=tc_soft_cap, trajectory=career_trajectory,
        employer_match_pct=employer_match_pct, employer_match_limit=employer_match_limit
    )
    ss = SocialSecurityConfig(
        enabled=ss_enabled, claiming_age=ss_claiming_age,
        monthly_benefit_today=ss_monthly, spouse_monthly_benefit=ss_spouse_monthly,
        spouse_claiming_age=ss_spouse_claiming
    )
    rng = np.random.default_rng(42)
    return run_vectorized(starting_tc, city, n_sims, rng, seed_amounts=seed,
                          family_config=family, career_config=career, ss_config=ss,
                          return_trajectories=True, life_expectancy=life_exp,
                          current_age=current_age, fire_horizon=fire_horizon,
                          hsa_annual_contrib=hsa_annual_contrib,
                          hsa_employer_contrib=hsa_employer_contrib,
                          rent_override=rent_override, lifestyle_creep_pct=lifestyle_creep_pct,
                          use_401k=use_401k, use_roth=use_roth, use_hsa=use_hsa,
                          use_mega_backdoor=use_mega_backdoor, mega_backdoor_cap=mega_backdoor_cap,
                          swr_override=swr_override)

with st.spinner("Running simulation..."):
    results: SimulationResults = run_sim(
        starting_tc, city, n_sims, seed_taxable, seed_401k, seed_roth, seed_hsa,
        marriage_age, kid_ages, spouse_works, spouse_salary, spouse_soft_cap, part_time_fraction, life_expectancy,
        career_trajectory, tc_soft_cap, employer_match_pct, employer_match_limit, start_age,
        ss_enabled, ss_claiming_age, ss_monthly, ss_spouse_monthly, ss_spouse_claiming,
        hsa_annual_contrib=hsa_annual_contrib, hsa_employer_contrib=hsa_employer_contrib,
        rent_override=rent_override, lifestyle_creep_pct=lifestyle_creep_pct,
        wedding_budget=wedding_budget, fire_horizon=fire_horizon,
        use_401k=use_401k, use_roth=use_roth, use_hsa=use_hsa,
        use_mega_backdoor=use_mega_backdoor, mega_backdoor_cap=mega_backdoor_cap,
        swr_override=swr_override
    )

# Sync current settings to URL query params (shareable link)
_params = {
    "tc": starting_tc, "city": city, "age": start_age, "sims": n_sims,
    "career": career_trajectory, "cap": tc_soft_cap,
    "fire_age": fire_horizon, "life_exp": life_expectancy,
    "kids": num_kids, "marriage": marriage_age, "spouse": int(spouse_works),
    "spouse_sal": spouse_salary,
}
if seed_taxable: _params["seed_tax"] = seed_taxable
if seed_401k:    _params["seed_401k"] = seed_401k
if seed_roth:    _params["seed_roth"] = seed_roth
if seed_hsa:     _params["seed_hsa"] = seed_hsa
if not use_401k: _params["use_401k"] = 0
if use_mega_backdoor: _params["mega_backdoor"] = 1
if mega_backdoor_cap is not None: _params["mbc"] = mega_backdoor_cap
if not use_roth: _params["use_roth"] = 0
if not use_hsa:  _params["use_hsa"] = 0
if swr_override is not None: _params["swr"] = round(swr_override * 100, 1)
st.query_params.update(_params)
if swr_override is None and "swr" in st.query_params:
    del st.query_params["swr"]

# =============================================================================
# KEY METRICS
# =============================================================================
FIRE_HORIZON_AGE = fire_horizon

# Voluntary FIRE = achieved FIRE before forced retirement at the selected horizon
voluntary_fire = results.fire_ages[results.fire_ages < FIRE_HORIZON_AGE]
all_retired = results.fire_ages[results.fire_ages < 99]
fire_ages_valid = voluntary_fire  # For histograms and percentile calculations
pct_fire = len(voluntary_fire) / len(results.fire_ages) * 100
median_fire_age = np.median(voluntary_fire) if len(voluntary_fire) > 0 else None
p10_fire_age = np.percentile(voluntary_fire, 10) if len(voluntary_fire) > 0 else None
p90_fire_age = np.percentile(voluntary_fire, 90) if len(voluntary_fire) > 0 else None

# Success = portfolio survived to life expectancy (everyone retires by 60)
n_sims_total = len(results.fire_ages)
n_survived = (~results.failed).sum()
pct_success = n_survived / n_sims_total * 100
pct_failed = results.failed.mean() * 100

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("FIRE Rate", f"{pct_fire:.1f}%",
              help=f"% of simulations that achieved voluntary FIRE before age {fire_horizon}")
with col2:
    st.metric("Success Rate", f"{pct_success:.1f}%",
              help=f"Portfolio survived to age {life_expectancy} (everyone retires by age {fire_horizon})")
with col3:
    st.metric("Median FIRE Age", f"{median_fire_age:.0f}" if median_fire_age else "N/A")
with col4:
    st.metric("P10 FIRE Age", f"{p10_fire_age:.0f}" if p10_fire_age else "N/A")
with col5:
    st.metric("P90 FIRE Age", f"{p90_fire_age:.0f}" if p90_fire_age else "N/A")

# FIRE number countdown row
current_nw = float(seed_taxable + seed_401k + seed_roth + seed_hsa)
if len(voluntary_fire) > 0:
    _fired_mask = results.fire_ages < FIRE_HORIZON_AGE
    # Principled FIRE target: required nest egg in today's dollars (not observed NW at FIRE).
    # Uses fixed retirement spending estimates / SWR, independent of income trajectory.
    fire_target = float(np.median(results.fire_number[_fired_mask]))
    fire_ret_spend = float(np.median(results.fire_spending[_fired_mask]))
    gap = fire_target - current_nw

    def _fmt(v):
        return f"${v/1e6:.2f}M" if abs(v) >= 1e6 else f"${v:,.0f}"

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("FIRE Target", _fmt(fire_target),
                  help="Required nest egg in today's dollars — based on estimated retirement "
                       "spending (housing + $45k discretionary + $24k healthcare + kids) "
                       "divided by your safe withdrawal rate. Independent of income.")
    with col2:
        st.metric("Retirement Spending", f"{_fmt(fire_ret_spend)}/yr",
                  help="Estimated annual retirement spending in today's dollars "
                       "(housing, discretionary, healthcare, kids, 529)")
    with col3:
        st.metric("Current Net Worth", _fmt(current_nw),
                  help="Starting net worth from your seed amounts")
    with col4:
        st.metric("Gap to FIRE", _fmt(gap) if gap > 0 else "Already there",
                  help="How much more you need in today's dollars to reach your FIRE number")

if pct_failed > 0:
    with st.expander(f"⚠️ {pct_failed:.1f}% of simulations ran out of money before age {life_expectancy}"):
        st.markdown(f"""
Everyone retires by age {fire_horizon} (voluntarily or forced by the simulation). A simulation "fails" when the portfolio
hits zero before age {life_expectancy}.

**Common causes:**
- Insufficient savings accumulated by retirement
- Bad sequence of returns early in retirement
- Inflation spikes eroding purchasing power
- Unexpected medical costs

**To improve the success rate:**
- Increase starting TC or savings rate
- Enable Social Security benefits (sidebar)
- Consider a less expensive city
- Switch to an aggressive career trajectory
        """)

# =============================================================================
# HELPERS
# =============================================================================

def _after_tax_nw(taxable, taxable_basis, t401k, roth, hsa, home_equity,
                   ret_tax_rate, ltcg_rate):
    """Convert nominal account balances to after-tax real values.

    taxable:       (N_YEARS, N) or (N,) taxable brokerage balance
    taxable_basis: matching cost-basis array
    t401k:         401k / traditional pre-tax balance
    roth:          Roth balance (already after-tax)
    hsa:           HSA balance (treated as after-tax)
    home_equity:   illiquid, not discounted
    ret_tax_rate:  effective income-tax rate on 401k withdrawals
    ltcg_rate:     long-term capital gains rate on taxable gains
    """
    gains = np.maximum(taxable - taxable_basis, 0)
    at_taxable  = taxable  - gains * ltcg_rate
    at_401k     = t401k    * (1 - ret_tax_rate)
    return at_taxable + at_401k + roth + hsa + home_equity

# =============================================================================
# CHARTS
# =============================================================================
tab1, tab2, tab3, tab4, tab6, tab7, tab8 = st.tabs([
    "Net Worth Fan", "Cash Flow",
    "Account Composition", "Income vs Spending",
    "Outcome Distributions", "FIRE Probability", "Sensitivity"
])

# Chart 1: Net Worth Fan Chart
with tab1:
    st.subheader("Net Worth Trajectory")

    real_terms = st.toggle("Inflation-adjusted (today's dollars)", value=True,
                           help="Divides by cumulative inflation to show real purchasing power")

    ages = results.ages
    if tax_adjust:
        nw = _after_tax_nw(results.taxable, results.taxable_basis, results.t401k,
                           results.roth, results.hsa, results.home_equity,
                           eff_ret_tax_rate, ltcg_rate)
    else:
        nw = results.net_worth
    if real_terms:
        infl_factors = (1 + INFLATION) ** (ages - start_age)
        nw = nw / infl_factors[:, np.newaxis]

    median = np.median(nw, axis=1)
    p5 = np.percentile(nw, 5, axis=1)
    p10 = np.percentile(nw, 10, axis=1)
    p25 = np.percentile(nw, 25, axis=1)
    p75 = np.percentile(nw, 75, axis=1)
    p90 = np.percentile(nw, 90, axis=1)
    p95 = np.percentile(nw, 95, axis=1)

    fig = go.Figure()

    # Add invisible traces for ranges to show in hover
    fig.add_trace(go.Scatter(
        x=ages, y=(p5 + p95) / 2, mode='lines',
        line=dict(width=0), showlegend=False,
        customdata=np.column_stack([p5, p95]),
        hovertemplate='P5-P95: $%{customdata[0]:,.0f} - $%{customdata[1]:,.0f}<extra></extra>',
        name='P5-P95'
    ))
    fig.add_trace(go.Scatter(
        x=ages, y=(p10 + p90) / 2, mode='lines',
        line=dict(width=0), showlegend=False,
        customdata=np.column_stack([p10, p90]),
        hovertemplate='P10-P90: $%{customdata[0]:,.0f} - $%{customdata[1]:,.0f}<extra></extra>',
        name='P10-P90'
    ))
    fig.add_trace(go.Scatter(
        x=ages, y=(p25 + p75) / 2, mode='lines',
        line=dict(width=0), showlegend=False,
        customdata=np.column_stack([p25, p75]),
        hovertemplate='P25-P75: $%{customdata[0]:,.0f} - $%{customdata[1]:,.0f}<extra></extra>',
        name='P25-P75'
    ))

    # 5-95 band
    fig.add_trace(go.Scatter(
        x=np.concatenate([ages, ages[::-1]]),
        y=np.concatenate([p95, p5[::-1]]),
        fill='toself', fillcolor='rgba(99, 110, 250, 0.1)',
        line=dict(color='rgba(255,255,255,0)'),
        name='P5-P95', showlegend=True,
        hoverinfo='skip'
    ))

    # 10-90 band
    fig.add_trace(go.Scatter(
        x=np.concatenate([ages, ages[::-1]]),
        y=np.concatenate([p90, p10[::-1]]),
        fill='toself', fillcolor='rgba(99, 110, 250, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='P10-P90', showlegend=True,
        hoverinfo='skip'
    ))

    # 25-75 band
    fig.add_trace(go.Scatter(
        x=np.concatenate([ages, ages[::-1]]),
        y=np.concatenate([p75, p25[::-1]]),
        fill='toself', fillcolor='rgba(99, 110, 250, 0.3)',
        line=dict(color='rgba(255,255,255,0)'),
        name='P25-P75', showlegend=True,
        hoverinfo='skip'
    ))

    # Median line
    fig.add_trace(go.Scatter(
        x=ages, y=median,
        mode='lines', line=dict(color='#636EFA', width=3),
        name='Median',
        hovertemplate='Median: $%{y:,.0f}<extra></extra>'
    ))

    # Mark median FIRE age
    if median_fire_age:
        fig.add_vline(x=median_fire_age, line_dash="dash", line_color="green",
                      annotation_text=f"Median FIRE: {median_fire_age:.0f}")

    fig.update_layout(
        xaxis_title="Age",
        yaxis_title="Net Worth",
        yaxis=dict(tickformat='$,.0f'),
        hovermode='x unified',
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

# Chart 2: Cash Flow Sankey Diagram
with tab2:
    st.subheader("Spending by Age")

    cf_col1, cf_col2 = st.columns([2, 3])
    with cf_col1:
        view = st.radio("Display", ["Annual", "Monthly"], horizontal=True, key="spend_view")
    with cf_col2:
        real_terms_cf = st.toggle("Inflation-adjusted (today's dollars)", value=True,
                                  help="Divides by cumulative inflation to show real purchasing power", key="real_cf")
    div = 12 if view == "Monthly" else 1
    unit = "/mo" if view == "Monthly" else "/yr"

    ages = results.ages
    infl_deflator = (1 + INFLATION) ** (ages - start_age)  # shape (n_years,)

    def _deflate(arr):
        """Divide by inflation factor per age if real_terms_cf is on."""
        return arr / infl_deflator if real_terms_cf else arr

    housing_med = _deflate(np.median(results.spending_housing,       axis=1)) / div
    disc_med    = _deflate(np.median(results.spending_discretionary, axis=1)) / div
    hc_med      = _deflate(np.median(results.spending_healthcare,    axis=1)) / div
    kids_med    = _deflate(np.median(results.spending_kids,          axis=1)) / div
    edu_med     = _deflate(np.median(results.spending_education,     axis=1)) / div
    ot_med      = _deflate(np.median(results.spending_one_time,      axis=1)) / div
    tax_med     = _deflate(np.median(results.taxes,                  axis=1)) / div
    total_med   = housing_med + disc_med + hc_med + kids_med + edu_med + ot_med + tax_med

    spend_categories = [
        ("Taxes",           tax_med,     "#FF6D00"),
        ("Housing",         housing_med, "#2196F3"),
        ("Discretionary",   disc_med,    "#FF9800"),
        ("Healthcare",      hc_med,      "#F44336"),
        ("Kids",            kids_med,    "#4CAF50"),
        ("Education (529)", edu_med,     "#00BCD4"),
        ("One-Time",        ot_med,      "#9C27B0"),
    ]

    fig = go.Figure()
    for name, vals, color in spend_categories:
        fig.add_trace(go.Bar(
            x=ages, y=vals, name=name,
            marker_color=color,
            hovertemplate=f'Age %{{x}}<br>{name}: $%{{y:,.0f}}{unit}<extra></extra>'
        ))

    if median_fire_age:
        fig.add_vline(x=median_fire_age, line_dash="dash", line_color="green",
                      annotation_text=f"Median FIRE: {median_fire_age:.0f}")

    fig.update_layout(
        barmode='stack',
        xaxis_title="Age",
        yaxis_title=f"Spending ({unit}{', today\'s $' if real_terms_cf else ', nominal $'})",
        yaxis=dict(tickformat='$,.0f'),
        hovermode='x unified',
        height=500,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Data table
    s401k_med  = _deflate(np.median(results.savings_401k,    axis=1)) / div
    sroth_med  = _deflate(np.median(results.savings_roth,    axis=1)) / div
    shsa_med   = _deflate(np.median(results.savings_hsa,     axis=1)) / div
    stax_med   = _deflate(np.median(results.savings_taxable, axis=1)) / div
    saved_med  = s401k_med + sroth_med + shsa_med + stax_med

    income_med_cf = _deflate(np.median(results.incomes, axis=1)) / div

    _dash = "—"
    _cf_dollar_note = "today's dollars" if real_terms_cf else "nominal dollars"
    st.caption(f"Median spending and savings by age ({_cf_dollar_note}). Free Cash Flow = Earned − Total Spend (working years only). Small differences from Total Saved are due to median aggregation across simulations.")

    sample_idx = list(range(len(ages)))

    _fire_age = int(median_fire_age) if median_fire_age else None

    def _row_label(age):
        if _fire_age and age == _fire_age:
            return f"{age} ★ FIRE"
        if _fire_age and age > _fire_age:
            return f"{age} (retired)"
        return str(age)

    rows = []
    for i in sample_idx:
        age = int(ages[i])
        free_cf = income_med_cf[i] - total_med[i]
        is_retired = income_med_cf[i] == 0
        rows.append({
            "Age":           _row_label(age),
            "Housing":       f"${housing_med[i]:,.0f}",
            "Discretionary": f"${disc_med[i]:,.0f}",
            "Healthcare":    f"${hc_med[i]:,.0f}",
            "Kids":          f"${kids_med[i]:,.0f}" if kids_med[i] > 0 else _dash,
            "Education":     f"${edu_med[i]:,.0f}" if edu_med[i] > 0 else _dash,
            "One-Time":      f"${ot_med[i]:,.0f}" if ot_med[i] > 0 else _dash,
            "Earned":        f"${income_med_cf[i]:,.0f}",
            "Taxes":         f"${tax_med[i]:,.0f}",
            "Total Spend":   f"${total_med[i]:,.0f}",
            "401(k)":        f"${s401k_med[i]:,.0f}" if s401k_med[i] > 0 else _dash,
            "Roth IRA":      f"${sroth_med[i]:,.0f}" if sroth_med[i] > 0 else _dash,
            "HSA":           f"${shsa_med[i]:,.0f}" if shsa_med[i] > 0 else _dash,
            "Brokerage":     f"${stax_med[i]:,.0f}" if stax_med[i] > 0 else _dash,
            "Total Saved":   f"${saved_med[i]:,.0f}",
            "Free Cash Flow": _dash if is_retired else (f"${free_cf:,.0f}" if free_cf >= 0 else f"-${abs(free_cf):,.0f}"),
            "Savings %":     f"{saved_med[i] / income_med_cf[i] * 100:.0f}%" if income_med_cf[i] > 0 else _dash,
        })

    # Single collapsed totals row
    _lifetime_savings_pct = saved_med.sum() / income_med_cf.sum() * 100 if income_med_cf.sum() > 0 else 0
    _working_mask = income_med_cf > 0
    _total_net_cf = (income_med_cf - total_med)[_working_mask].sum()
    rows.append({
        "Age":           "Total",
        "Earned":        f"${income_med_cf.sum():,.0f}",
        "Housing":       f"${housing_med.sum():,.0f}",
        "Discretionary": f"${disc_med.sum():,.0f}",
        "Healthcare":    f"${hc_med.sum():,.0f}",
        "Kids":          f"${kids_med.sum():,.0f}",
        "Education":     f"${edu_med.sum():,.0f}",
        "One-Time":      f"${ot_med.sum():,.0f}",
        "Taxes":         f"${tax_med.sum():,.0f}",
        "Total Spend":   f"${total_med.sum():,.0f}",
        "401(k)":        f"${s401k_med.sum():,.0f}",
        "Roth IRA":      f"${sroth_med.sum():,.0f}",
        "HSA":           f"${shsa_med.sum():,.0f}",
        "Brokerage":     f"${stax_med.sum():,.0f}",
        "Total Saved":   f"${saved_med.sum():,.0f}",
        "Free Cash Flow": f"${_total_net_cf:,.0f}" if _total_net_cf >= 0 else f"-${abs(_total_net_cf):,.0f}",
        "Savings %":     f"{_lifetime_savings_pct:.0f}%",
    })

    import pandas as pd
    df_cf = pd.DataFrame(rows)

    # Style: highlight FIRE row in green, retired rows in light blue, totals row in gray
    def _style_cf_row(row):
        label = str(row["Age"])
        if "★ FIRE" in label:
            return ["background-color: #1e2e1e"] * len(row)
        if "(retired)" in label:
            return ["background-color: #1a2030"] * len(row)
        if label == "Total":
            return ["background-color: #252525; font-weight: bold"] * len(row)
        return [""] * len(row)

    styled_df = df_cf.style.apply(_style_cf_row, axis=1)
    st.dataframe(styled_df, use_container_width=True, hide_index=True)

# Chart 5: FIRE Age Histogram
with tab6:
    st.subheader("Outcome Distributions")

    col_sel, col_tog = st.columns([3, 2])
    with col_sel:
        outcome_selector = st.selectbox(
            "Select Metric",
            ["Ending Net Worth (at life expectancy)", "Net Worth at FIRE", "Spending at FIRE",
             "FIRE Age Distribution", "Years Until FIRE"],
            help="Choose which outcome to visualize"
        )
    with col_tog:
        real_terms_t6 = st.toggle("Inflation-adjusted (today's dollars)", value=True,
                                  help="Deflates dollar values to today's purchasing power", key="real_t6")

    if outcome_selector == "Ending Net Worth (at life expectancy)":
        if tax_adjust:
            ending_nw = _after_tax_nw(results.taxable[-1], results.taxable_basis[-1],
                                      results.t401k[-1], results.roth[-1],
                                      results.hsa[-1], results.home_equity[-1],
                                      eff_ret_tax_rate, ltcg_rate)
        else:
            ending_nw = results.net_worth[-1, :]
        if real_terms_t6:
            ending_nw = ending_nw / (1 + INFLATION) ** (life_expectancy - start_age)
        valid_data = ending_nw[ending_nw > 0]  # Exclude failures

        # Trim extreme outliers for better visualization
        p1 = np.percentile(valid_data, 1)
        p99 = np.percentile(valid_data, 99)
        trimmed_data = valid_data[(valid_data >= p1) & (valid_data <= p99)]

        fig = go.Figure()

        # Calculate appropriate bin size (around 100-200k increments)
        data_range = trimmed_data.max() - trimmed_data.min()
        bin_size = max(100000, data_range / 50)  # At least 100k bins, or 50 bins total

        fig.add_trace(go.Histogram(
            x=trimmed_data,
            xbins=dict(
                start=trimmed_data.min(),
                end=trimmed_data.max(),
                size=bin_size
            ),
            marker_color='#636EFA',
            name='Ending Net Worth'
        ))

        # Add percentile markers
        p10 = np.percentile(valid_data, 10)
        p50 = np.percentile(valid_data, 50)
        p90 = np.percentile(valid_data, 90)

        # Only add markers if they're within the visible range
        if p1 <= p10 <= p99:
            fig.add_vline(x=p10, line_dash="dash", line_color="red",
                          annotation_text=f"P10: ${p10:,.0f}", annotation_position="top")
        if p1 <= p50 <= p99:
            fig.add_vline(x=p50, line_dash="solid", line_color="green",
                          annotation_text=f"P50: ${p50:,.0f}", annotation_position="top")
        if p1 <= p90 <= p99:
            fig.add_vline(x=p90, line_dash="dash", line_color="blue",
                          annotation_text=f"P90: ${p90:,.0f}", annotation_position="top")

        fig.update_layout(
            xaxis_title=f"Net Worth at Age {life_expectancy} {'(today\'s $)' if real_terms_t6 else '(nominal)'}{'· after-tax' if tax_adjust else ''} — P1-P99 shown",
            xaxis=dict(tickformat='$,.0f'),
            yaxis_title="Count",
            height=500
        )

        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean", f"${np.mean(valid_data):,.0f}")
        with col2:
            st.metric("Median", f"${p50:,.0f}")
        with col3:
            st.metric("Std Dev", f"${np.std(valid_data):,.0f}")
        with col4:
            pct_survived = len(valid_data) / len(ending_nw) * 100
            st.metric("Survived", f"{pct_survived:.1f}%")

        st.caption(f"📊 Showing P1-P99 range ({_amt(int(p1))} – {_amt(int(p99))}) to focus on main distribution. Excludes {len(valid_data) - len(trimmed_data)} extreme outliers.")
        st.plotly_chart(fig, use_container_width=True)

    elif outcome_selector == "Net Worth at FIRE":
        # Get net worth at FIRE age for each simulation
        fire_nw = []
        for sim_idx in range(results.net_worth.shape[1]):
            fire_age = results.fire_ages[sim_idx]
            if fire_age < 99:  # Only include those who FIRE'd
                age_idx = np.where(results.ages == fire_age)[0]
                if len(age_idx) > 0:
                    ai = age_idx[0]
                    if tax_adjust:
                        val = _after_tax_nw(
                            results.taxable[ai, sim_idx], results.taxable_basis[ai, sim_idx],
                            results.t401k[ai, sim_idx], results.roth[ai, sim_idx],
                            results.hsa[ai, sim_idx], results.home_equity[ai, sim_idx],
                            eff_ret_tax_rate, ltcg_rate)
                    else:
                        val = results.net_worth[ai, sim_idx]
                    if real_terms_t6:
                        val = val / (1 + INFLATION) ** (fire_age - start_age)
                    fire_nw.append(val)

        fire_nw = np.array(fire_nw)

        if len(fire_nw) > 0:
            # Trim extreme outliers for better visualization
            p1 = np.percentile(fire_nw, 1)
            p99 = np.percentile(fire_nw, 99)
            trimmed_data = fire_nw[(fire_nw >= p1) & (fire_nw <= p99)]

            fig = go.Figure()

            # Calculate appropriate bin size
            data_range = trimmed_data.max() - trimmed_data.min()
            bin_size = max(100000, data_range / 50)  # At least 100k bins, or 50 bins total

            fig.add_trace(go.Histogram(
                x=trimmed_data,
                xbins=dict(
                    start=trimmed_data.min(),
                    end=trimmed_data.max(),
                    size=bin_size
                ),
                marker_color='#00CC96',
                name='Net Worth at FIRE'
            ))

            # Add percentile markers
            p10 = np.percentile(fire_nw, 10)
            p50 = np.percentile(fire_nw, 50)
            p90 = np.percentile(fire_nw, 90)

            # Only add markers if they're within the visible range
            if p1 <= p10 <= p99:
                fig.add_vline(x=p10, line_dash="dash", line_color="red",
                              annotation_text=f"P10: ${p10:,.0f}", annotation_position="top")
            if p1 <= p50 <= p99:
                fig.add_vline(x=p50, line_dash="solid", line_color="green",
                              annotation_text=f"P50: ${p50:,.0f}", annotation_position="top")
            if p1 <= p90 <= p99:
                fig.add_vline(x=p90, line_dash="dash", line_color="blue",
                              annotation_text=f"P90: ${p90:,.0f}", annotation_position="top")

            fig.update_layout(
                xaxis_title=f"Net Worth at FIRE {'(today\'s $)' if real_terms_t6 else '(nominal)'}{'· after-tax' if tax_adjust else ''} — P1-P99 shown",
                xaxis=dict(tickformat='$,.0f'),
                yaxis_title="Count",
                height=500
            )

            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean", f"${np.mean(fire_nw):,.0f}")
            with col2:
                st.metric("Median", f"${p50:,.0f}")
            with col3:
                st.metric("Std Dev", f"${np.std(fire_nw):,.0f}")
            with col4:
                st.metric("N Simulations", f"{len(fire_nw)}")

            st.caption(f"📊 Showing P1-P99 range ({_amt(int(p1))} – {_amt(int(p99))}) to focus on main distribution. Excludes {len(fire_nw) - len(trimmed_data)} extreme outliers.")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No simulations reached FIRE")

    elif outcome_selector == "Spending at FIRE":
        # Get spending at FIRE age for each simulation
        fire_spending = []
        for sim_idx in range(results.spending.shape[1]):
            fire_age = results.fire_ages[sim_idx]
            if fire_age < 99:  # Only include those who FIRE'd
                age_idx = np.where(results.ages == fire_age)[0]
                if len(age_idx) > 0:
                    val = results.spending[age_idx[0], sim_idx]
                    if real_terms_t6:
                        val = val / (1 + INFLATION) ** (fire_age - start_age)
                    fire_spending.append(val)

        fire_spending = np.array(fire_spending)

        if len(fire_spending) > 0:
            # Trim extreme outliers for better visualization
            p1 = np.percentile(fire_spending, 1)
            p99 = np.percentile(fire_spending, 99)
            trimmed_data = fire_spending[(fire_spending >= p1) & (fire_spending <= p99)]

            fig = go.Figure()

            # Calculate appropriate bin size
            data_range = trimmed_data.max() - trimmed_data.min()
            bin_size = max(5000, data_range / 50)  # At least 5k bins, or 50 bins total

            fig.add_trace(go.Histogram(
                x=trimmed_data,
                xbins=dict(
                    start=trimmed_data.min(),
                    end=trimmed_data.max(),
                    size=bin_size
                ),
                marker_color='#EF553B',
                name='Spending at FIRE'
            ))

            # Add percentile markers
            p10 = np.percentile(fire_spending, 10)
            p50 = np.percentile(fire_spending, 50)
            p90 = np.percentile(fire_spending, 90)

            # Only add markers if they're within the visible range
            if p1 <= p10 <= p99:
                fig.add_vline(x=p10, line_dash="dash", line_color="red",
                              annotation_text=f"P10: ${p10:,.0f}", annotation_position="top")
            if p1 <= p50 <= p99:
                fig.add_vline(x=p50, line_dash="solid", line_color="green",
                              annotation_text=f"P50: ${p50:,.0f}", annotation_position="top")
            if p1 <= p90 <= p99:
                fig.add_vline(x=p90, line_dash="dash", line_color="blue",
                              annotation_text=f"P90: ${p90:,.0f}", annotation_position="top")

            fig.update_layout(
                xaxis_title=f"Annual Spending at FIRE {'(today\'s $)' if real_terms_t6 else '(nominal)'} — P1-P99 shown",
                xaxis=dict(tickformat='$,.0f'),
                yaxis_title="Count",
                height=500
            )

            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean", f"${np.mean(fire_spending):,.0f}")
            with col2:
                st.metric("Median", f"${p50:,.0f}")
            with col3:
                st.metric("Std Dev", f"${np.std(fire_spending):,.0f}")
            with col4:
                st.metric("N Simulations", f"{len(fire_spending)}")

            st.caption(f"📊 Showing P1-P99 range ({_amt(int(p1))} – {_amt(int(p99))}) to focus on main distribution. Excludes {len(fire_spending) - len(trimmed_data)} extreme outliers.")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No simulations reached FIRE")

    elif outcome_selector == "FIRE Age Distribution":
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=fire_ages_valid,
            nbinsx=30,
            name='FIRE Age',
            marker_color='#636EFA'
        ))

        if median_fire_age:
            fig.add_vline(x=median_fire_age, line_dash="dash", line_color="red",
                          annotation_text=f"Median: {median_fire_age:.0f}")

        never_fire_pct = 100 - pct_fire
        fig.update_layout(
            xaxis_title="FIRE Age",
            yaxis_title="Count",
            height=500,
            annotations=[
                dict(
                    x=0.95, y=0.95, xref="paper", yref="paper",
                    text=f"Never FIRE: {never_fire_pct:.1f}%",
                    showarrow=False, font=dict(size=14),
                    bgcolor="rgba(255,255,255,0.8)"
                )
            ]
        )

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Median", f"{median_fire_age:.0f}" if median_fire_age else "N/A")
        with col2:
            st.metric("P10", f"{p10_fire_age:.0f}" if p10_fire_age else "N/A")
        with col3:
            st.metric("P90", f"{p90_fire_age:.0f}" if p90_fire_age else "N/A")
        with col4:
            st.metric("FIRE Rate", f"{pct_fire:.1f}%")

        st.plotly_chart(fig, use_container_width=True)

    elif outcome_selector == "Years Until FIRE":
        years_to_fire = fire_ages_valid - start_age

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=years_to_fire,
            nbinsx=30,
            marker_color='#AB63FA',
            name='Years Until FIRE'
        ))

        # Add percentile markers
        p10 = np.percentile(years_to_fire, 10)
        p50 = np.percentile(years_to_fire, 50)
        p90 = np.percentile(years_to_fire, 90)

        fig.add_vline(x=p10, line_dash="dash", line_color="red",
                      annotation_text=f"P10: {p10:.1f} yrs", annotation_position="top")
        fig.add_vline(x=p50, line_dash="solid", line_color="green",
                      annotation_text=f"P50: {p50:.1f} yrs", annotation_position="top")
        fig.add_vline(x=p90, line_dash="dash", line_color="blue",
                      annotation_text=f"P90: {p90:.1f} yrs", annotation_position="top")

        fig.update_layout(
            xaxis_title="Years Until FIRE",
            yaxis_title="Count",
            height=500
        )

        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean", f"{np.mean(years_to_fire):.1f} yrs")
        with col2:
            st.metric("Median", f"{p50:.1f} yrs")
        with col3:
            st.metric("Std Dev", f"{np.std(years_to_fire):.1f} yrs")
        with col4:
            st.metric("FIRE Rate", f"{pct_fire:.1f}%")

        st.plotly_chart(fig, use_container_width=True)

# Chart 4: Account Composition Stacked Area
with tab3:
    st.subheader("Account Composition Over Time (Median)")

    if tax_adjust:
        st.caption(f"After-tax values: 401k discounted at {eff_ret_tax_rate:.0%} effective rate · Taxable gains at {ltcg_rate:.0%} LTCG · Roth/HSA shown at full value")

    ages = results.ages
    if tax_adjust:
        gains_med = np.maximum(results.taxable - results.taxable_basis, 0)
        taxable_med = np.median(results.taxable - gains_med * ltcg_rate, axis=1)
        t401k_med   = np.median(results.t401k * (1 - eff_ret_tax_rate), axis=1)
    else:
        taxable_med = np.median(results.taxable, axis=1)
        t401k_med   = np.median(results.t401k, axis=1)
    roth_med = np.median(results.roth, axis=1)
    hsa_med = np.median(results.hsa, axis=1)
    home_med = np.median(results.home_equity, axis=1)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=ages, y=taxable_med, name='Taxable' + (' (after LTCG)' if tax_adjust else ''),
        mode='lines', stackgroup='one', fillcolor='rgba(99, 110, 250, 0.7)'
    ))
    fig.add_trace(go.Scatter(
        x=ages, y=t401k_med, name='401(k)' + (f' (after {eff_ret_tax_rate:.0%} tax)' if tax_adjust else ''),
        mode='lines', stackgroup='one', fillcolor='rgba(239, 85, 59, 0.7)'
    ))
    fig.add_trace(go.Scatter(
        x=ages, y=roth_med, name='Roth IRA',
        mode='lines', stackgroup='one', fillcolor='rgba(0, 204, 150, 0.7)'
    ))
    fig.add_trace(go.Scatter(
        x=ages, y=hsa_med, name='HSA',
        mode='lines', stackgroup='one', fillcolor='rgba(171, 99, 250, 0.7)'
    ))
    fig.add_trace(go.Scatter(
        x=ages, y=home_med, name='Home Equity',
        mode='lines', stackgroup='one', fillcolor='rgba(255, 161, 90, 0.7)'
    ))

    # Mark median FIRE age
    if median_fire_age:
        fig.add_vline(x=median_fire_age, line_dash="dash", line_color="green",
                      annotation_text=f"Median FIRE: {median_fire_age:.0f}")

    fig.update_layout(
        xaxis_title="Age",
        yaxis_title="Balance",
        yaxis=dict(tickformat='$,.0f'),
        hovermode='x unified',
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

# Chart 4: Income vs Spending
with tab4:
    st.subheader("Income vs Spending Trajectories (Median)")

    real_terms_t4 = st.toggle("Inflation-adjusted (today's dollars)", value=True,
                              help="Deflates to today's purchasing power", key="real_t4")

    ages = results.ages
    income_med = np.median(results.incomes, axis=1)
    ss_income_med = np.median(results.ss_income, axis=1)
    total_income_med = income_med + ss_income_med
    spending_med = np.median(results.spending, axis=1)

    if real_terms_t4:
        infl_factors_t4 = (1 + INFLATION) ** (ages - start_age)
        income_med = income_med / infl_factors_t4
        ss_income_med = ss_income_med / infl_factors_t4
        total_income_med = total_income_med / infl_factors_t4
        spending_med = spending_med / infl_factors_t4

    net = total_income_med - spending_med  # positive = surplus, negative = deficit
    surplus = np.where(net >= 0, net, 0)
    deficit = np.where(net < 0, net, 0)

    fig = go.Figure()

    # Surplus and deficit fills (drawn first so lines render on top)
    fig.add_trace(go.Scatter(
        x=ages, y=surplus,
        fill='tozeroy', fillcolor='rgba(0, 204, 150, 0.25)',
        line=dict(width=0), name='Surplus', showlegend=True,
        hovertemplate='Surplus: $%{y:,.0f}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=ages, y=deficit,
        fill='tozeroy', fillcolor='rgba(239, 85, 59, 0.25)',
        line=dict(width=0), name='Deficit', showlegend=True,
        hovertemplate='Deficit: $%{y:,.0f}<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=ages, y=total_income_med,
        mode='lines', name='Income',
        line=dict(color='#00CC96', width=2),
        hovertemplate='Income: $%{y:,.0f}<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=ages, y=spending_med,
        mode='lines', name='Total Spending',
        line=dict(color='#EF553B', width=2),
        hovertemplate='Spending: $%{y:,.0f}<extra></extra>'
    ))

    if median_fire_age:
        fig.add_vline(x=median_fire_age, line_dash="dash", line_color="green",
                      annotation_text=f"Median FIRE: {median_fire_age:.0f}")

    fig.update_layout(
        xaxis_title="Age",
        yaxis=dict(title="Amount per Year", tickformat='$,.0f'),
        hovermode='x unified',
        height=500,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

# Chart 7: FIRE Probability Curve (CDF)
with tab7:
    st.subheader("Cumulative FIRE Probability by Age")

    ages_range = np.arange(start_age, FIRE_HORIZON_AGE + 1)
    cdf = [(results.fire_ages <= age).mean() * 100 for age in ages_range]

    # Also run comparison with zero seed
    @st.cache_data
    def run_comparison_sim(starting_tc, city, n_sims, marriage_age, kid_ages, spouse_works, spouse_salary, part_time_fraction, life_exp, career_traj, soft_cap, emp_match_pct, emp_match_limit,
                           ss_enabled, ss_claiming_age, ss_monthly, ss_spouse_monthly, ss_spouse_claiming):
        family = FamilyConfig(
            marriage_age=marriage_age, kid_ages=kid_ages, spouse_works=spouse_works,
            spouse_salary=spouse_salary, part_time_fraction=part_time_fraction
        )
        career = CareerConfig(soft_cap=soft_cap, trajectory=career_traj,
                             employer_match_pct=emp_match_pct, employer_match_limit=emp_match_limit)
        ss = SocialSecurityConfig(
            enabled=ss_enabled, claiming_age=ss_claiming_age,
            monthly_benefit_today=ss_monthly, spouse_monthly_benefit=ss_spouse_monthly,
            spouse_claiming_age=ss_spouse_claiming
        )
        rng = np.random.default_rng(42)
        fire_ages, _, _ = run_vectorized(starting_tc, city, n_sims, rng, seed_amounts=SeedAmounts(),
                              family_config=family, career_config=career, ss_config=ss,
                              return_trajectories=False, life_expectancy=life_exp)
        return fire_ages

    zero_seed_ages = run_comparison_sim(
        starting_tc, city, n_sims, marriage_age, kid_ages, spouse_works, spouse_salary, part_time_fraction, life_expectancy,
        career_trajectory, tc_soft_cap, employer_match_pct, employer_match_limit,
        ss_enabled, ss_claiming_age, ss_monthly, ss_spouse_monthly, ss_spouse_claiming
    )
    cdf_zero = [(zero_seed_ages <= age).mean() * 100 for age in ages_range]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=ages_range, y=cdf_zero,
        mode='lines', name='$0 Seed',
        line=dict(color='#EF553B', width=2, dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=ages_range, y=cdf,
        mode='lines', name=f'${seed_total/1000:.0f}K Seed',
        line=dict(color='#636EFA', width=2)
    ))

    # Add reference lines
    fig.add_hline(y=50, line_dash="dot", line_color="gray", annotation_text="50%")
    fig.add_hline(y=90, line_dash="dot", line_color="gray", annotation_text="90%")

    fig.update_layout(
        xaxis_title="Age",
        yaxis_title="% FIRE'd by This Age",
        yaxis_range=[0, 105],
        hovermode='x unified',
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

# Chart 8: Ablation Analysis
with tab8:
    st.subheader("Ablation Analysis: What Moves the Needle?")
    st.caption(
        "Each row varies one lever while holding everything else at your current settings. "
        "Switch metrics below — FIRE age and success rate tell different stories."
    )

    ablation_metric = st.radio(
        "Metric",
        ["Median FIRE Age", "Success Rate (% not running out of money)"],
        horizontal=True,
    )

    @st.cache_data
    def run_ablation(base_tc, city, n_sims, start_age,
                     seed_taxable, seed_401k, seed_roth, seed_hsa,
                     marriage_age, kid_ages_base, spouse_works_base, spouse_salary_base,
                     part_time_fraction,
                     career_traj, soft_cap, emp_match_pct, emp_match_limit,
                     use_401k, use_roth, use_hsa_base, use_mega_backdoor_base,
                     hsa_annual_contrib, hsa_employer_contrib,
                     ss_enabled, ss_claiming_age, ss_monthly, ss_spouse_monthly, ss_spouse_claiming,
                     fire_horizon, life_expectancy, swr_override, lifestyle_creep_pct):
        from concurrent.futures import ThreadPoolExecutor

        def _run_scenario(overrides):
            _tc       = overrides.get('tc',                base_tc)
            _city     = overrides.get('city',              city)
            _seed_tax = overrides.get('seed_taxable',      seed_taxable)
            _kid_ages = overrides.get('kid_ages',          kid_ages_base)
            _sp_works = overrides.get('spouse_works',      spouse_works_base)
            _sp_sal   = overrides.get('spouse_salary',     spouse_salary_base)
            _use_hsa  = overrides.get('use_hsa',           use_hsa_base)
            _use_mega = overrides.get('use_mega_backdoor', use_mega_backdoor_base)
            _hsa_c    = overrides.get('hsa_annual_contrib', hsa_annual_contrib)

            seed   = SeedAmounts(taxable=_seed_tax, t401k=seed_401k, roth=seed_roth, hsa=seed_hsa)
            family = FamilyConfig(
                marriage_age=marriage_age, kid_ages=_kid_ages,
                spouse_works=_sp_works,
                spouse_salary=_sp_sal if _sp_works else 0,
                part_time_fraction=part_time_fraction,
            )
            career = CareerConfig(soft_cap=soft_cap, trajectory=career_traj,
                                  employer_match_pct=emp_match_pct, employer_match_limit=emp_match_limit)
            ss = SocialSecurityConfig(
                enabled=ss_enabled, claiming_age=ss_claiming_age,
                monthly_benefit_today=ss_monthly, spouse_monthly_benefit=ss_spouse_monthly,
                spouse_claiming_age=ss_spouse_claiming,
            )
            rng = np.random.default_rng(42)
            fire_ages, failed, _ = run_vectorized(
                _tc, _city, n_sims, rng,
                seed_amounts=seed, family_config=family, career_config=career, ss_config=ss,
                current_age=start_age, fire_horizon=fire_horizon, life_expectancy=life_expectancy,
                use_401k=use_401k, use_roth=use_roth, use_hsa=_use_hsa,
                use_mega_backdoor=_use_mega,
                hsa_annual_contrib=_hsa_c if _use_hsa else None,
                hsa_employer_contrib=hsa_employer_contrib,
                swr_override=swr_override, lifestyle_creep_pct=lifestyle_creep_pct,
                return_trajectories=False,
            )
            valid = fire_ages[(fire_ages < 99) & ~failed]
            med_fire_age = float(np.median(valid)) if len(valid) > 0 else float(fire_horizon)
            success_rate = float((~failed).mean()) * 100
            return (med_fire_age, success_rate)

        levers = {
            "City": [
                ("Sacramento",    {"city": "Sacramento"}),
                ("San Francisco", {"city": "San Francisco"}),
                ("NYC",           {"city": "New York City"}),
            ],
            "Kids": [
                ("0 kids", {"kid_ages": ()}),
                ("1 kid",  {"kid_ages": (31,)}),
                ("2 kids", {"kid_ages": (31, 33)}),
                ("3 kids", {"kid_ages": (31, 33, 35)}),
            ],
            "Spouse": [
                ("No spouse",   {"spouse_works": False, "spouse_salary": 0}),
                ("Spouse $80k", {"spouse_works": True,  "spouse_salary": 80000}),
            ],
            "Starting Savings": [
                ("$0",    {"seed_taxable": 0}),
                ("$100k", {"seed_taxable": 100_000}),
                ("$250k", {"seed_taxable": 250_000}),
                ("$500k", {"seed_taxable": 500_000}),
            ],
            "HSA": [
                ("Off",      {"use_hsa": False}),
                ("On (max)", {"use_hsa": True, "hsa_annual_contrib": HSA_FAMILY_LIMIT}),
            ],
            "Mega Backdoor": [
                ("Off",      {"use_mega_backdoor": False}),
                ("On (max)", {"use_mega_backdoor": True}),
            ],
        }

        tasks = []
        for lever, scenarios in levers.items():
            for idx, (label, overrides) in enumerate(scenarios):
                tasks.append((lever, idx, label, overrides))

        results = {lever: [None] * len(scenarios) for lever, scenarios in levers.items()}
        with ThreadPoolExecutor(max_workers=8) as ex:
            futures = [(lever, idx, label, ex.submit(_run_scenario, overrides))
                       for lever, idx, label, overrides in tasks]
            for lever, idx, label, fut in futures:
                results[lever][idx] = (label,) + fut.result()  # (label, fire_age, success_rate)

        return results

    ablation = run_ablation(
        starting_tc, city, n_sims, start_age,
        seed_taxable, seed_401k, seed_roth, seed_hsa,
        marriage_age, kid_ages, spouse_works, spouse_salary,
        part_time_fraction,
        career_trajectory, tc_soft_cap, employer_match_pct, employer_match_limit,
        use_401k, use_roth, use_hsa, use_mega_backdoor,
        hsa_annual_contrib, hsa_employer_contrib,
        ss_enabled, ss_claiming_age, ss_monthly, ss_spouse_monthly, ss_spouse_claiming,
        fire_horizon, life_expectancy, swr_override, lifestyle_creep_pct,
    )

    # Pick which column index to plot: 1 = fire_age, 2 = success_rate
    val_idx = 1 if ablation_metric == "Median FIRE Age" else 2
    higher_is_better = (val_idx == 2)  # success rate: higher = better; fire age: lower = better

    def _gradient_color(score):
        """score 0.0 = worst (red) → 1.0 = best (green), via orange at 0.5."""
        r1, g1, b1 = 0xEF, 0x55, 0x3B  # red
        r2, g2, b2 = 0xFF, 0xAA, 0x00  # orange (midpoint)
        r3, g3, b3 = 0x00, 0xCC, 0x96  # green
        if score <= 0.5:
            t = score * 2
            r, g, b = int(r1 + (r2 - r1) * t), int(g1 + (g2 - g1) * t), int(b1 + (b2 - b1) * t)
        else:
            t = (score - 0.5) * 2
            r, g, b = int(r2 + (r3 - r2) * t), int(g2 + (g3 - g2) * t), int(b2 + (b3 - b2) * t)
        return f"#{r:02X}{g:02X}{b:02X}"

    def _row_colors(vals, higher_is_better):
        lo, hi = min(vals), max(vals)
        if hi == lo:
            return ["#AAAAAA"] * len(vals)
        scores = [(v - lo) / (hi - lo) for v in vals]
        if not higher_is_better:
            scores = [1 - s for s in scores]
        return [_gradient_color(s) for s in scores]

    # Sort by spread on the selected metric (most impactful first)
    sorted_levers = sorted(
        ablation.items(),
        key=lambda kv: max(row[val_idx] for row in kv[1]) - min(row[val_idx] for row in kv[1]),
        reverse=True,
    )

    fig = go.Figure()
    lever_names = [lever for lever, _ in sorted_levers]

    for lever, scenarios in sorted_levers:
        vals   = [row[val_idx] for row in scenarios]
        labels = [row[0]       for row in scenarios]
        colors = _row_colors(vals, higher_is_better)

        fig.add_trace(go.Scatter(
            x=[min(vals), max(vals)],
            y=[lever, lever],
            mode="lines",
            line=dict(color="#DDDDDD", width=2),
            showlegend=False,
            hoverinfo="skip",
        ))

        fig.add_trace(go.Scatter(
            x=vals,
            y=[lever] * len(vals),
            mode="markers",
            marker=dict(
                size=14,
                color=colors,
                line=dict(color="white", width=1.5),
            ),
            text=labels,
            customdata=vals,
            hovertemplate="%{text}: %{customdata:.1f}<extra>" + lever + "</extra>",
            showlegend=False,
        ))

        for i, (row, val) in enumerate(zip(scenarios, vals)):
            fig.add_annotation(
                x=val, y=lever,
                text=row[0],
                showarrow=False,
                yshift=18 if i % 2 == 0 else -18,
                font=dict(size=10, color=colors[i]),
                xanchor="center",
            )

        # Spread label flush to the right edge
        spread = max(vals) - min(vals)
        spread_text = f"Δ {spread:.1f} yrs" if val_idx == 1 else f"Δ {spread:.1f} ppt"
        fig.add_annotation(
            x=1.01, xref="paper",
            y=lever,
            text=f"<b>{spread_text}</b>",
            showarrow=False,
            font=dict(size=11, color="#555555"),
            xanchor="left",
        )

    all_vals = [row[val_idx] for _, scenarios in sorted_levers for row in scenarios]
    x_pad = max(1, (max(all_vals) - min(all_vals)) * 0.08)

    if ablation_metric == "Median FIRE Age":
        x_title = "Median FIRE Age"
        x_range = [min(all_vals) - x_pad, max(all_vals) + x_pad]
    else:
        x_title = "Success Rate (%)"
        x_range = [max(0, min(all_vals) - x_pad), min(100, max(all_vals) + x_pad)]

    fig.update_layout(
        xaxis=dict(
            title=x_title,
            range=x_range,
            showgrid=True,
            gridcolor="#EEEEEE",
        ),
        yaxis=dict(
            categoryorder="array",
            categoryarray=lever_names[::-1],
            showgrid=False,
        ),
        height=max(420, len(sorted_levers) * 90),
        margin=dict(l=120, r=110, t=40, b=60),
        plot_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# DATA TABLE
# =============================================================================
st.markdown("---")
st.subheader("Minimum TC for 90% FIRE Confidence")

# Option to use baseline or personalized settings
table_mode = st.radio(
    "Table Mode",
    ["Baseline (Static)", "Your Settings (Dynamic)"],
    help="Baseline shows standard scenario ($0 seed, 2 kids). Your Settings uses your current configuration."
)

@st.cache_data
def compute_min_tc_table(seed_taxable, seed_401k, seed_roth, seed_hsa,
                         marriage_age, kid_ages, spouse_works, spouse_salary, part_time_fraction,
                         city_names, career_traj, soft_cap, emp_match_pct, emp_match_limit,
                         ss_enabled, ss_claiming_age, ss_monthly, ss_spouse_monthly, ss_spouse_claiming):
    ages_to_check = [40, 42, 44, 46, 48, 50, 52, 55]
    seed = SeedAmounts(taxable=seed_taxable, t401k=seed_401k, roth=seed_roth, hsa=seed_hsa)
    family = FamilyConfig(
        marriage_age=marriage_age, kid_ages=kid_ages, spouse_works=spouse_works,
        spouse_salary=spouse_salary, part_time_fraction=part_time_fraction
    )
    career = CareerConfig(soft_cap=soft_cap, trajectory=career_traj,
                         employer_match_pct=emp_match_pct, employer_match_limit=emp_match_limit)
    ss = SocialSecurityConfig(
        enabled=ss_enabled, claiming_age=ss_claiming_age,
        monthly_benefit_today=ss_monthly, spouse_monthly_benefit=ss_spouse_monthly,
        spouse_claiming_age=ss_spouse_claiming
    )

    data = []
    for city_name in city_names:
        row = {'City': city_name}
        for ta in ages_to_check:
            tc = find_min_tc(city_name, ta, 90, seed_amounts=seed, family_config=family,
                           career_config=career, ss_config=ss)
            row[f'Age {ta}'] = f"${tc/1000:.0f}K"
        data.append(row)
    return data

if table_mode == "Baseline (Static)":
    # Use fixed baseline parameters
    baseline_seed_taxable, baseline_seed_401k, baseline_seed_roth, baseline_seed_hsa = 0, 0, 0, 0
    baseline_marriage_age = 29
    baseline_kid_ages = (31, 33)
    baseline_spouse_works = True
    baseline_spouse_salary = 80000
    baseline_part_time_fraction = 0.5

    with st.spinner("Computing baseline minimum TC table..."):
        min_tc_data = compute_min_tc_table(
            baseline_seed_taxable, baseline_seed_401k, baseline_seed_roth, baseline_seed_hsa,
            baseline_marriage_age, baseline_kid_ages, baseline_spouse_works,
            baseline_spouse_salary, baseline_part_time_fraction,
            tuple(all_cities.keys()), career_trajectory, tc_soft_cap, employer_match_pct, employer_match_limit,
            ss_enabled, ss_claiming_age, ss_monthly, ss_spouse_monthly, ss_spouse_claiming
        )
    st.caption(f"Baseline: {_amt(0)} starting seed, married at 29, 2 kids (ages 31, 33), spouse works ({_amt(80000)}, 50% part-time after kids)")
else:
    # Use current user settings
    if st.button("Calculate with Your Settings", type="primary"):
        with st.spinner("Computing personalized minimum TC table..."):
            min_tc_data = compute_min_tc_table(
                seed_taxable, seed_401k, seed_roth, seed_hsa,
                marriage_age, kid_ages, spouse_works, spouse_salary, part_time_fraction,
                tuple(all_cities.keys()), career_trajectory, tc_soft_cap, employer_match_pct, employer_match_limit,
                ss_enabled, ss_claiming_age, ss_monthly, ss_spouse_monthly, ss_spouse_claiming
            )
            st.session_state.personalized_table = min_tc_data

    if 'personalized_table' in st.session_state:
        min_tc_data = st.session_state.personalized_table
        st.caption(f"Using your settings: {_amt(int(seed_total))} seed, {len(kid_ages)} kids, spouse {'works' if spouse_works else 'stays home'}")
    else:
        st.info("Click 'Calculate with Your Settings' to see how your configuration affects minimum TC requirements.")
        min_tc_data = None

if min_tc_data:
    st.dataframe(min_tc_data, use_container_width=True, hide_index=True)

# =============================================================================
# CITY CONFIGURATION EDITOR
# =============================================================================
st.markdown("---")
st.subheader("City Configuration Editor")
st.caption("Edit existing cities or add your own custom cities")

# Initialize session state for custom cities if not exists
if 'custom_cities' not in st.session_state:
    st.session_state.custom_cities = {}

with st.expander("Add or Edit City", expanded=False):
    col1, col2 = st.columns([1, 3])

    with col1:
        edit_mode = st.radio("Mode", ["Add New City", "Edit Existing City"], label_visibility="collapsed")

    with col2:
        if edit_mode == "Edit Existing City":
            city_to_edit = st.selectbox("Select City to Edit", list(all_cities.keys()))
            base_config = all_cities[city_to_edit]
            city_name = city_to_edit
            is_new = False
        else:
            city_name = st.text_input("New City Name", placeholder="e.g. Austin, Texas")
            base_config = CITIES['Sacramento']  # Use Sacramento as template
            is_new = True

    if city_name:
        st.markdown("### Rent & Housing")
        col1, col2, col3 = st.columns(3)
        with col1:
            one_br_rent = st.number_input("1BR Rent (Monthly)", min_value=0, value=int(base_config.one_br_rent), step=100, key=f"1br_{city_name}")
        with col2:
            nice_one_br_rent = st.number_input("Nice 1BR Rent (Monthly)", min_value=0, value=int(base_config.nice_one_br_rent), step=100, key=f"nice1br_{city_name}")
        with col3:
            family_rent = st.number_input("Family Rent (Monthly)", min_value=0, value=int(base_config.family_rent), step=100, key=f"famrent_{city_name}")

        st.markdown("### Home Purchase")
        col1, col2 = st.columns(2)
        with col1:
            has_home_price = st.checkbox("Home Purchase Available", value=base_config.home_price is not None, key=f"hasprice_{city_name}")
            if has_home_price:
                home_price = st.number_input("Home Price", min_value=0, value=int(base_config.home_price) if base_config.home_price else 500000, step=10000, key=f"homeprice_{city_name}")
            else:
                home_price = None
        with col2:
            down_payment_pct = st.slider("Down Payment %", min_value=0.0, max_value=1.0, value=base_config.down_payment_pct, step=0.05, format="%.0f%%", key=f"down_{city_name}")

        st.markdown("### Rates & Costs")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            mortgage_rate = st.number_input("Mortgage Rate", min_value=0.0, max_value=0.20, value=base_config.mortgage_rate, step=0.001, format="%.3f", key=f"mortgage_{city_name}")
            property_tax_rate = st.number_input("Property Tax Rate", min_value=0.0, max_value=0.05, value=base_config.property_tax_rate, step=0.001, format="%.3f", key=f"proptax_{city_name}")
        with col2:
            home_maintenance_pct = st.number_input("Home Maintenance %", min_value=0.0, max_value=0.05, value=base_config.home_maintenance_pct, step=0.001, format="%.3f", key=f"maint_{city_name}")
            insurance_premium = st.number_input("Annual Insurance", min_value=0, value=int(base_config.insurance_premium), step=100, key=f"ins_{city_name}")
        with col3:
            insurance_inflation = st.number_input("Insurance Inflation", min_value=0.0, max_value=0.20, value=base_config.insurance_inflation, step=0.01, format="%.2f", key=f"insinfl_{city_name}")
            utility_premium = st.number_input("Annual Utilities", min_value=0, value=int(base_config.utility_premium), step=100, key=f"util_{city_name}")
        with col4:
            home_appreciation = st.number_input("Home Appreciation", min_value=0.0, max_value=0.20, value=base_config.home_appreciation, step=0.01, format="%.2f", key=f"apprec_{city_name}")

        st.markdown("### Tax Rates")
        col1, col2 = st.columns(2)
        with col1:
            state_tax_rate = st.number_input("State Tax Rate (Working)", min_value=0.0, max_value=0.20, value=base_config.state_tax_rate, step=0.01, format="%.2f", key=f"statetax_{city_name}")
        with col2:
            retirement_state_tax = st.number_input("State Tax Rate (Retired)", min_value=0.0, max_value=0.20, value=base_config.retirement_state_tax, step=0.01, format="%.2f", key=f"rettax_{city_name}")

        # Save button
        if st.button(f"{'Update' if not is_new else 'Add'} {city_name}", type="primary"):
            new_config = CityConfig(
                one_br_rent=float(one_br_rent),
                nice_one_br_rent=float(nice_one_br_rent),
                family_rent=float(family_rent),
                home_price=float(home_price) if has_home_price else None,
                down_payment_pct=float(down_payment_pct),
                mortgage_rate=float(mortgage_rate),
                property_tax_rate=float(property_tax_rate),
                home_maintenance_pct=float(home_maintenance_pct),
                insurance_premium=float(insurance_premium),
                insurance_inflation=float(insurance_inflation),
                utility_premium=float(utility_premium),
                home_appreciation=float(home_appreciation),
                state_tax_rate=float(state_tax_rate),
                retirement_state_tax=float(retirement_state_tax),
            )

            st.session_state.custom_cities[city_name] = new_config
            st.success(f"{'Updated' if not is_new else 'Added'} {city_name}!")
            st.rerun()

# --- Assumptions Panel ---
st.markdown("---")
st.markdown("#### Model Assumptions")
st.caption(
    f"**IRS limits (2025):** 401(k) {_amt(FOUR01K_LIMIT)} · Roth IRA {_amt(ROTH_IRA_LIMIT)} · HSA {_amt(HSA_INDIVIDUAL_LIMIT)} individual / {_amt(HSA_FAMILY_LIMIT)} family  \n"
    f"**One-time costs:** Car+move {_amt(OT_CAR_MOVE)} (age 28) · Baby setup {_amt(OT_BABY_SETUP)} · Mid-life upgrade {_amt(OT_MID_UPGRADE)} (age 38) · Home closing {_amt(HOME_CLOSING_COSTS)} · Job search {_amt(JL_SEARCH_COST)}  \n"
    f"**Discretionary base:** {_amt(DISC_YOUNG)}/yr (20s) → {_amt(DISC_MID)}/yr (30s) · +{_amt(DISC_STEP_35)} at 35 · +{_amt(DISC_STEP_40)} at 40 · {_amt(DISC_FAMILY)}/yr (family stage)  \n"
    f"**Healthcare (working):** {_amt(HC_YOUNG)}/yr under 40 · {_amt(HC_OLDER)}/yr 40+  \n"
    f"**Healthcare (retired):** {_amt(HC_RET_BASE)}/yr base · +{_amt(HC_RET_AGE_STEP)}/yr per age · {HC_RET_REAL_GROWTH:.0%}/yr real growth · {_amt(HC_MEDICARE)}/yr Medicare (65+)  \n"
    f"**Job loss:** {JL_ANNUAL_PROB:.0%}/yr probability · {JL_MONTHS_MIN}–{JL_MONTHS_MAX} months unemployed · re-entry at {JL_REENTRY_MIN:.0%}–{JL_REENTRY_MAX:.0%} of prior salary  \n"
    f"**Market returns:** normal {MR_NORMAL_MEAN:.0%} ± {MR_NORMAL_STD:.0%} · recession {MR_RECESSION_MEAN:.0%} ± {MR_RECESSION_STD:.0%} · recession prob {RECESSION_PROB:.0%}/yr  \n"
    f"**Health shock:** {HEALTH_SHOCK_PROB:.1%}/yr · {_amt(HEALTH_SHOCK_COST)} per event"
)

# Display current cities
if st.session_state.custom_cities:
    st.markdown("### Custom Cities")
    for custom_city_name in st.session_state.custom_cities.keys():
        col1, col2 = st.columns([4, 1])
        with col1:
            st.write(f"**{custom_city_name}**")
        with col2:
            if st.button("Remove", key=f"remove_{custom_city_name}"):
                del st.session_state.custom_cities[custom_city_name]
                st.rerun()

