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

        st.subheader("401(k) Employer Match")
        employer_match_pct = st.slider(
            "Match Rate (%)",
            min_value=0, max_value=100, value=50, step=5,
            format="%d%%",
            help="Employer contributes this % of your contribution (e.g. 50% = 50¢ per $1 you contribute)"
        ) / 100.0
        employer_match_limit = st.slider(
            "Match Limit (% of IRS limit)",
            min_value=0, max_value=100, value=43, step=1,
            format="%d%%",
            help=f"Employer matches up to this % of the IRS 401(k) limit ($23,000). 43% ≈ $10k/yr."
        ) / 100.0

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
        use_mega_backdoor = st.toggle("Mega Backdoor Roth", value=bool(_qp("mega_backdoor", 0)),
            help="After-tax 401k contributions converted to Roth (up to $70k total 401k limit minus pre-tax and employer match). Requires employer plan support.")
        use_roth = st.toggle("Roth IRA", value=bool(_qp("use_roth", 1)), help="Annual Roth IRA contributions (subject to income limits in reality)")
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

        num_kids = st.radio("Number of Kids", [0, 1, 2], index=_qp("kids", 2), horizontal=True)
        kid_ages = ()
        wedding_budget = st.number_input(
            "Wedding Budget ($)", min_value=0, max_value=500000, value=60000, step=5000,
            help="One-time cost in the year after your marriage age"
        )

        if num_kids >= 1:
            kid1_age = st.slider("First Kid Born (Your Age)", 26, 45, 31)
            kid_ages = (kid1_age,)
            if num_kids == 2:
                kid2_age = st.slider("Second Kid Born (Your Age)", kid1_age, 47, max(33, kid1_age + 2))
                kid_ages = (kid1_age, kid2_age)

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
            help="Extra discretionary spending as % of gross income. Tapers to zero as you approach FIRE age 60 (people tend to optimize spending near the finish line)."
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
            use_401k=True, use_roth=True, use_hsa=True, use_mega_backdoor=False):
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
                          use_mega_backdoor=use_mega_backdoor)

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
        use_mega_backdoor=use_mega_backdoor
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
if not use_roth: _params["use_roth"] = 0
if not use_hsa:  _params["use_hsa"] = 0
st.query_params.update(_params)

# =============================================================================
# KEY METRICS
# =============================================================================
FIRE_HORIZON_AGE = fire_horizon

# Voluntary FIRE = achieved FIRE before forced retirement at 60
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
              help="% of simulations that achieved voluntary FIRE before age 60")
with col2:
    st.metric("Success Rate", f"{pct_success:.1f}%",
              help=f"Portfolio survived to age {life_expectancy} (everyone retires at 60)")
with col3:
    st.metric("Median FIRE Age", f"{median_fire_age:.0f}" if median_fire_age else "N/A")
with col4:
    st.metric("P10 FIRE Age", f"{p10_fire_age:.0f}" if p10_fire_age else "N/A")
with col5:
    st.metric("P90 FIRE Age", f"{p90_fire_age:.0f}" if p90_fire_age else "N/A")

# FIRE number countdown row
current_nw = float(np.median(results.net_worth[0]))
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

    view = st.radio("Display", ["Annual", "Monthly"], horizontal=True, key="spend_view")
    div = 12 if view == "Monthly" else 1
    unit = "/mo" if view == "Monthly" else "/yr"

    ages = results.ages
    housing_med = np.median(results.spending_housing,       axis=1) / div
    disc_med    = np.median(results.spending_discretionary, axis=1) / div
    hc_med      = np.median(results.spending_healthcare,    axis=1) / div
    kids_med    = np.median(results.spending_kids,          axis=1) / div
    edu_med     = np.median(results.spending_education,     axis=1) / div
    ot_med      = np.median(results.spending_one_time,      axis=1) / div
    tax_med     = np.median(results.taxes,                  axis=1) / div
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
        yaxis_title=f"Spending ({unit})",
        yaxis=dict(tickformat='$,.0f'),
        hovermode='x unified',
        height=500,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Data table — every 5 years
    s401k_med  = np.median(results.savings_401k,    axis=1) / div
    sroth_med  = np.median(results.savings_roth,    axis=1) / div
    shsa_med   = np.median(results.savings_hsa,     axis=1) / div
    stax_med   = np.median(results.savings_taxable, axis=1) / div
    saved_med  = s401k_med + sroth_med + shsa_med + stax_med

    income_med_cf = np.median(results.incomes, axis=1) / div

    _dash = "—"
    st.caption("Median spending and savings by age")
    sample_idx = list(range(0, len(ages)))
    rows = [
        {
            "Age":           int(ages[i]),
            "Housing":       f"${housing_med[i]:,.0f}",
            "Discretionary": f"${disc_med[i]:,.0f}",
            "Healthcare":    f"${hc_med[i]:,.0f}",
            "Kids":          f"${kids_med[i]:,.0f}",
            "Education":     f"${edu_med[i]:,.0f}",
            "One-Time":      f"${ot_med[i]:,.0f}",
            "Taxes":         f"${tax_med[i]:,.0f}",
            "401(k)":        f"${s401k_med[i]:,.0f}",
            "Roth IRA":      f"${sroth_med[i]:,.0f}",
            "HSA":           f"${shsa_med[i]:,.0f}",
            "Brokerage":     f"${stax_med[i]:,.0f}",
            "Spending":      f"${total_med[i]:,.0f}",
            "Earned":        f"${income_med_cf[i]:,.0f}",
            "Total Saved":   f"${saved_med[i]:,.0f}",
            "Savings %":     f"{saved_med[i] / income_med_cf[i] * 100:.0f}%" if income_med_cf[i] > 0 else _dash,
        }
        for i in sample_idx
    ]

    # Aggregate summary rows at the bottom
    rows.append({
        "Age": "Total Spending",
        "Housing":       f"${housing_med.sum():,.0f}",
        "Discretionary": f"${disc_med.sum():,.0f}",
        "Healthcare":    f"${hc_med.sum():,.0f}",
        "Kids":          f"${kids_med.sum():,.0f}",
        "Education":     f"${edu_med.sum():,.0f}",
        "One-Time":      f"${ot_med.sum():,.0f}",
        "Taxes":         f"${tax_med.sum():,.0f}",
        "401(k)": _dash, "Roth IRA": _dash, "HSA": _dash, "Brokerage": _dash,
        "Spending":    f"${total_med.sum():,.0f}",
        "Earned":      _dash,
        "Total Saved": _dash,
        "Savings %":   _dash,
    })
    rows.append({
        "Age": "Total Earned",
        "Housing": _dash, "Discretionary": _dash, "Healthcare": _dash,
        "Kids": _dash, "Education": _dash, "One-Time": _dash, "Taxes": _dash,
        "401(k)": _dash, "Roth IRA": _dash, "HSA": _dash, "Brokerage": _dash,
        "Spending":    _dash,
        "Earned":      f"${income_med_cf.sum():,.0f}",
        "Total Saved": _dash,
        "Savings %":   _dash,
    })
    _lifetime_savings_pct = saved_med.sum() / income_med_cf.sum() * 100 if income_med_cf.sum() > 0 else 0
    rows.append({
        "Age": "Total Saved",
        "Housing": _dash, "Discretionary": _dash, "Healthcare": _dash,
        "Kids": _dash, "Education": _dash, "One-Time": _dash, "Taxes": _dash,
        "401(k)":  f"${s401k_med.sum():,.0f}",
        "Roth IRA": f"${sroth_med.sum():,.0f}",
        "HSA":      f"${shsa_med.sum():,.0f}",
        "Brokerage": f"${stax_med.sum():,.0f}",
        "Spending":    _dash,
        "Earned":      _dash,
        "Total Saved": f"${saved_med.sum():,.0f}",
        "Savings %":   f"{_lifetime_savings_pct:.0f}%",
    })

    st.dataframe(rows, use_container_width=True, hide_index=True)

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
            xaxis_title=f"Net Worth at Age {life_expectancy} {'(today\'s $)' if real_terms_t6 else '(nominal)'} — P1-P99 shown",
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
                    val = results.net_worth[age_idx[0], sim_idx]
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
                xaxis_title=f"Net Worth at FIRE {'(today\'s $)' if real_terms_t6 else '(nominal)'} — P1-P99 shown",
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

    ages = results.ages
    taxable_med = np.median(results.taxable, axis=1)
    t401k_med = np.median(results.t401k, axis=1)
    roth_med = np.median(results.roth, axis=1)
    hsa_med = np.median(results.hsa, axis=1)
    home_med = np.median(results.home_equity, axis=1)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=ages, y=taxable_med, name='Taxable',
        mode='lines', stackgroup='one', fillcolor='rgba(99, 110, 250, 0.7)'
    ))
    fig.add_trace(go.Scatter(
        x=ages, y=t401k_med, name='401(k)',
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

# Chart 8: Sensitivity Tornado
with tab8:
    st.subheader("Sensitivity Analysis: Impact on Median FIRE Age")
    st.caption("Shows how ±20% change in each parameter affects median FIRE age")

    if median_fire_age is None:
        st.warning("Not enough simulations reached FIRE to perform sensitivity analysis")
    else:
        @st.cache_data
        def run_sensitivity(base_tc, city, n_sims, seed_taxable, seed_401k, seed_roth, seed_hsa,
                           marriage_age, kid_ages, spouse_works, spouse_salary, part_time_fraction,
                           career_traj, soft_cap, emp_match_pct, emp_match_limit,
                           ss_enabled, ss_claiming_age, ss_monthly, ss_spouse_monthly, ss_spouse_claiming):
            results = {}
            base_params = {
                'starting_tc': base_tc,
                'seed_taxable': seed_taxable,
                'spouse_salary': spouse_salary if spouse_works else 0,
            }

            ss = SocialSecurityConfig(
                enabled=ss_enabled, claiming_age=ss_claiming_age,
                monthly_benefit_today=ss_monthly, spouse_monthly_benefit=ss_spouse_monthly,
                spouse_claiming_age=ss_spouse_claiming
            )

            for param, base_val in base_params.items():
                if base_val == 0:
                    continue

                low_val = int(base_val * 0.8)
                high_val = int(base_val * 1.2)

                ages_list = []
                for val in [low_val, high_val]:
                    seed = SeedAmounts(
                        taxable=val if param == 'seed_taxable' else seed_taxable,
                        t401k=seed_401k, roth=seed_roth, hsa=seed_hsa
                    )
                    family = FamilyConfig(
                        marriage_age=marriage_age, kid_ages=kid_ages, spouse_works=spouse_works,
                        spouse_salary=val if param == 'spouse_salary' else spouse_salary,
                        part_time_fraction=part_time_fraction
                    )
                    career = CareerConfig(soft_cap=soft_cap, trajectory=career_traj,
                                         employer_match_pct=emp_match_pct, employer_match_limit=emp_match_limit)
                    tc = val if param == 'starting_tc' else base_tc
                    rng = np.random.default_rng(42)
                    fire_ages, _, _ = run_vectorized(tc, city, n_sims, rng, seed_amounts=seed,
                                              family_config=family, career_config=career,
                                              ss_config=ss, return_trajectories=False)
                    valid = fire_ages[fire_ages < 99]
                    ages_list.append(np.median(valid) if len(valid) > 0 else 99)

                results[param] = (ages_list[0], ages_list[1])

            return results

        sensitivity = run_sensitivity(
            starting_tc, city, n_sims, seed_taxable, seed_401k, seed_roth, seed_hsa,
            marriage_age, kid_ages, spouse_works, spouse_salary, part_time_fraction,
            career_trajectory, tc_soft_cap, employer_match_pct, employer_match_limit,
            ss_enabled, ss_claiming_age, ss_monthly, ss_spouse_monthly, ss_spouse_claiming
        )

        params = []
        low_deltas = []
        high_deltas = []

        for param, (low_age, high_age) in sensitivity.items():
            readable = {
                'starting_tc': 'Total Compensation',
                'seed_taxable': 'Starting Taxable',
                'spouse_salary': 'Spouse Salary'
            }.get(param, param)

            params.append(readable)
            low_deltas.append(low_age - median_fire_age)  # -20% param
            high_deltas.append(high_age - median_fire_age)  # +20% param

        fig = go.Figure()

        fig.add_trace(go.Bar(
            y=params, x=low_deltas,
            orientation='h', name='-20%',
            marker_color='#EF553B'
        ))
        fig.add_trace(go.Bar(
            y=params, x=high_deltas,
            orientation='h', name='+20%',
            marker_color='#00CC96'
        ))

        fig.update_layout(
            xaxis_title="Change in Median FIRE Age (years)",
            barmode='group',
            height=400
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

