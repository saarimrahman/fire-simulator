"""
./venv/bin/streamlit run fire_dashboard.py
"""


import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from fire import (
    run_vectorized, find_min_tc, SeedAmounts, FamilyConfig, SimulationResults,
    CITIES, CURRENT_AGE, FIRE_HORIZON, LIFE_EXPECTANCY, CityConfig, add_custom_cities
)

st.set_page_config(page_title="FIRE Simulator", page_icon="🔥", layout="wide")

st.title("FIRE Simulation Dashboard")

# =============================================================================
# SIDEBAR: Controls Panel
# =============================================================================
with st.sidebar:
    st.header("Simulation Settings")

    n_sims = st.select_slider(
        "Number of Simulations",
        options=[1000, 2500, 5000, 10000, 20000, 50000],
        value=10000,
        help="More simulations = more accurate but slower"
    )

    # Merge built-in cities with custom cities
    if 'custom_cities' not in st.session_state:
        st.session_state.custom_cities = {}

    # Add custom cities to the global CITIES dict
    if st.session_state.custom_cities:
        add_custom_cities(st.session_state.custom_cities)

    all_cities = {**CITIES, **st.session_state.custom_cities}

    city = st.selectbox("City", list(all_cities.keys()), index=0)

    st.subheader("Income")
    starting_tc = st.slider(
        "Starting Total Compensation ($)",
        min_value=100000, max_value=500000, value=200000, step=10000,
        format="$%d"
    )

    st.subheader("Starting Balances")
    col1, col2 = st.columns(2)
    with col1:
        seed_taxable = st.number_input("Taxable ($)", 0, 1000000, 0, step=10000)
        seed_401k = st.number_input("401(k) ($)", 0, 1000000, 0, step=10000)
    with col2:
        seed_roth = st.number_input("Roth IRA ($)", 0, 500000, 0, step=1000)
        seed_hsa = st.number_input("HSA ($)", 0, 100000, 0, step=1000)

    seed_total = seed_taxable + seed_401k + seed_roth + seed_hsa
    st.metric("Total Starting Seed", f"${seed_total:,.0f}")

    st.subheader("Family")
    marriage_age = st.slider("Marriage Age", 25, 40, 29)

    num_kids = st.radio("Number of Kids", [0, 1, 2], index=2, horizontal=True)
    kid_ages = ()
    if num_kids >= 1:
        kid1_age = st.slider("First Kid Born (Your Age)", 26, 45, 31)
        kid_ages = (kid1_age,)
        if num_kids == 2:
            kid2_age = st.slider("Second Kid Born (Your Age)", kid1_age, 47, max(33, kid1_age + 2))
            kid_ages = (kid1_age, kid2_age)

    st.subheader("Spouse Income")
    spouse_works = st.toggle("Spouse Works", value=True)

    if spouse_works:
        spouse_salary = st.slider("Spouse Salary ($)", 0, 200000, 80000, step=5000, format="$%d")
        part_time_fraction = st.slider("Part-time Fraction", 0.0, 1.0, 0.5, step=0.1,
                                       help="After kids start school")
    else:
        spouse_salary = 0
        part_time_fraction = 0.5

    st.subheader("Housing")
    cfg = all_cities[city]
    buy_home = st.toggle("Buy Home", value=cfg.home_price is not None,
                         disabled=cfg.home_price is None,
                         help="Not available in SF (no home price configured)")

    st.subheader("Retirement Planning")
    life_expectancy = st.slider(
        "Life Expectancy",
        min_value=75, max_value=100, value=LIFE_EXPECTANCY, step=1,
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
    part_time_fraction=part_time_fraction,
)

@st.cache_data
def run_sim(starting_tc, city, n_sims, seed_taxable, seed_401k, seed_roth, seed_hsa,
            marriage_age, kid_ages, spouse_works, spouse_salary, part_time_fraction, life_exp):
    seed = SeedAmounts(taxable=seed_taxable, t401k=seed_401k, roth=seed_roth, hsa=seed_hsa)
    family = FamilyConfig(
        marriage_age=marriage_age, kid_ages=kid_ages, spouse_works=spouse_works,
        spouse_salary=spouse_salary, part_time_fraction=part_time_fraction
    )
    rng = np.random.default_rng(42)
    return run_vectorized(starting_tc, city, n_sims, rng, seed_amounts=seed,
                          family_config=family, return_trajectories=True, life_expectancy=life_exp)

with st.spinner("Running simulation..."):
    results: SimulationResults = run_sim(
        starting_tc, city, n_sims, seed_taxable, seed_401k, seed_roth, seed_hsa,
        marriage_age, kid_ages, spouse_works, spouse_salary, part_time_fraction, life_expectancy
    )

# =============================================================================
# KEY METRICS
# =============================================================================
fire_ages_valid = results.fire_ages[results.fire_ages < 99]
pct_fire = len(fire_ages_valid) / len(results.fire_ages) * 100
median_fire_age = np.median(fire_ages_valid) if len(fire_ages_valid) > 0 else None
p10_fire_age = np.percentile(fire_ages_valid, 10) if len(fire_ages_valid) > 0 else None
p90_fire_age = np.percentile(fire_ages_valid, 90) if len(fire_ages_valid) > 0 else None

# Success = FIRE'd AND portfolio survived to life expectancy
n_sims_total = len(results.fire_ages)
n_fired = (results.fire_ages < 99).sum()
n_survived = ((results.fire_ages < 99) & ~results.failed).sum()
pct_success = n_survived / n_sims_total * 100
pct_failed_after_fire = (n_fired - n_survived) / max(n_fired, 1) * 100

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("FIRE Rate", f"{pct_fire:.1f}%", help="% of simulations that reached FIRE")
with col2:
    st.metric("Success Rate", f"{pct_success:.1f}%",
              help=f"FIRE'd AND portfolio survived to age {life_expectancy}")
with col3:
    st.metric("Median FIRE Age", f"{median_fire_age:.0f}" if median_fire_age else "N/A")
with col4:
    st.metric("P10 FIRE Age", f"{p10_fire_age:.0f}" if p10_fire_age else "N/A")
with col5:
    st.metric("P90 FIRE Age", f"{p90_fire_age:.0f}" if p90_fire_age else "N/A")

if pct_failed_after_fire > 0:
    st.warning(f"⚠️ {pct_failed_after_fire:.1f}% of FIRE'd simulations ran out of money before age {life_expectancy}")
    with st.expander("Why do some simulations fail?"):
        st.markdown(f"""
        **This small failure rate ({pct_failed_after_fire:.1f}%) is actually expected and realistic!**

        Even though the simulation uses dynamic, conservative safe withdrawal rates (SWR) based on retirement length:
        - Someone FIREing at 40 with 50 years left uses ~2.5% SWR (very conservative)
        - Someone FIREing at 50 with 40 years left uses ~3.1% SWR

        **Failures occur due to:**
        1. **Sequence of Returns Risk**: Hitting multiple bad market years early in retirement
        2. **Barely Meeting FIRE**: Portfolios right at the threshold have less buffer
        3. **Spending Adjustments Lag**: The 10% spending cuts when withdrawal rate > 5% may not be enough in extreme scenarios

        This {pct_failed_after_fire:.1f}% failure rate shows the simulation is capturing realistic tail risk.
        In practice, you'd have additional levers like Social Security, downsizing, part-time work, etc.
        """)

# =============================================================================
# CHARTS
# =============================================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "Net Worth Fan", "Cash Flow Sankey", "Spending Deep Dive",
    "Account Composition", "Income vs Spending", "FIRE Histogram",
    "Outcome Distributions", "FIRE Probability", "Sensitivity"
])

# Chart 1: Net Worth Fan Chart
with tab1:
    st.subheader("Net Worth Trajectory")

    ages = results.ages
    nw = results.net_worth

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
    st.subheader("Cash Flow Sankey Diagram")

    # Life stage selector
    life_stage = st.radio(
        "Select Life Stage",
        ["Pre-Marriage", "Early Family", "Peak Earning", "Pre-FIRE"],
        horizontal=True,
        help="Shows median cash flows for the selected life stage"
    )

    # Define age ranges for each life stage
    stage_ranges = {
        "Pre-Marriage": (CURRENT_AGE, marriage_age - 1),
        "Early Family": (marriage_age, marriage_age + 10),
        "Peak Earning": (marriage_age + 10, min(marriage_age + 20, FIRE_HORIZON)),
        "Pre-FIRE": (max(int(median_fire_age) - 5, marriage_age + 15) if median_fire_age else 40,
                     int(median_fire_age) if median_fire_age else 45)
    }

    start_age, end_age = stage_ranges[life_stage]

    # Get indices for the age range
    age_mask = (results.ages >= start_age) & (results.ages <= end_age)
    age_indices = np.where(age_mask)[0]

    if len(age_indices) > 0:
        # Calculate median flows across the period
        income_flow = np.median(results.incomes[age_indices, :])
        taxes_flow = np.median(results.taxes[age_indices, :])

        # Spending categories
        housing_flow = np.median(results.spending_housing[age_indices, :])
        disc_flow = np.median(results.spending_discretionary[age_indices, :])
        kids_flow = np.median(results.spending_kids[age_indices, :])
        edu_flow = np.median(results.spending_education[age_indices, :])
        hc_flow = np.median(results.spending_healthcare[age_indices, :])
        ot_flow = np.median(results.spending_one_time[age_indices, :])

        # Savings categories
        savings_401k_flow = np.median(results.savings_401k[age_indices, :])
        savings_roth_flow = np.median(results.savings_roth[age_indices, :])
        savings_hsa_flow = np.median(results.savings_hsa[age_indices, :])
        savings_taxable_flow = np.median(results.savings_taxable[age_indices, :])

        # Build Sankey diagram
        # Nodes: Income, Taxes, Net Income, Spending categories, Savings categories
        nodes = [
            "Gross Income",      # 0
            "Taxes",             # 1
            "Net Income",        # 2
            "Housing",           # 3
            "Discretionary",     # 4
            "Healthcare",        # 5
            "Kids",              # 6
            "Education (529)",   # 7
            "One-Time",          # 8
            "401(k) Savings",    # 9
            "Roth IRA Savings",  # 10
            "HSA Savings",       # 11
            "Taxable Savings"    # 12
        ]

        # Links: source, target, value, color
        links = []

        # Income -> Taxes
        if taxes_flow > 0:
            links.append((0, 1, taxes_flow, 'rgba(239, 85, 59, 0.4)'))

        # Income -> Net Income
        net_income = income_flow - taxes_flow
        if net_income > 0:
            links.append((0, 2, net_income, 'rgba(99, 110, 250, 0.4)'))

        # Net Income -> Spending categories
        if housing_flow > 0:
            links.append((2, 3, housing_flow, 'rgba(255, 161, 90, 0.4)'))
        if disc_flow > 0:
            links.append((2, 4, disc_flow, 'rgba(171, 99, 250, 0.4)'))
        if hc_flow > 0:
            links.append((2, 5, hc_flow, 'rgba(255, 99, 132, 0.4)'))
        if kids_flow > 0:
            links.append((2, 6, kids_flow, 'rgba(255, 206, 86, 0.4)'))
        if edu_flow > 0:
            links.append((2, 7, edu_flow, 'rgba(75, 192, 192, 0.4)'))
        if ot_flow > 0:
            links.append((2, 8, ot_flow, 'rgba(153, 102, 255, 0.4)'))

        # Net Income -> Savings categories
        if savings_401k_flow > 0:
            links.append((2, 9, savings_401k_flow, 'rgba(0, 204, 150, 0.5)'))
        if savings_roth_flow > 0:
            links.append((2, 10, savings_roth_flow, 'rgba(0, 204, 150, 0.5)'))
        if savings_hsa_flow > 0:
            links.append((2, 11, savings_hsa_flow, 'rgba(0, 204, 150, 0.5)'))
        if savings_taxable_flow > 0:
            links.append((2, 12, savings_taxable_flow, 'rgba(0, 204, 150, 0.5)'))

        # Filter out zero flows and unpack
        links = [l for l in links if l[2] > 0]
        sources, targets, values, colors = zip(*links) if links else ([], [], [], [])

        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=nodes,
                color=['#636EFA', '#EF553B', '#636EFA', '#FFA15A', '#AB63FA',
                       '#FF6384', '#FFCE56', '#4BC0C0', '#9966FF', '#00CC96',
                       '#00CC96', '#00CC96', '#00CC96']
            ),
            link=dict(
                source=list(sources),
                target=list(targets),
                value=list(values),
                color=list(colors)
            )
        )])

        fig.update_layout(
            title=f"Median Annual Cash Flow (Age {start_age}-{end_age})",
            font_size=12,
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)

        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Gross Income", f"${income_flow:,.0f}")
        with col2:
            st.metric("Total Spending", f"${housing_flow + disc_flow + kids_flow + edu_flow + hc_flow + ot_flow:,.0f}")
        with col3:
            total_savings = savings_401k_flow + savings_roth_flow + savings_hsa_flow + savings_taxable_flow
            savings_rate = total_savings / net_income * 100 if net_income > 0 else 0
            st.metric("Savings Rate", f"{savings_rate:.1f}%")
    else:
        st.warning("No data available for selected life stage")

# Chart 3: Spending Deep Dive
with tab3:
    st.subheader("Spending Deep Dive")

    subtab1, subtab2 = st.tabs(["Timeline", "Life Stage Comparison"])

    # Subtab A: Timeline - Stacked area chart
    with subtab1:
        st.caption("Median spending by category over time")

        ages = results.ages
        housing_med = np.median(results.spending_housing, axis=1)
        disc_med = np.median(results.spending_discretionary, axis=1)
        hc_med = np.median(results.spending_healthcare, axis=1)
        kids_med = np.median(results.spending_kids, axis=1)
        edu_med = np.median(results.spending_education, axis=1)
        ot_med = np.median(results.spending_one_time, axis=1)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=ages, y=housing_med, name='Housing',
            mode='lines', stackgroup='one', fillcolor='rgba(255, 161, 90, 0.7)',
            hovertemplate='Age %{x}<br>Housing: $%{y:,.0f}<extra></extra>'
        ))
        fig.add_trace(go.Scatter(
            x=ages, y=disc_med, name='Discretionary',
            mode='lines', stackgroup='one', fillcolor='rgba(171, 99, 250, 0.7)',
            hovertemplate='Age %{x}<br>Discretionary: $%{y:,.0f}<extra></extra>'
        ))
        fig.add_trace(go.Scatter(
            x=ages, y=hc_med, name='Healthcare',
            mode='lines', stackgroup='one', fillcolor='rgba(255, 99, 132, 0.7)',
            hovertemplate='Age %{x}<br>Healthcare: $%{y:,.0f}<extra></extra>'
        ))
        fig.add_trace(go.Scatter(
            x=ages, y=kids_med, name='Kids',
            mode='lines', stackgroup='one', fillcolor='rgba(255, 206, 86, 0.7)',
            hovertemplate='Age %{x}<br>Kids: $%{y:,.0f}<extra></extra>'
        ))
        fig.add_trace(go.Scatter(
            x=ages, y=edu_med, name='Education (529)',
            mode='lines', stackgroup='one', fillcolor='rgba(75, 192, 192, 0.7)',
            hovertemplate='Age %{x}<br>Education: $%{y:,.0f}<extra></extra>'
        ))
        fig.add_trace(go.Scatter(
            x=ages, y=ot_med, name='One-Time',
            mode='lines', stackgroup='one', fillcolor='rgba(153, 102, 255, 0.7)',
            hovertemplate='Age %{x}<br>One-Time: $%{y:,.0f}<extra></extra>'
        ))

        # Mark median FIRE age
        if median_fire_age:
            fig.add_vline(x=median_fire_age, line_dash="dash", line_color="green",
                          annotation_text=f"Median FIRE: {median_fire_age:.0f}")

        fig.update_layout(
            xaxis_title="Age",
            yaxis_title="Spending per Year",
            yaxis=dict(tickformat='$,.0f'),
            hovermode='x unified',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

    # Subtab B: Life Stage Comparison
    with subtab2:
        st.caption("Average spending by category across life stages")

        # Define life stages
        stages = {
            "Pre-Marriage": (CURRENT_AGE, marriage_age - 1),
            "Early Family": (marriage_age, marriage_age + 10),
            "Peak Earning": (marriage_age + 10, min(marriage_age + 20, FIRE_HORIZON)),
            "Pre-FIRE": (max(int(median_fire_age) - 5, marriage_age + 15) if median_fire_age else 40,
                         int(median_fire_age) if median_fire_age else 45)
        }

        categories = ['Housing', 'Discretionary', 'Healthcare', 'Kids', 'Education', 'One-Time']
        stage_data = {cat: [] for cat in categories}
        stage_names = []

        for stage_name, (start_age, end_age) in stages.items():
            age_mask = (results.ages >= start_age) & (results.ages <= end_age)
            age_indices = np.where(age_mask)[0]

            if len(age_indices) > 0:
                stage_names.append(stage_name)
                stage_data['Housing'].append(np.median(results.spending_housing[age_indices, :]))
                stage_data['Discretionary'].append(np.median(results.spending_discretionary[age_indices, :]))
                stage_data['Healthcare'].append(np.median(results.spending_healthcare[age_indices, :]))
                stage_data['Kids'].append(np.median(results.spending_kids[age_indices, :]))
                stage_data['Education'].append(np.median(results.spending_education[age_indices, :]))
                stage_data['One-Time'].append(np.median(results.spending_one_time[age_indices, :]))

        fig = go.Figure()

        colors = ['rgba(255, 161, 90, 0.8)', 'rgba(171, 99, 250, 0.8)', 'rgba(255, 99, 132, 0.8)',
                  'rgba(255, 206, 86, 0.8)', 'rgba(75, 192, 192, 0.8)', 'rgba(153, 102, 255, 0.8)']

        for i, category in enumerate(categories):
            fig.add_trace(go.Bar(
                name=category,
                x=stage_names,
                y=stage_data[category],
                marker_color=colors[i],
                hovertemplate='%{y:$,.0f}<extra></extra>'
            ))

        fig.update_layout(
            barmode='group',
            xaxis_title="Life Stage",
            yaxis_title="Average Spending per Year",
            yaxis=dict(tickformat='$,.0f'),
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

# Chart 6: FIRE Age Histogram
with tab6:
    st.subheader("FIRE Age Distribution")

    fig = go.Figure()

    # Histogram of FIRE ages (excluding 99s)
    fig.add_trace(go.Histogram(
        x=fire_ages_valid,
        nbinsx=30,
        name='FIRE Age',
        marker_color='#636EFA'
    ))

    # Mark median
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
    st.plotly_chart(fig, use_container_width=True)

# Chart 7: Outcome Distributions
with tab7:
    st.subheader("Outcome Distributions")

    outcome_selector = st.selectbox(
        "Select Metric",
        ["Ending Net Worth (at life expectancy)", "Net Worth at FIRE", "Spending at FIRE", "Years Until FIRE"],
        help="Choose which outcome to visualize"
    )

    if outcome_selector == "Ending Net Worth (at life expectancy)":
        # Get net worth at last age
        ending_nw = results.net_worth[-1, :]  # Keep in dollars
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
            xaxis_title=f"Net Worth at Age {life_expectancy} (P1-P99 range shown)",
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

        st.caption(f"📊 Showing P1-P99 range (${p1:,.0f} - ${p99:,.0f}) to focus on main distribution. Excludes {len(valid_data) - len(trimmed_data)} extreme outliers.")
        st.plotly_chart(fig, use_container_width=True)

    elif outcome_selector == "Net Worth at FIRE":
        # Get net worth at FIRE age for each simulation
        fire_nw = []
        for sim_idx in range(results.net_worth.shape[1]):
            fire_age = results.fire_ages[sim_idx]
            if fire_age < 99:  # Only include those who FIRE'd
                # Find the age index
                age_idx = np.where(results.ages == fire_age)[0]
                if len(age_idx) > 0:
                    fire_nw.append(results.net_worth[age_idx[0], sim_idx])

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
                xaxis_title="Net Worth at FIRE (P1-P99 range shown)",
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

            st.caption(f"📊 Showing P1-P99 range (${p1:,.0f} - ${p99:,.0f}) to focus on main distribution. Excludes {len(fire_nw) - len(trimmed_data)} extreme outliers.")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No simulations reached FIRE")

    elif outcome_selector == "Spending at FIRE":
        # Get spending at FIRE age for each simulation
        fire_spending = []
        for sim_idx in range(results.spending.shape[1]):
            fire_age = results.fire_ages[sim_idx]
            if fire_age < 99:  # Only include those who FIRE'd
                # Find the age index
                age_idx = np.where(results.ages == fire_age)[0]
                if len(age_idx) > 0:
                    fire_spending.append(results.spending[age_idx[0], sim_idx])

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
                xaxis_title="Annual Spending at FIRE (P1-P99 range shown)",
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

            st.caption(f"📊 Showing P1-P99 range (${p1:,.0f} - ${p99:,.0f}) to focus on main distribution. Excludes {len(fire_spending) - len(trimmed_data)} extreme outliers.")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No simulations reached FIRE")

    elif outcome_selector == "Years Until FIRE":
        years_to_fire = fire_ages_valid - CURRENT_AGE

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
with tab4:
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

# Chart 5: Income vs Spending
with tab5:
    st.subheader("Income vs Spending Trajectories (Median)")

    ages = results.ages
    income_med = np.median(results.incomes, axis=1)
    spending_med = np.median(results.spending, axis=1)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=ages, y=income_med,
        mode='lines', name='Household Income',
        line=dict(color='#00CC96', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=ages, y=spending_med,
        mode='lines', name='Total Spending',
        line=dict(color='#EF553B', width=2)
    ))

    # Fill the savings gap
    fig.add_trace(go.Scatter(
        x=np.concatenate([ages, ages[::-1]]),
        y=np.concatenate([income_med, spending_med[::-1]]),
        fill='toself', fillcolor='rgba(0, 204, 150, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Savings', showlegend=True
    ))

    # Mark median FIRE age
    if median_fire_age:
        fig.add_vline(x=median_fire_age, line_dash="dash", line_color="green",
                      annotation_text=f"Median FIRE: {median_fire_age:.0f}")

    fig.update_layout(
        xaxis_title="Age",
        yaxis_title="Amount per Year",
        yaxis=dict(tickformat='$,.0f'),
        hovermode='x unified',
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

# Chart 8: FIRE Probability Curve (CDF)
with tab8:
    st.subheader("Cumulative FIRE Probability by Age")

    ages_range = np.arange(CURRENT_AGE, FIRE_HORIZON + 1)
    cdf = [(results.fire_ages <= age).mean() * 100 for age in ages_range]

    # Also run comparison with zero seed
    @st.cache_data
    def run_comparison_sim(starting_tc, city, n_sims, marriage_age, kid_ages, spouse_works, spouse_salary, part_time_fraction, life_exp):
        family = FamilyConfig(
            marriage_age=marriage_age, kid_ages=kid_ages, spouse_works=spouse_works,
            spouse_salary=spouse_salary, part_time_fraction=part_time_fraction
        )
        rng = np.random.default_rng(42)
        fire_ages, _, _ = run_vectorized(starting_tc, city, n_sims, rng, seed_amounts=SeedAmounts(),
                              family_config=family, return_trajectories=False, life_expectancy=life_exp)
        return fire_ages

    zero_seed_ages = run_comparison_sim(
        starting_tc, city, n_sims, marriage_age, kid_ages, spouse_works, spouse_salary, part_time_fraction, life_expectancy
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

# Chart 9: Sensitivity Tornado
with tab9:
    st.subheader("Sensitivity Analysis: Impact on Median FIRE Age")
    st.caption("Shows how ±20% change in each parameter affects median FIRE age")

    if median_fire_age is None:
        st.warning("Not enough simulations reached FIRE to perform sensitivity analysis")
    else:
        @st.cache_data
        def run_sensitivity(base_tc, city, n_sims, seed_taxable, seed_401k, seed_roth, seed_hsa,
                           marriage_age, kid_ages, spouse_works, spouse_salary, part_time_fraction):
            results = {}
            base_params = {
                'starting_tc': base_tc,
                'seed_taxable': seed_taxable,
                'spouse_salary': spouse_salary if spouse_works else 0,
            }

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
                    tc = val if param == 'starting_tc' else base_tc
                    rng = np.random.default_rng(42)
                    fire_ages, _, _ = run_vectorized(tc, city, n_sims, rng, seed_amounts=seed,
                                              family_config=family, return_trajectories=False)
                    valid = fire_ages[fire_ages < 99]
                    ages_list.append(np.median(valid) if len(valid) > 0 else 99)

                results[param] = (ages_list[0], ages_list[1])  # (low_result, high_result)

            return results

        sensitivity = run_sensitivity(
            starting_tc, city, n_sims, seed_taxable, seed_401k, seed_roth, seed_hsa,
            marriage_age, kid_ages, spouse_works, spouse_salary, part_time_fraction
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
                         city_names):
    ages_to_check = [40, 42, 44, 46, 48, 50, 52, 55]
    seed = SeedAmounts(taxable=seed_taxable, t401k=seed_401k, roth=seed_roth, hsa=seed_hsa)
    family = FamilyConfig(
        marriage_age=marriage_age, kid_ages=kid_ages, spouse_works=spouse_works,
        spouse_salary=spouse_salary, part_time_fraction=part_time_fraction
    )

    data = []
    for city_name in city_names:
        row = {'City': city_name}
        for ta in ages_to_check:
            tc = find_min_tc(city_name, ta, 90, seed_amounts=seed, family_config=family)
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
            tuple(all_cities.keys())
        )
    st.caption("Baseline: $0 starting seed, married at 29, 2 kids (ages 31, 33), spouse works ($80K, 50% part-time after kids)")
else:
    # Use current user settings
    if st.button("Calculate with Your Settings", type="primary"):
        with st.spinner("Computing personalized minimum TC table..."):
            min_tc_data = compute_min_tc_table(
                seed_taxable, seed_401k, seed_roth, seed_hsa,
                marriage_age, kid_ages, spouse_works, spouse_salary, part_time_fraction,
                tuple(all_cities.keys())
            )
            st.session_state.personalized_table = min_tc_data
    
    if 'personalized_table' in st.session_state:
        min_tc_data = st.session_state.personalized_table
        st.caption(f"Using your settings: ${seed_total:,.0f} seed, {len(kid_ages)} kids, spouse {'works' if spouse_works else 'stays home'}")
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

