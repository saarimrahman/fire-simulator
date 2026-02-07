import numpy as np
import time
from dataclasses import dataclass
from typing import Optional
from tabulate import tabulate

# Table format: 'simple', 'presto', 'plain', 'grid', 'github', 'pipe', etc.
TABLE_FMT = 'simple'

np.random.seed(42)
N_SIMS = 10_000

CURRENT_AGE = 25
FIRE_HORIZON = 60      # Age up to which we check for FIRE eligibility
LIFE_EXPECTANCY = 90   # Simulation runs through retirement until this age
N_YEARS = LIFE_EXPECTANCY - CURRENT_AGE + 1
SWR_DEFAULT = 0.04  # Only used as fallback

def calc_swr(fire_age: int, life_expectancy: int) -> float:
    """Calculate safe withdrawal rate based on retirement length.

    Based on research showing 4% works for ~30 years, with adjustments:
    - Shorter retirements can use higher rates
    - Longer retirements (early FIRE) need lower rates
    """
    years = life_expectancy - fire_age
    if years <= 0:
        return 0.05
    # Approximate: 1/(years * 0.8) gives ~4.2% for 30 years
    # Clamp between 3% (very long) and 5% (short)
    return max(0.03, min(0.05, 1.0 / (years * 0.8)))

INFLATION = 0.03
FOUR01K_LIMIT = 23000
ROTH_IRA_LIMIT = 7000
HSA_FAMILY_LIMIT = 8300
COLLEGE_COST_PER_KID = 300000
COLLEGE_YEARS = 4
COLLEGE_ANNUAL_COST = COLLEGE_COST_PER_KID / COLLEGE_YEARS
INVESTMENT_RETURN_529 = 0.06
KID1_BORN_AGE = 31
KID2_BORN_AGE = 33
KID1_COLLEGE_AGE = KID1_BORN_AGE + 18
KID2_COLLEGE_AGE = KID2_BORN_AGE + 18
HEALTH_SHOCK_PROB = 0.15
HEALTH_SHOCK_COST = 20000
CONTRIB_PER_KID = 300000 * 0.06 / ((1.06)**18 - 1)

@dataclass
class CityConfig:
    """Configuration for a city's housing and tax parameters."""
    one_br_rent: float
    nice_one_br_rent: float
    family_rent: float
    home_price: Optional[float]
    down_payment_pct: float
    mortgage_rate: float
    property_tax_rate: float
    home_maintenance_pct: float
    insurance_premium: float
    insurance_inflation: float
    utility_premium: float
    home_appreciation: float
    state_tax_rate: float
    retirement_state_tax: float

CITIES = {
    'Sacramento': CityConfig(
        one_br_rent=4500,
        nice_one_br_rent=5000,
        family_rent=2400,
        home_price=535000,
        down_payment_pct=0.20,
        mortgage_rate=0.065,
        property_tax_rate=0.011,
        home_maintenance_pct=0.01,
        insurance_premium=6000,
        insurance_inflation=0.08,
        utility_premium=3600,
        home_appreciation=0.03,
        state_tax_rate=0.06,
        retirement_state_tax=0.05,
    ),
    'Dublin': CityConfig(
        one_br_rent=4500,
        nice_one_br_rent=5000,
        family_rent=3600,
        home_price=1300000,
        down_payment_pct=0.20,
        mortgage_rate=0.065,
        property_tax_rate=0.011,
        home_maintenance_pct=0.01,
        insurance_premium=7200,
        insurance_inflation=0.08,
        utility_premium=3600,
        home_appreciation=0.04,
        state_tax_rate=0.06,
        retirement_state_tax=0.05,
    ),
    'San Francisco': CityConfig(
        one_br_rent=4500,
        nice_one_br_rent=5000,
        family_rent=5900,
        home_price=None,
        down_payment_pct=0,
        mortgage_rate=0.065,
        property_tax_rate=0,
        home_maintenance_pct=0,
        insurance_premium=0,
        insurance_inflation=0,
        utility_premium=2400,
        home_appreciation=0,
        state_tax_rate=0.06,
        retirement_state_tax=0.05,
    ),
}

def add_custom_cities(custom_cities_dict):
    """Add or update cities in the global CITIES dict.

    Args:
        custom_cities_dict: dict of {city_name: CityConfig}
    """
    global CITIES
    CITIES.update(custom_cities_dict)

def get_all_cities():
    """Return the current CITIES dict."""
    return CITIES

def calc_taxes_vec(gross, state_rate, t401k=0, hsa_c=0):
    agi = gross - t401k - hsa_c
    fica = np.minimum(gross, 168600) * 0.0765 + np.maximum(0, gross - 168600) * 0.0145
    taxable = np.maximum(0, agi - 29200)
    brackets = [(23200,0.10),(71000,0.12),(106750,0.22),(182750,0.24),(103550,0.32),(243750,0.35),(1e15,0.37)]
    federal = np.zeros_like(gross, dtype=float)
    rem = taxable.copy()
    for w, r in brackets:
        federal += np.minimum(rem, w) * r
        rem = np.maximum(rem - w, 0)
    state = np.maximum(0, agi - 14600) * state_rate
    return fica + federal + state

@dataclass
class SeedAmounts:
    """Starting balances for different account types."""
    taxable: float = 0
    t401k: float = 0
    roth: float = 0
    hsa: float = 0

@dataclass
class SimulationResults:
    """Full trajectory data from simulation runs."""
    fire_ages: np.ndarray           # (N,) FIRE age per simulation
    ages: np.ndarray                # (N_YEARS,) age at each year
    incomes: np.ndarray             # (N_YEARS, N) household income
    spending: np.ndarray            # (N_YEARS, N) total spending
    taxable: np.ndarray             # (N_YEARS, N) taxable account balance
    t401k: np.ndarray               # (N_YEARS, N) 401k balance
    roth: np.ndarray                # (N_YEARS, N) Roth IRA balance
    hsa: np.ndarray                 # (N_YEARS, N) HSA balance
    home_equity: np.ndarray         # (N_YEARS, N) home value - mortgage
    net_worth: np.ndarray           # (N_YEARS, N) total net worth
    fired_status: np.ndarray        # (N_YEARS, N) whether FIRE'd at each year
    failed: np.ndarray              # (N,) whether portfolio hit $0 before life expectancy
    failure_ages: np.ndarray        # (N,) age at which portfolio failed (99 if survived)
    # Spending breakdown by category
    spending_housing: np.ndarray    # (N_YEARS, N) housing costs
    spending_discretionary: np.ndarray  # (N_YEARS, N) discretionary/lifestyle
    spending_kids: np.ndarray       # (N_YEARS, N) child-rearing costs
    spending_education: np.ndarray  # (N_YEARS, N) 529 contributions
    spending_healthcare: np.ndarray # (N_YEARS, N) healthcare costs
    spending_one_time: np.ndarray   # (N_YEARS, N) one-time expenses
    # Cash flow breakdown
    taxes: np.ndarray               # (N_YEARS, N) total taxes paid
    savings_401k: np.ndarray        # (N_YEARS, N) 401k contributions
    savings_roth: np.ndarray        # (N_YEARS, N) Roth IRA contributions
    savings_hsa: np.ndarray         # (N_YEARS, N) HSA contributions
    savings_taxable: np.ndarray     # (N_YEARS, N) taxable account contributions

@dataclass
class FamilyConfig:
    """Configuration for marriage, kids, and spouse income."""
    # Marriage & kids
    marriage_age: int = 29
    kid_ages: tuple = (31, 33)            # Ages when kids are born (empty tuple = no kids)

    # Spouse work pattern
    spouse_works: bool = True             # Easy toggle to disable spouse income entirely
    spouse_salary: float = 80000          # Base full-time salary at marriage
    spouse_salary_growth: float = 0.03    # Annual growth rate
    part_time_fraction: float = 0.5       # Part-time as fraction of full-time

    # Timeline: spouse stops working before first kid, resumes part-time when youngest starts school
    work_gap_before_first_kid: int = 1    # Years before first kid to stop working
    kid_school_age: int = 5               # Age when kids start school

    # Kid costs
    college_cost_per_kid: float = 300000
    annual_cost_per_kid: float = 8000

def calc_spouse_income(age: int, cfg: FamilyConfig, inflation: float, noise: float = 1.0) -> float:
    """Calculate spouse income for a given age based on family configuration.

    Work pattern:
    - Before marriage: $0
    - Marriage → 1yr before first kid: Full-time
    - 1yr before first kid → last kid turns 5: $0 (child-rearing gap)
    - Last kid turns 5 onward: Part-time permanently
    """
    if not cfg.spouse_works or age < cfg.marriage_age:
        return 0.0

    if cfg.kid_ages:
        gap_start = min(cfg.kid_ages) - cfg.work_gap_before_first_kid
        gap_end = max(cfg.kid_ages) + cfg.kid_school_age

        if gap_start <= age < gap_end:
            return 0.0  # Child-rearing gap
        elif age >= gap_end:
            fraction = cfg.part_time_fraction  # Part-time after kids in school
        else:
            fraction = 1.0  # Full-time before kids
    else:
        fraction = 1.0  # No kids = always full-time

    years_since_marriage = age - cfg.marriage_age
    salary = cfg.spouse_salary * ((1 + cfg.spouse_salary_growth) ** years_since_marriage)
    return salary * fraction * inflation * noise

def run_vectorized(starting_tc, city_name, n_sims, rng, seed_amounts=None, family_config=None,
                   return_trajectories=False, life_expectancy=None):
    """
    seed_amounts: SeedAmounts dataclass or dict with dollar amounts per account, e.g.
        SeedAmounts(taxable=165000, t401k=75000, roth=45000, hsa=15000)
        or {'taxable': 165000, '401k': 75000, 'roth': 45000, 'hsa': 15000}
    family_config: FamilyConfig dataclass for spouse/kid settings
    return_trajectories: if True, return SimulationResults with full trajectory data
    life_expectancy: age to simulate through (default: LIFE_EXPECTANCY constant)
    """
    if life_expectancy is None:
        life_expectancy = LIFE_EXPECTANCY
    n_years = life_expectancy - CURRENT_AGE + 1
    N = n_sims; cfg = CITIES[city_name]
    infl = (1 + INFLATION) ** np.arange(n_years)

    # Initialize trajectory arrays if needed
    if return_trajectories:
        traj_incomes = np.zeros((n_years, N))
        traj_spending = np.zeros((n_years, N))
        traj_taxable = np.zeros((n_years, N))
        traj_t401k = np.zeros((n_years, N))
        traj_roth = np.zeros((n_years, N))
        traj_hsa = np.zeros((n_years, N))
        traj_home_equity = np.zeros((n_years, N))
        traj_net_worth = np.zeros((n_years, N))
        traj_fired = np.zeros((n_years, N), dtype=bool)
        # Spending breakdown
        traj_spending_housing = np.zeros((n_years, N))
        traj_spending_disc = np.zeros((n_years, N))
        traj_spending_kids = np.zeros((n_years, N))
        traj_spending_education = np.zeros((n_years, N))
        traj_spending_healthcare = np.zeros((n_years, N))
        traj_spending_one_time = np.zeros((n_years, N))
        # Cash flow breakdown
        traj_taxes = np.zeros((n_years, N))
        traj_savings_401k = np.zeros((n_years, N))
        traj_savings_roth = np.zeros((n_years, N))
        traj_savings_hsa = np.zeros((n_years, N))
        traj_savings_taxable = np.zeros((n_years, N))

    if seed_amounts is None:
        seed_amounts = SeedAmounts()
    elif isinstance(seed_amounts, dict):
        # Convert dict to SeedAmounts for backward compatibility
        seed_amounts = SeedAmounts(
            taxable=seed_amounts.get('taxable', 0),
            t401k=seed_amounts.get('401k', 0),
            roth=seed_amounts.get('roth', 0),
            hsa=seed_amounts.get('hsa', 0)
        )

    if family_config is None:
        family_config = FamilyConfig()

    # Derive kid ages from config
    kid_ages = sorted(family_config.kid_ages) if family_config.kid_ages else []
    kid1_born = kid_ages[0] if len(kid_ages) >= 1 else 999
    kid2_born = kid_ages[1] if len(kid_ages) >= 2 else 999
    kid1_college = kid1_born + 18
    kid2_college = kid2_born + 18
    contrib_per_kid = family_config.college_cost_per_kid * 0.06 / ((1.06)**18 - 1)

    income_noise = rng.normal(0, 0.02, size=(n_years, N))
    spouse_noise = np.maximum(rng.normal(1.0, 0.1, size=(n_years, N)), 0.5)
    spouse_works_roll = rng.random(N) < 0.90  # 90% chance spouse works when in working period
    recession = rng.random(size=(n_years, N)) < 0.15
    mr_arr = np.where(recession, rng.normal(-0.10, 0.12, size=(n_years, N)),
                      rng.normal(0.09, 0.14, size=(n_years, N)))
    jl_roll = rng.random(size=(n_years, N))
    jl_thresh = np.where(recession, 0.15, 0.03)
    hs_roll = rng.random(size=(n_years, N))

    tc = np.full(N, starting_tc, dtype=float)
    incomes = np.zeros((n_years, N))
    # Build income trajectory up to FIRE_HORIZON (working years)
    for i, age in enumerate(range(CURRENT_AGE, FIRE_HORIZON + 1)):
        if age == 27: tc *= 1.15
        elif age == 30: tc *= 1.20
        elif age == 34: tc *= 1.10
        elif age == 38: tc *= 1.10
        if i > 0 and age not in [27, 30, 34, 38]:
            tc *= np.maximum((1.01 + income_noise[i]) * (1 + INFLATION), 0.98)
        incomes[i] = tc.copy()
        # Add spouse income based on family config
        spouse_inc = calc_spouse_income(age, family_config, infl[i], noise=1.0)
        incomes[i] += spouse_inc * spouse_noise[i] * spouse_works_roll
    incomes = np.where(jl_roll < jl_thresh, incomes * 0.5, incomes)

    # SEEDED starting balances (explicit dollar amounts per account)
    taxable = np.full(N, float(seed_amounts.taxable))
    t401k = np.full(N, float(seed_amounts.t401k))
    roth = np.full(N, float(seed_amounts.roth))
    roth_basis = np.full(N, float(seed_amounts.roth))  # assume all Roth is basis at start
    hsa_bal = np.full(N, float(seed_amounts.hsa))

    c529_1 = np.zeros(N); c529_2 = np.zeros(N)
    home_val = np.zeros(N); mortgage = np.zeros(N)
    owns_home = np.zeros(N, dtype=bool); fired = np.zeros(N, dtype=bool)
    fire_ages = np.full(N, 99, dtype=int); fixed_pmt = np.zeros(N)
    ret_base_spend = np.zeros(N)
    has_home = cfg.home_price is not None

    # Track portfolio failures (ran out of money before life expectancy)
    failed = np.zeros(N, dtype=bool)
    failure_ages = np.full(N, 99, dtype=int)

    for i, age in enumerate(range(CURRENT_AGE, life_expectancy + 1)):
        ye = age - CURRENT_AGE; inf = infl[ye]; mr = mr_arr[i]
        # Income is zero after FIRE or after FIRE_HORIZON (forced retirement)
        working_years_idx = min(i, FIRE_HORIZON - CURRENT_AGE)
        year_inc = np.where(fired | (age > FIRE_HORIZON), 0.0, incomes[working_years_idx])

        if age < 28:
            housing = cfg.one_br_rent*12*inf*np.ones(N); disc = 30000*inf*np.ones(N); st = 0.055
        elif age < 31:
            housing = cfg.nice_one_br_rent*12*inf*np.ones(N); disc = 40000*inf*np.ones(N); st = 0.055
        else:
            st = cfg.state_tax_rate; disc = 35000*inf*np.ones(N)
            housing = cfg.family_rent*12*inf*np.ones(N)
            if has_home and age == 33:
                pp = cfg.home_price*infl[ye-8]; down = pp*cfg.down_payment_pct
                mortgage[:] = pp-down; taxable -= down; owns_home[:] = True; home_val[:] = pp
                r_m = cfg.mortgage_rate/12
                fixed_pmt[:] = mortgage*r_m*(1+r_m)**360/((1+r_m)**360-1)

        ca_prem = np.zeros(N); utility = cfg.utility_premium*inf if age >= 31 else 0
        if has_home and age >= 33:
            home_val = np.where(owns_home, home_val*(1+cfg.home_appreciation), home_val)
            ann_m = fixed_pmt*12; interest = mortgage*cfg.mortgage_rate
            mortgage = np.maximum(mortgage - np.maximum(np.minimum(ann_m-interest, mortgage), 0), 0)
            pm = home_val*(cfg.property_tax_rate+cfg.home_maintenance_pct)
            ca_prem = np.where(owns_home, cfg.insurance_premium*((1+cfg.insurance_inflation)**(age-33)), 0)
            housing = np.where(owns_home, np.where(mortgage > 0, ann_m+pm, pm), housing)
        housing += utility + ca_prem

        kids = np.zeros(N)
        kid_cost = family_config.annual_cost_per_kid
        if age >= kid1_born: kids += kid_cost*inf
        if age >= kid2_born: kids += kid_cost*inf
        if age >= 36: kids *= 1.5
        # Kids leave home at 22 (college graduation)
        if kid1_born < 999 and age >= kid1_born + 22: kids -= kid_cost*inf*1.5
        if kid2_born < 999 and age >= kid2_born + 22: kids -= kid_cost*inf*1.5
        kids = np.maximum(kids, 0)

        c529c = np.zeros(N)
        college_annual = family_config.college_cost_per_kid / COLLEGE_YEARS
        if kid1_born <= age < kid1_college:
            c = contrib_per_kid*inf; c529_1 = (c529_1+c)*(1+INVESTMENT_RETURN_529); c529c += c
        if kid2_born <= age < kid2_college:
            c = contrib_per_kid*inf; c529_2 = (c529_2+c)*(1+INVESTMENT_RETURN_529); c529c += c
        for off in range(COLLEGE_YEARS):
            if age == kid1_college+off: c529_1 -= np.minimum(college_annual*inf, c529_1)
            if age == kid2_college+off: c529_2 -= np.minimum(college_annual*inf, c529_2)
        if age == kid1_college+COLLEGE_YEARS: taxable += c529_1; c529_1[:] = 0
        if age == kid2_college+COLLEGE_YEARS: taxable += c529_2; c529_2[:] = 0

        if age >= 35: disc += 10000*inf
        if age >= 40: disc += 5000*inf

        ot = np.zeros(N)
        if age == 28: ot += 40000*inf
        if age == 31: ot += 15000*inf
        if age == 33 and has_home: ot += 25000*inf
        if age == 30: ot += 35000*inf
        if age == 38: ot += 40000*inf

        hc = np.where(fired, 24000*inf, np.where(age < 35, 6000*inf, 12000*inf)*np.ones(N))
        hc += (hs_roll[i] < HEALTH_SHOCK_PROB).astype(float) * HEALTH_SHOCK_COST * inf

        total_spend = housing + disc + kids + c529c + ot + hc

        t401k_c = np.where(~fired & (year_inc > 0), np.minimum(FOUR01K_LIMIT*inf, year_inc*0.5), 0)
        hsa_c = np.where(~fired & (year_inc > 0), (HSA_FAMILY_LIMIT if age >= kid1_born else 4150)*inf, 0.0)
        taxes = np.where(~fired, calc_taxes_vec(year_inc, st, t401k_c, hsa_c), 0.0)

        net_inc = year_inc - taxes - total_spend
        wp = (~fired) & (net_inc > 0)
        a401 = np.where(wp, np.minimum(t401k_c, net_inc*0.5), 0)
        ar = np.where(wp, np.minimum(ROTH_IRA_LIMIT*inf, net_inc*0.3), 0)
        ah = np.where(wp, np.minimum(hsa_c, net_inc*0.2), 0)
        t401k += a401; roth += ar; roth_basis += ar; hsa_bal += ah
        taxable += np.where(wp, net_inc-a401-ar-ah, 0)
        taxable += np.where((~fired)&(net_inc<=0), net_inc, 0)

        total_port = taxable + t401k + roth + hsa_bal
        ret_base_spend = np.where(fired & (ret_base_spend==0), total_spend, ret_base_spend)
        wd_rate = np.where(fired & (total_port > 0), ret_base_spend/total_port, 0)
        adj = np.where(wd_rate > 0.05, 0.90, np.where(wd_rate < 0.03, 1.10, 1+INFLATION))
        ret_base_spend = np.where(fired, ret_base_spend*adj, ret_base_spend)

        if age < 60:
            # Pre-60 withdrawal: taxable first, then Roth basis, then HSA, then 401k with penalty
            active_retired = fired & ~failed  # Only withdraw from non-failed portfolios
            draw = np.where(active_retired, ret_base_spend, 0)
            d_tax = np.minimum(draw, np.maximum(taxable, 0)); rem1 = draw - d_tax
            d_roth = np.minimum(rem1, np.maximum(roth_basis, 0)); rem2 = rem1 - d_roth
            d_hsa = np.minimum(rem2*0.5, np.maximum(hsa_bal, 0)); rem3 = rem2 - d_hsa
            pen = rem3*0.10; tax_401 = calc_taxes_vec(rem3, cfg.retirement_state_tax)
            taxable -= d_tax; roth -= d_roth; roth_basis -= d_roth; hsa_bal -= d_hsa
            t401k -= np.where(active_retired, rem3+pen+tax_401, 0)
        else:
            # Post-60 withdrawal: proportional from all accounts
            active_retired = fired & ~failed  # Only withdraw from non-failed portfolios
            sp = np.maximum(total_port, 1); tf = t401k/sp
            rt = np.where(active_retired, calc_taxes_vec(ret_base_spend*tf, cfg.retirement_state_tax), 0)
            td = np.where(active_retired, ret_base_spend+rt, 0)
            taxable -= np.where(active_retired, td*taxable/sp, 0); t401k -= np.where(active_retired, td*t401k/sp, 0)
            roth -= np.where(active_retired, td*roth/sp, 0); hsa_bal -= np.where(active_retired, td*hsa_bal/sp, 0)

        # Apply market returns (skip failed portfolios to avoid overflow)
        hr = (1+mr)**0.5-1; wm = ~fired
        ns = np.where(wm & (net_inc > 0), net_inc, 0)
        active = ~failed
        taxable = np.where(active & wm, (taxable-ns)*(1+mr)+ns*(1+hr), np.where(active, taxable*(1+mr), taxable))
        t401k = np.where(active, t401k*(1+mr), t401k)
        roth = np.where(active, roth*(1+mr), roth)
        hsa_bal = np.where(active, hsa_bal*(1+mr*0.8), hsa_bal)

        total_liq = taxable + t401k + roth + hsa_bal + c529_1 + c529_2
        accessible = taxable + roth_basis + hsa_bal

        # Record trajectory data
        if return_trajectories:
            traj_incomes[i] = incomes[i]
            traj_spending[i] = total_spend
            traj_taxable[i] = taxable.copy()
            traj_t401k[i] = t401k.copy()
            traj_roth[i] = roth.copy()
            traj_hsa[i] = hsa_bal.copy()
            traj_home_equity[i] = np.where(owns_home, home_val - mortgage, 0)
            traj_net_worth[i] = total_liq + np.where(owns_home, home_val - mortgage, 0)
            traj_fired[i] = fired.copy()
            # Spending breakdown
            traj_spending_housing[i] = housing
            traj_spending_disc[i] = disc
            traj_spending_kids[i] = kids
            traj_spending_education[i] = c529c
            traj_spending_healthcare[i] = hc
            traj_spending_one_time[i] = ot
            # Cash flow breakdown
            traj_taxes[i] = taxes
            traj_savings_401k[i] = a401
            traj_savings_roth[i] = ar
            traj_savings_hsa[i] = ah
            traj_savings_taxable[i] = np.where(wp, net_inc - a401 - ar - ah, 0)

        # Check for FIRE eligibility only during working years (up to FIRE_HORIZON)
        if 30 <= age <= FIRE_HORIZON:
            if has_home and age >= 33:
                rh = home_val*(cfg.property_tax_rate+cfg.home_maintenance_pct)
                rh += cfg.insurance_premium*((1+cfg.insurance_inflation)**(age-33))
                rh += cfg.utility_premium*inf
            else:
                rh = cfg.family_rent*12*inf + cfg.utility_premium*inf
            rd = 45000*inf; rhc = 24000*inf + HEALTH_SHOCK_PROB*HEALTH_SHOCK_COST*inf
            r529 = np.zeros(N)
            if age < kid1_college: r529 += contrib_per_kid*inf
            if age < kid2_college: r529 += contrib_per_kid*inf
            rt = rh + rd + rhc + r529 + kids
            # Dynamic SWR based on how long money needs to last
            swr = calc_swr(age, life_expectancy)
            fn = rt / swr
            bridge = rt * max(0, 60-age)
            can_fire = (~fired) & (total_liq >= fn) & (accessible >= bridge)
            fire_ages = np.where(can_fire, age, fire_ages)
            fired = fired | can_fire
            ret_base_spend = np.where(can_fire, rt, ret_base_spend)

        # Track portfolio failures (portfolio went to $0 while retired)
        just_failed = fired & ~failed & (total_port <= 0)
        failure_ages = np.where(just_failed, age, failure_ages)
        failed = failed | just_failed

    if return_trajectories:
        return SimulationResults(
            fire_ages=fire_ages,
            ages=np.arange(CURRENT_AGE, life_expectancy + 1),
            incomes=traj_incomes,
            spending=traj_spending,
            taxable=traj_taxable,
            t401k=traj_t401k,
            roth=traj_roth,
            hsa=traj_hsa,
            home_equity=traj_home_equity,
            net_worth=traj_net_worth,
            fired_status=traj_fired,
            failed=failed,
            failure_ages=failure_ages,
            # Spending breakdown
            spending_housing=traj_spending_housing,
            spending_discretionary=traj_spending_disc,
            spending_kids=traj_spending_kids,
            spending_education=traj_spending_education,
            spending_healthcare=traj_spending_healthcare,
            spending_one_time=traj_spending_one_time,
            # Cash flow breakdown
            taxes=traj_taxes,
            savings_401k=traj_savings_401k,
            savings_roth=traj_savings_roth,
            savings_hsa=traj_savings_hsa,
            savings_taxable=traj_savings_taxable,
        )
    return fire_ages, failed, failure_ages

def find_min_tc(city, target_age, conf_pct, seed_amounts=None, family_config=None, lo=100000, hi=700000, tol=5000):
    while hi - lo > tol:
        mid = round((lo + hi) / 2 / 5000) * 5000
        rng = np.random.default_rng(42)
        fire_ages, _, _ = run_vectorized(mid, city, N_SIMS, rng, seed_amounts=seed_amounts, family_config=family_config)
        pct = (fire_ages <= target_age).mean() * 100
        if pct >= conf_pct:
            hi = mid
        else:
            lo = mid + 5000
    return hi

if __name__ == "__main__":
    # ============================================================
    # MAIN
    # ============================================================
    t0 = time.time()

    # Explicit per-account seed amounts (adjust each independently)
    SEED_AMOUNTS = SeedAmounts(
        taxable=210000,   # general investments / brokerage
        t401k=90000,      # 401k balance
        roth=3500,        # Roth IRA balance
        hsa=0,            # HSA balance
    )
    ZERO_SEED = SeedAmounts()
    SEED_TOTAL = SEED_AMOUNTS.taxable + SEED_AMOUNTS.t401k + SEED_AMOUNTS.roth + SEED_AMOUNTS.hsa

    # Family configuration - adjust these to model different scenarios
    FAMILY_CONFIG = FamilyConfig(
        marriage_age=29,
        kid_ages=(31, 33),              # Ages when kids are born (empty tuple = no kids)
        spouse_works=True,              # Set False to disable spouse income entirely
        spouse_salary=80000,            # Base full-time salary at marriage
        spouse_salary_growth=0.03,      # Annual growth rate
        part_time_fraction=0.5,         # Part-time as fraction of full-time
        work_gap_before_first_kid=1,    # Years before first kid to stop working
        kid_school_age=5,               # Age when youngest kid starts school
    )

    print("=" * 95)
    print(f"FIRE ANALYSIS: ${SEED_TOTAL/1000:.0f}K STARTING SEED vs $0 SEED COMPARISON")
    print("=" * 95)
    print(f"\nStarting seed: ${SEED_TOTAL:,}")
    print(f"  Taxable / investments: ${SEED_AMOUNTS.taxable:>8,}")
    print(f"  401(k):                ${SEED_AMOUNTS.t401k:>8,}")
    print(f"  Roth IRA:              ${SEED_AMOUNTS.roth:>8,}")
    print(f"  HSA:                   ${SEED_AMOUNTS.hsa:>8,}")
    print()
    print(f"Family config:")
    print(f"  Marriage age:          {FAMILY_CONFIG.marriage_age}")
    print(f"  Kids born at ages:     {FAMILY_CONFIG.kid_ages}")
    print(f"  Spouse works:          {FAMILY_CONFIG.spouse_works}")
    if FAMILY_CONFIG.spouse_works:
        print(f"  Spouse salary:         ${FAMILY_CONFIG.spouse_salary:,.0f}")
        print(f"  Part-time fraction:    {FAMILY_CONFIG.part_time_fraction:.0%}")
        gap_start = min(FAMILY_CONFIG.kid_ages) - FAMILY_CONFIG.work_gap_before_first_kid if FAMILY_CONFIG.kid_ages else None
        gap_end = max(FAMILY_CONFIG.kid_ages) + FAMILY_CONFIG.kid_school_age if FAMILY_CONFIG.kid_ages else None
        if gap_start and gap_end:
            print(f"  Work gap:              age {gap_start}-{gap_end} (child-rearing)")
            print(f"  Part-time from:        age {gap_end}+")
    print()
    print(f"  Full stress test: liquidity + CA insurance + health shocks")
    print(f"  {N_SIMS:,} simulations")
    print()

    # ============================================================
    # PART 1: Distribution at $220K TC with $300K seed
    # ============================================================
    seed_label = f"${SEED_TOTAL/1000:.0f}K"
    print("━" * 95)
    print(f"PART 1: FIRE probability at $220K TC — {seed_label} seed vs $0 seed")
    print("━" * 95)
    print()

    ages_check = list(range(38, 61, 2))
    table_data = []

    for city in CITIES:
        for seed, label in [(ZERO_SEED, "$0"), (SEED_AMOUNTS, seed_label)]:
            rng = np.random.default_rng(42)
            fire_ages, failed, _ = run_vectorized(220000, city, N_SIMS, rng, seed_amounts=seed, family_config=FAMILY_CONFIG)
            row = [city if label == "$0" else "", label]
            for a in ages_check:
                pct = (fire_ages <= a).mean() * 100
                row.append(f"{pct:.1f}%")
            # Add success rate (FIRE'd and didn't fail)
            success_rate = ((fire_ages < 99) & ~failed).mean() * 100
            row.append(f"{success_rate:.1f}%")
            table_data.append(row)
        table_data.append([])  # Empty row for spacing

    headers = ["City", "Seed"] + [f"%≤{a}" for a in ages_check] + ["Success"]
    print(tabulate(table_data, headers=headers, tablefmt=TABLE_FMT, stralign="right"))
    print()

    # ============================================================
    # PART 2: How many years does $300K buy you?
    # ============================================================
    print("━" * 95)
    print(f"PART 2: Years gained from {seed_label} seed (median & 90th percentile FIRE age)")
    print("━" * 95)
    print()

    tc_scenarios = [200000, 220000, 250000, 300000]
    table_data = []

    for city in CITIES:
        for tc in tc_scenarios:
            rng0 = np.random.default_rng(42)
            ages0, failed0, _ = run_vectorized(tc, city, N_SIMS, rng0, seed_amounts=ZERO_SEED, family_config=FAMILY_CONFIG)
            rng1 = np.random.default_rng(42)
            ages1, failed1, _ = run_vectorized(tc, city, N_SIMS, rng1, seed_amounts=SEED_AMOUNTS, family_config=FAMILY_CONFIG)

            v0 = ages0[ages0 < 99]; v1 = ages1[ages1 < 99]
            med0 = np.median(v0) if len(v0) > 0 else 99
            med1 = np.median(v1) if len(v1) > 0 else 99
            p90_0 = np.percentile(v0, 90) if len(v0) > 0 else 99
            p90_1 = np.percentile(v1, 90) if len(v1) > 0 else 99

            d_med = med0 - med1 if med0 < 99 and med1 < 99 else float('nan')
            d_p90 = p90_0 - p90_1 if p90_0 < 99 and p90_1 < 99 else float('nan')

            m0 = f"{med0:.0f}" if med0 < 99 else "N/A"
            m1 = f"{med1:.0f}" if med1 < 99 else "N/A"
            p0 = f"{p90_0:.0f}" if p90_0 < 99 else "N/A"
            p1 = f"{p90_1:.0f}" if p90_1 < 99 else "N/A"
            dm = f"-{d_med:.0f}yr" if not np.isnan(d_med) else "N/A"
            dp = f"-{d_p90:.0f}yr" if not np.isnan(d_p90) else "N/A"

            table_data.append([
                city,
                f"${tc/1000:.0f}K",
                m0,
                m1,
                dm,
                p0,
                p1,
                dp
            ])
        table_data.append([])  # Empty row for spacing

    headers = ["City", "TC", f"Med($0)", f"Med({seed_label})", "Δ Med", "90%($0)", f"90%({seed_label})", "Δ 90%"]
    print(tabulate(table_data, headers=headers, tablefmt=TABLE_FMT, stralign="right"))
    print()

    # ============================================================
    # PART 3: Min TC for 90% FIRE at each age — with $300K seed
    # ============================================================
    print("━" * 95)
    print(f"PART 3: Minimum TC for 90% FIRE confidence — $0 seed vs {seed_label} seed")
    print("━" * 95)
    print()

    ages_to_check = [40, 42, 44, 46, 48, 50, 52, 55]

    for seed, label in [(ZERO_SEED, "$0 seed"), (SEED_AMOUNTS, f"{seed_label} seed")]:
        print(f"\n{label}:")
        table_data = []
        for city in CITIES:
            row = [city]
            for ta in ages_to_check:
                tc = find_min_tc(city, ta, 90, seed_amounts=seed, family_config=FAMILY_CONFIG)
                row.append(f"${tc/1000:.0f}K")
            table_data.append(row)

        headers = ["City"] + [f"age {a}" for a in ages_to_check]
        print(tabulate(table_data, headers=headers, tablefmt=TABLE_FMT, stralign="right"))
        print()

    elapsed = time.time() - t0
    print(f"\n\nTotal runtime: {elapsed:.1f}s")