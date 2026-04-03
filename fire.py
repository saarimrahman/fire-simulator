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
HSA_INDIVIDUAL_LIMIT = 4300
HSA_FAMILY_LIMIT = 8550
COLLEGE_COST_PER_KID = 300000
COLLEGE_YEARS = 4
COLLEGE_ANNUAL_COST = COLLEGE_COST_PER_KID / COLLEGE_YEARS
INVESTMENT_RETURN_529 = 0.06
KID1_BORN_AGE = 31
KID2_BORN_AGE = 33
KID1_COLLEGE_AGE = KID1_BORN_AGE + 18
KID2_COLLEGE_AGE = KID2_BORN_AGE + 18
HEALTH_SHOCK_PROB = 0.15
HEALTH_SHOCK_COST = 20_000

# One-time expenses
OT_CAR_MOVE     = 40_000   # Age 28: first car purchase / relocation
OT_BABY_SETUP   = 15_000   # Age of first kid birth: gear, nursery setup
OT_MID_UPGRADE  = 40_000   # Age 38: car replacement / home upgrade
HOME_CLOSING_COSTS = 25_000  # Closing costs when buying a home

# Working-year healthcare (employer-sponsored plan)
HC_YOUNG  = 6_000   # Age < 35: premiums + out-of-pocket
HC_OLDER  = 12_000  # Age 35+: higher premiums as you age off parents' / cheaper plans

# Retirement healthcare
HC_RET_BASE         = 30_000  # ACA marketplace base premium at age 40
HC_RET_AGE_STEP     = 800     # Additional per year above 40 (premiums rise with age)
HC_RET_REAL_GROWTH  = 0.02    # Real annual growth above CPI for ACA premiums
HC_MEDICARE         = 13_000  # Post-65: Medicare premiums + out-of-pocket

# Discretionary spending base (today's dollars, inflation-adjusted in simulation)
DISC_YOUNG  = 30_000  # Age < 28: single, modest lifestyle
DISC_MID    = 40_000  # Age 28–30: nicer apartment, more social spending
DISC_FAMILY = 35_000  # Age 31+: settles down after having kids
DISC_STEP_35 = 10_000  # Extra at 35: travel, hobbies pick up
DISC_STEP_40 =  5_000  # Extra at 40: further lifestyle expansion

# Job loss model
JL_ANNUAL_PROB  = 0.04   # 4% annual probability of layoff
JL_MONTHS_MIN   = 6      # Minimum months unemployed
JL_MONTHS_MAX   = 14     # Maximum months unemployed (exclusive)
JL_REENTRY_MIN  = 0.90   # Worst-case re-entry salary (90% of prior)
JL_REENTRY_MAX  = 1.05   # Best-case re-entry salary (105% of prior)
JL_SEARCH_COST  = 5_000  # Job search costs (flights, courses, wardrobe)

# Market returns
MR_NORMAL_MEAN   =  0.09  # Mean annual return, non-recession years
MR_NORMAL_STD    =  0.14  # Std dev, non-recession years
MR_RECESSION_MEAN = -0.10  # Mean annual return, recession years
MR_RECESSION_STD  =  0.12  # Std dev, recession years
RECESSION_PROB    =  0.15  # Annual probability of recession

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
    city_tax_rate: float = 0.0

CITIES = {
    'Sacramento': CityConfig(
        one_br_rent=1800,
        nice_one_br_rent=2200,
        family_rent=2500,
        home_price=530000,
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
        one_br_rent=2550,
        nice_one_br_rent=3200,
        family_rent=3800,
        home_price=1400000,
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
        one_br_rent=3600,
        nice_one_br_rent=4200,
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
    'New York City': CityConfig(
        one_br_rent=4300,
        nice_one_br_rent=5000,
        family_rent=5500,
        home_price=None,
        down_payment_pct=0,
        mortgage_rate=0.065,
        property_tax_rate=0,
        home_maintenance_pct=0,
        insurance_premium=0,
        insurance_inflation=0,
        utility_premium=3000,
        home_appreciation=0,
        state_tax_rate=0.0685,
        retirement_state_tax=0.06,
        city_tax_rate=0.03876,
    ),
}

def calc_retirement_hc(age, inf, N):
    """Age-graduated retirement healthcare costs.

    Pre-65: ACA marketplace premiums (HC_RET_BASE at 40, +HC_RET_AGE_STEP/yr, rising HC_RET_REAL_GROWTH real above CPI).
    Post-65: Medicare + out-of-pocket (HC_MEDICARE/yr).
    """
    if age < 65:
        base = HC_RET_BASE + max(0, age - 40) * HC_RET_AGE_STEP
        hc_inf = inf * ((1 + HC_RET_REAL_GROWTH) ** max(0, age - 40))
        return base * hc_inf * np.ones(N)
    else:
        return HC_MEDICARE * inf * np.ones(N)

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

def calc_taxes_vec(gross, state_rate, t401k=0, hsa_c=0, inf=1.0, city_tax_rate=0.0):
    agi = gross - t401k - hsa_c
    fica = np.minimum(gross, 168600*inf) * 0.0765 + np.maximum(0, gross - 168600*inf) * 0.0145
    taxable = np.maximum(0, agi - 29200*inf)
    brackets = [(23200*inf,0.10),(71000*inf,0.12),(106750*inf,0.22),(182750*inf,0.24),(103550*inf,0.32),(243750*inf,0.35),(1e15,0.37)]
    federal = np.zeros_like(gross, dtype=float)
    rem = taxable.copy()
    for w, r in brackets:
        federal += np.minimum(rem, w) * r
        rem = np.maximum(rem - w, 0)
    state = np.maximum(0, agi - 14600*inf) * state_rate
    city = np.maximum(0, agi - 12000*inf) * city_tax_rate
    return fica + federal + state + city

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
    ss_income: np.ndarray           # (N_YEARS, N) Social Security income

@dataclass
class FamilyConfig:
    """Configuration for marriage, kids, and spouse income."""
    # Marriage & kids
    marriage_age: int = 29
    kid_ages: tuple = (31, 33)            # Ages when kids are born (empty tuple = no kids)

    # Spouse work pattern
    spouse_works: bool = True             # Easy toggle to disable spouse income entirely
    spouse_salary: float = 80000          # Base full-time salary at marriage
    spouse_salary_growth: float = 0.01    # Real (above-inflation) annual growth rate; infl[i] is applied at call-site
    spouse_soft_cap: float = 150000       # Salary ceiling (growth tapers near this, like main earner)
    part_time_fraction: float = 0.5       # Part-time as fraction of full-time

    # Timeline: spouse stops working before first kid, resumes part-time when youngest starts school
    work_gap_before_first_kid: int = 1    # Years before first kid to stop working
    kid_school_age: int = 5               # Age when kids start school

    # Wedding & kids
    wedding_budget: float = 60000
    college_cost_per_kid: float = 300000
    annual_cost_per_kid: float = 8000


@dataclass
class SocialSecurityConfig:
    """Configuration for Social Security benefits."""
    enabled: bool = True
    claiming_age: int = 67              # FRA (full retirement age), can be 62-70
    monthly_benefit_today: float = 2500 # Estimated monthly benefit in today's dollars at FRA
    spouse_monthly_benefit: float = 1250 # Spouse benefit (often 50% of primary)
    spouse_claiming_age: int = 67
    cola_rate: float = 0.02             # Cost-of-living adjustment (historically ~2%)

    def get_benefit_at_age(self, claiming_age: int) -> float:
        """Adjust benefit for early/late claiming relative to FRA of 67.

        Early claiming (62-66): ~6.67% per year reduction
        Late claiming (68-70): ~8% per year increase (delayed retirement credits)
        """
        fra = 67
        if claiming_age < 62:
            return 0
        years_diff = claiming_age - fra
        if years_diff < 0:
            adjustment = 1 + years_diff * 0.0667  # ~6.67% reduction per year early
        elif years_diff > 0:
            adjustment = 1 + years_diff * 0.08     # 8% increase per year delayed
        else:
            adjustment = 1.0
        return self.monthly_benefit_today * 12 * max(adjustment, 0.0)

    def get_spouse_benefit_at_age(self, claiming_age: int) -> float:
        """Same early/late logic for spouse."""
        fra = 67
        if claiming_age < 62:
            return 0
        years_diff = claiming_age - fra
        if years_diff < 0:
            adjustment = 1 + years_diff * 0.0667
        elif years_diff > 0:
            adjustment = 1 + years_diff * 0.08
        else:
            adjustment = 1.0
        return self.spouse_monthly_benefit * 12 * max(adjustment, 0.0)


@dataclass
class CareerConfig:
    """Configuration for career/salary progression with stochastic promotions."""
    soft_cap: float = 600_000             # TC ceiling (growth tapers near this)
    trajectory: str = "moderate"          # "aggressive", "moderate", "conservative"

    # Promotion windows: (start_age, end_age)
    # Promotions happen stochastically within these windows
    promo_1_window: tuple = (26, 29)      # First major promotion (mid-level → senior)
    promo_2_window: tuple = (29, 33)      # Second major promotion (senior → staff)
    promo_3_window: tuple = (33, 40)      # Third promotion (staff → senior staff)
    promo_4_window: tuple = (40, 50)      # Fourth promotion (principal, rare)

    # Promotion raise ranges (min, max) as multipliers
    promo_raise_range: tuple = (1.12, 1.20)  # 12-20% raise per promotion

    # 401k employer match
    employer_match_pct: float = 0.50      # 50% match (common default)
    employer_match_limit: float = 0.435   # Match up to 43.5% of IRS limit (~$10k)


# Cumulative promotion probabilities by trajectory
# Format: {trajectory: [promo1_prob, promo2_prob, promo3_prob, promo4_prob]}
PROMO_PROBABILITIES = {
    "conservative": [0.80, 0.70, 0.40, 0.10],
    "moderate":     [0.90, 0.85, 0.60, 0.25],
    "aggressive":   [0.95, 0.92, 0.80, 0.45],
}

# Base annual real growth rates by trajectory (before inflation)
BASE_GROWTH_RATES = {
    "conservative": 0.01,   # 1% real + 3% inflation = 4% nominal
    "moderate":     0.015,  # 1.5% real + 3% inflation = 4.5% nominal
    "aggressive":   0.02,   # 2% real + 3% inflation = 5% nominal
}

def calc_spouse_income(age: int, cfg: FamilyConfig, inflation: float, noise: float = 1.0) -> float:
    """Calculate spouse income for a given age based on family configuration.

    Work pattern:
    - Before marriage: $0
    - Marriage → 1yr before first kid: Full-time
    - 1yr before first kid → last kid turns 5: $0 (child-rearing gap)
    - Last kid turns 5 onward: Part-time permanently

    ``cfg.spouse_salary`` is the base salary expressed in today's (year-0) dollars.
    ``cfg.spouse_salary_growth`` is a *real* (above-inflation) annual growth rate; the
    loop accumulates only real purchasing-power growth.  The caller passes the cumulative
    inflation multiplier ``infl[i]`` as ``inflation`` so that the result is in nominal
    year-i dollars — inflation is applied exactly once.

    Soft-cap tapering mirrors the main earner (both base and cap are in real/today's dollars):
    - Below 60% of cap: full real growth
    - 60-100% of cap: tapered growth (linear decay to ~30% of normal)
    - Above cap: capped at ceiling
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

    # Calculate salary with soft cap (similar to main earner's TC cap)
    years_since_marriage = age - cfg.marriage_age
    base_salary = cfg.spouse_salary
    soft_cap = cfg.spouse_soft_cap

    # Apply growth year by year with tapering near cap
    salary = base_salary
    for _ in range(years_since_marriage):
        cap_ratio = salary / soft_cap
        if cap_ratio < 0.6:
            # Below 60% of cap: full growth
            growth = cfg.spouse_salary_growth
        elif cap_ratio < 1.0:
            # 60-100% of cap: tapered growth (linear decay to ~30% of normal growth)
            growth = cfg.spouse_salary_growth * (1 - cap_ratio) / 0.4 * 0.3
        else:
            # Above cap: no growth (capped)
            growth = 0.0
        salary = salary * (1 + growth)

    # Cap at soft cap
    salary = min(salary, soft_cap)

    return salary * fraction * inflation * noise


def simulate_career_growth(
    starting_tc: float,
    n_sims: int,
    n_years: int,
    rng: np.random.Generator,
    career_config: CareerConfig = None,
    current_age: int = None,
    fire_horizon: int = None,
) -> np.ndarray:
    """
    Simulate TC trajectories with stochastic promotions and soft cap.

    Each simulation gets:
    - Random promotion timing within windows
    - Random promotion raise amounts (12-20%)
    - Probability-based promotion occurrence
    - Soft cap that tapers growth as TC approaches ceiling

    Returns: (n_years, n_sims) array of TC values
    """
    if career_config is None:
        career_config = CareerConfig()
    if current_age is None:
        current_age = CURRENT_AGE

    cfg = career_config
    trajectory = cfg.trajectory
    soft_cap = cfg.soft_cap
    base_growth = BASE_GROWTH_RATES.get(trajectory, 0.015)
    promo_probs = PROMO_PROBABILITIES.get(trajectory, PROMO_PROBABILITIES["moderate"])

    # Initialize TC array
    tc = np.full(n_sims, starting_tc, dtype=float)
    incomes = np.zeros((n_years, n_sims))

    # Pre-generate promotion decisions for each simulation
    # For each promo window, determine: (1) if promoted, (2) when, (3) raise amount
    promo_windows = [cfg.promo_1_window, cfg.promo_2_window, cfg.promo_3_window, cfg.promo_4_window]

    # Track which promotions each sim has received
    promos_received = np.zeros((4, n_sims), dtype=bool)
    promo_ages = np.full((4, n_sims), 999, dtype=int)  # Age when promotion happens
    promo_raises = np.zeros((4, n_sims))  # Raise multiplier for each promo

    for promo_idx, (window_start, window_end) in enumerate(promo_windows):
        # Determine if each sim gets this promotion
        will_promote = rng.random(n_sims) < promo_probs[promo_idx]

        # For those who promote, pick a random age within the window
        window_size = window_end - window_start + 1
        promo_year_offset = rng.integers(0, window_size, size=n_sims)
        promo_ages[promo_idx] = np.where(will_promote, window_start + promo_year_offset, 999)

        # Random raise amount within range
        raise_min, raise_max = cfg.promo_raise_range
        promo_raises[promo_idx] = rng.uniform(raise_min, raise_max, size=n_sims)

    # Annual noise for base growth
    growth_noise = rng.normal(0, 0.015, size=(n_years, n_sims))

    # Simulate year by year
    for year_idx in range(n_years):
        age = current_age + year_idx

        # Store current TC
        incomes[year_idx] = tc.copy()

        # Stop growth after fire_horizon (they're retired)
        if fire_horizon is not None and age >= fire_horizon:
            continue

        # Check for promotions this year
        for promo_idx in range(4):
            promoted_this_year = (promo_ages[promo_idx] == age) & ~promos_received[promo_idx]
            tc = np.where(promoted_this_year, tc * promo_raises[promo_idx], tc)
            promos_received[promo_idx] = promos_received[promo_idx] | promoted_this_year

        # Apply growth with hard soft-cap
        # The cap acts as a ceiling that salaries gravitate toward but rarely exceed long-term
        cap_ratio = tc / soft_cap

        # Growth rate depends on distance from cap:
        # - Below 60% of cap: full growth (promotions + raises)
        # - 60-100% of cap: growth tapers linearly to zero
        # - Above cap: mean reversion back toward cap
        growth_mult = np.where(
            cap_ratio < 0.6,
            # Below 60%: full growth
            (1 + base_growth + growth_noise[year_idx]) * (1 + INFLATION),
            np.where(
                cap_ratio < 1.0,
                # 60-100%: linear taper, minimal growth at cap
                (1 + base_growth * (1 - cap_ratio) / 0.4 * 0.3 + growth_noise[year_idx] * 0.5) * (1 + INFLATION * 0.7),
                # Above cap: mean reversion (slight decline toward cap)
                1 + INFLATION * 0.3 - 0.02 * np.minimum(cap_ratio - 1.0, 0.5)
            )
        )

        # Clamp to prevent extreme values
        growth_mult = np.clip(growth_mult, 0.97, 1.08)

        tc = tc * growth_mult

    return incomes

def run_vectorized(starting_tc, city_name, n_sims, rng, seed_amounts=None, family_config=None,
                   career_config=None, ss_config=None, return_trajectories=False,
                   life_expectancy=None, current_age=None, fire_horizon=None,
                   hsa_annual_contrib=None, hsa_employer_contrib=0, rent_override=None,
                   lifestyle_creep_pct=0.025, use_401k=True, use_roth=True, use_hsa=True):
    """
    seed_amounts: SeedAmounts dataclass or dict with dollar amounts per account, e.g.
        SeedAmounts(taxable=165000, t401k=75000, roth=45000, hsa=15000)
        or {'taxable': 165000, '401k': 75000, 'roth': 45000, 'hsa': 15000}
    family_config: FamilyConfig dataclass for spouse/kid settings
    career_config: CareerConfig dataclass for salary progression settings
    return_trajectories: if True, return SimulationResults with full trajectory data
    life_expectancy: age to simulate through (default: LIFE_EXPECTANCY constant)
    current_age: starting age for simulation (default: CURRENT_AGE constant)
    hsa_annual_contrib: annual HSA contribution target in today's dollars (default: IRS max)
    hsa_employer_contrib: annual employer HSA contribution in today's dollars (free money, default: 0)
    rent_override: monthly rent in today's dollars, overrides all city rent tiers (default: use city config)
    """
    if life_expectancy is None:
        life_expectancy = LIFE_EXPECTANCY
    if current_age is None:
        current_age = CURRENT_AGE
    if fire_horizon is None:
        fire_horizon = FIRE_HORIZON
    n_years = life_expectancy - current_age + 1
    N = n_sims; cfg = CITIES[city_name]
    if rent_override is not None:
        cfg = CityConfig(
            one_br_rent=rent_override, nice_one_br_rent=rent_override, family_rent=rent_override,
            home_price=cfg.home_price, down_payment_pct=cfg.down_payment_pct,
            mortgage_rate=cfg.mortgage_rate, property_tax_rate=cfg.property_tax_rate,
            home_maintenance_pct=cfg.home_maintenance_pct, insurance_premium=cfg.insurance_premium,
            insurance_inflation=cfg.insurance_inflation, utility_premium=cfg.utility_premium,
            home_appreciation=cfg.home_appreciation, state_tax_rate=cfg.state_tax_rate,
            retirement_state_tax=cfg.retirement_state_tax, city_tax_rate=cfg.city_tax_rate,
        )

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
        traj_ss_income = np.zeros((n_years, N))

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

    if career_config is None:
        career_config = CareerConfig()

    if ss_config is None:
        ss_config = SocialSecurityConfig()

    # Pre-compute Social Security annual benefits by age (in today's dollars, nominal adjusted later)
    ss_primary_annual = ss_config.get_benefit_at_age(ss_config.claiming_age) if ss_config.enabled else 0
    ss_spouse_annual = ss_config.get_spouse_benefit_at_age(ss_config.spouse_claiming_age) if ss_config.enabled else 0

    # Derive kid ages from config
    kid_ages = sorted(family_config.kid_ages) if family_config.kid_ages else []
    kid1_born = kid_ages[0] if len(kid_ages) >= 1 else 999
    kid2_born = kid_ages[1] if len(kid_ages) >= 2 else 999
    kid1_college = kid1_born + 18
    kid2_college = kid2_born + 18
    contrib_per_kid = family_config.college_cost_per_kid * 0.06 / ((1.06)**18 - 1)

    spouse_noise = np.maximum(rng.normal(1.0, 0.1, size=(n_years, N)), 0.5)
    spouse_works_roll = rng.random(N) < 0.90  # 90% chance spouse works when in working period
    recession = rng.random(size=(n_years, N)) < RECESSION_PROB
    mr_arr = np.where(recession, rng.normal(MR_RECESSION_MEAN, MR_RECESSION_STD, size=(n_years, N)),
                      rng.normal(MR_NORMAL_MEAN, MR_NORMAL_STD, size=(n_years, N)))
    jl_occurs = rng.random(size=(n_years, N)) < JL_ANNUAL_PROB
    jl_duration = rng.integers(JL_MONTHS_MIN, JL_MONTHS_MAX, size=(n_years, N))
    jl_reentry_discount = rng.uniform(JL_REENTRY_MIN, JL_REENTRY_MAX, size=(n_years, N))
    hs_roll = rng.random(size=(n_years, N))

    # Stochastic inflation: two-component model.
    #
    # Component 1 — persistent AR(1) trend (the "sticky" part of inflation):
    #   core[t] = rho*core[t-1] + (1-rho)*mu + eps[t]
    #   Unconditional mean = mu = 3%.  Captures multi-year inflation cycles.
    #
    # Component 2 — transient stagflation shock (observation-layer only, does NOT feed
    #   back into the persistent state):
    #   inf_rates[t] = core[t] + recession[t] * stag[t]
    #   E[inf_rates] = 3% + P(rec)*E[stag] ≈ 3% + 0.15*4% = 3.6% — historically realistic.
    #
    # Separating the two components prevents the AR(1) from permanently amplifying recession
    # shocks (which would push the unconditional mean to ~7% otherwise).
    _mu_inf    = 0.030   # long-run core mean (3%)
    _rho_inf   = 0.85    # persistence of the trend component
    _sigma_inf = 0.015   # idiosyncratic annual shock volatility
    _eps_inf     = rng.normal(0, _sigma_inf, size=(n_years, N))
    _stagflation = rng.uniform(0.02, 0.06, size=(n_years, N))  # transient spike during recessions

    _inf_core = np.empty((n_years, N))
    _inf_core[0] = _mu_inf
    for _t in range(1, n_years):
        _inf_core[_t] = _rho_inf * _inf_core[_t - 1] + (1 - _rho_inf) * _mu_inf + _eps_inf[_t]

    # Observed inflation = persistent trend + transient stagflation overlay
    inf_rates = _inf_core + recession * _stagflation
    inf_rates = np.clip(inf_rates, -0.005, 0.15)   # allow mild deflation, cap at 15%
    # Cumulative multipliers shape (n_years, N): infl[t, sim] = product of (1+inf_rate) up to year t
    infl = np.cumprod(1 + inf_rates, axis=0)

    # Generate stochastic income trajectories using new career model
    incomes = simulate_career_growth(starting_tc, N, n_years, rng, career_config, current_age=current_age, fire_horizon=fire_horizon)

    # Add spouse income based on family config
    for i, age in enumerate(range(current_age, fire_horizon + 1)):
        spouse_inc = calc_spouse_income(age, family_config, infl[i], noise=1.0)
        incomes[i] += spouse_inc * spouse_noise[i] * spouse_works_roll

    months_unemployed = np.zeros(N, dtype=int)

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
    fire_swr = np.full(N, SWR_DEFAULT)  # Track SWR at FIRE time for each simulation
    has_home = cfg.home_price is not None
    home_purchase_age = np.zeros(N, dtype=int)  # Age when each sim buys; 0 = never bought

    # Track portfolio failures (ran out of money before life expectancy)
    failed = np.zeros(N, dtype=bool)
    failure_ages = np.full(N, 99, dtype=int)

    for i, age in enumerate(range(current_age, life_expectancy + 1)):
        ye = age - current_age; inf = infl[ye]; mr = mr_arr[i]
        just_bought = np.zeros(N, dtype=bool)
        # Income is zero after FIRE or after FIRE_HORIZON (forced retirement).
        # incomes has n_years rows so indexing with i is always in-bounds.
        year_inc = np.where(fired | (age > fire_horizon), 0.0, incomes[i])

        # Two-phase job loss: layoff gap then re-entry at slight discount
        months_unemployed = np.maximum(months_unemployed - 12, 0)
        new_layoffs = jl_occurs[i] & ~fired & (months_unemployed == 0)
        months_unemployed = np.where(new_layoffs, jl_duration[i], months_unemployed)
        fraction_working = np.clip(1 - months_unemployed / 12, 0, 1)
        year_inc = year_inc * fraction_working * np.where(months_unemployed > 0, jl_reentry_discount[i], 1.0)

        if age < 28:
            housing = cfg.one_br_rent*12*inf*np.ones(N); disc = DISC_YOUNG*inf*np.ones(N); st = 0.055
        elif age < 31:
            housing = cfg.nice_one_br_rent*12*inf*np.ones(N); disc = DISC_MID*inf*np.ones(N); st = 0.055
        else:
            st = cfg.state_tax_rate; disc = DISC_FAMILY*inf*np.ones(N)
            housing = cfg.family_rent*12*inf*np.ones(N)

        ca_prem = np.zeros(N); utility = cfg.utility_premium*inf if age >= 31 else 0
        if has_home and age >= 33:
            # Each year, simulations that haven't bought yet attempt to buy.
            # They succeed as soon as taxable savings cover the down payment.
            can_try = ~owns_home & ~fired
            pp = cfg.home_price * infl[ye]; down = pp * cfg.down_payment_pct
            just_bought = can_try & (taxable >= down)
            mortgage = np.where(just_bought, pp - down, mortgage)
            taxable = np.where(just_bought, taxable - down, taxable)
            owns_home = owns_home | just_bought
            home_val = np.where(just_bought, pp, home_val)
            home_purchase_age = np.where(just_bought, age, home_purchase_age)
            r_m = cfg.mortgage_rate / 12
            new_pmt = (pp - down) * r_m * (1 + r_m)**360 / ((1 + r_m)**360 - 1)
            fixed_pmt = np.where(just_bought, new_pmt, fixed_pmt)

            home_val = np.where(owns_home, home_val*(1+cfg.home_appreciation), home_val)
            ann_m = fixed_pmt*12; interest = mortgage*cfg.mortgage_rate
            mortgage = np.maximum(mortgage - np.maximum(np.minimum(ann_m-interest, mortgage), 0), 0)
            pm = home_val*(cfg.property_tax_rate+cfg.home_maintenance_pct)
            years_owned = np.where(owns_home, age - home_purchase_age, 0)
            ca_prem = np.where(owns_home, cfg.insurance_premium*((1+cfg.insurance_inflation)**years_owned), 0)
            housing = np.where(owns_home, np.where(mortgage > 0, ann_m+pm, pm), housing)
        housing += utility + ca_prem

        # Calculate kid costs based on each kid's age with realistic scaling
        kids = np.zeros(N)
        kid_cost = family_config.annual_cost_per_kid
        for kid_born in [kid1_born, kid2_born]:
            if kid_born >= 999:
                continue
            kid_age = age - kid_born
            if kid_age < 0 or kid_age >= 22:
                continue  # Not born yet or left home

            # Realistic cost scaling by kid's age (USDA data patterns):
            # - Ages 0-5: baseline (childcare heavy but smaller consumption)
            # - Ages 6-12: 1.2x (school activities, food, clothes)
            # - Ages 13-17: 1.5x (teens eat more, activities, driving)
            # - Ages 18-21: 1.3x (if still at home, less childcare but other costs)
            if kid_age >= 18:
                cost_mult = 1.3
            elif kid_age >= 13:
                cost_mult = 1.5
            elif kid_age >= 6:
                cost_mult = 1.2
            else:
                cost_mult = 1.0

            kids += kid_cost * inf * cost_mult

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

        if age >= 35: disc += DISC_STEP_35*inf
        if age >= 40: disc += DISC_STEP_40*inf
        creep = lifestyle_creep_pct * max(0, 1 - (age - 35) / 25) if age >= 35 else lifestyle_creep_pct
        disc += creep * year_inc

        ot = np.zeros(N)
        if age == 28: ot += OT_CAR_MOVE*inf
        if age == kid1_born: ot += OT_BABY_SETUP*inf
        if has_home: ot += np.where(just_bought, HOME_CLOSING_COSTS*inf, 0)
        if age == family_config.marriage_age: ot += family_config.wedding_budget * inf
        if age == 38: ot += OT_MID_UPGRADE*inf
        ot += np.where(new_layoffs, JL_SEARCH_COST*inf, 0)  # job search costs

        hc = np.where(fired, calc_retirement_hc(age, inf, N), np.where(age < 35, HC_YOUNG*inf, HC_OLDER*inf)*np.ones(N))
        hc += (hs_roll[i] < HEALTH_SHOCK_PROB).astype(float) * HEALTH_SHOCK_COST * inf

        total_spend = housing + disc + kids + c529c + ot + hc

        t401k_c = np.where(~fired & (year_inc > 0) & use_401k, np.minimum(FOUR01K_LIMIT*inf, year_inc*0.5), 0)
        hsa_limit = HSA_FAMILY_LIMIT if age >= family_config.marriage_age else HSA_INDIVIDUAL_LIMIT
        hsa_target = (hsa_annual_contrib if hsa_annual_contrib is not None else hsa_limit) if use_hsa else 0
        hsa_target = min(hsa_target, hsa_limit)
        hsa_c = np.where(~fired & (year_inc > 0), hsa_target*inf, 0.0)
        taxes = np.where(~fired, calc_taxes_vec(year_inc, st, t401k_c, hsa_c, inf, cfg.city_tax_rate), 0.0)

        net_inc = year_inc - taxes - total_spend
        wp = (~fired) & (net_inc > 0)
        a401 = np.where(wp, np.minimum(t401k_c, net_inc*0.5), 0) if use_401k else np.zeros(N)
        ar   = np.where(wp, np.minimum(ROTH_IRA_LIMIT*inf, net_inc*0.3), 0) if use_roth else np.zeros(N)
        ah   = np.where(wp, np.minimum(hsa_c, net_inc*0.2), 0) if use_hsa else np.zeros(N)
        # Calculate employer 401k match (free money, not from net_inc)
        matchable_cap = FOUR01K_LIMIT * inf * career_config.employer_match_limit
        matchable = np.minimum(matchable_cap, a401)
        employer_match = np.where(wp & use_401k, matchable * career_config.employer_match_pct, 0)
        # Employer HSA contribution (free money, credited during working years only)
        hsa_emp = np.where(~fired & (year_inc > 0) & use_hsa, hsa_employer_contrib * inf, 0.0)
        t401k += a401 + employer_match; roth += ar; roth_basis += ar; hsa_bal += ah + hsa_emp
        taxable += np.where(wp, net_inc-a401-ar-ah, 0)
        taxable += np.where((~fired)&(net_inc<=0), net_inc, 0)
        # Can't overdraft a brokerage — clamp working-year deficit draw at zero
        taxable = np.maximum(taxable, 0)

        total_port = taxable + t401k + roth + hsa_bal
        ret_base_spend = np.where(fired & (ret_base_spend==0), total_spend, ret_base_spend)
        wd_rate = np.where(fired & (total_port > 0), ret_base_spend/np.maximum(total_port, 1), 0)
        # Graduated spending adjustment based on withdrawal rate vs target SWR
        adj = np.where(
            wd_rate > fire_swr * 1.50, 0.85,     # 50%+ above target: aggressive 15% cut
            np.where(
                wd_rate > fire_swr * 1.25, 0.92,  # 25-50% above: moderate 8% cut
                np.where(
                    wd_rate < fire_swr * 0.50, 1.08,  # 50%+ below: generous 8% raise
                    np.where(
                        wd_rate < fire_swr * 0.75, 1.05,  # 25-50% below: modest 5% raise
                        1 + inf_rates[i]  # normal: inflation adjustment
                    )
                )
            )
        )
        ret_base_spend = np.where(fired, ret_base_spend*adj, ret_base_spend)
        # Floor: retirement spending can't drop below 60% of initial retirement spending baseline
        # (tracks a basic needs floor that scales with inflation)
        ret_spend_floor = np.where(fired & (ret_base_spend > 0), 30000 * inf, 0)
        ret_base_spend = np.where(fired, np.maximum(ret_base_spend, ret_spend_floor), ret_base_spend)

        # Social Security income (reduces portfolio withdrawal needs)
        ss_inc = np.zeros(N)
        if ss_config.enabled:
            if age >= ss_config.claiming_age:
                # SS COLA tracks CPI, so using simulated inflation (inf) is the correct
                # nominal adjustment. No separate cola_mult — that would double-count.
                ss_inc += ss_primary_annual * inf
            if age >= ss_config.spouse_claiming_age and age >= family_config.marriage_age:
                ss_inc += ss_spouse_annual * inf

        # ---- Retirement withdrawals ----
        # Tax rules by account:
        #   Taxable brokerage: only long-term capital gains taxed (~15%), not full amount
        #   Roth: always tax-free (contributions + growth after 59.5)
        #   HSA: tax-free for medical; non-medical before 65 = income tax + 20% penalty;
        #        non-medical after 65 = income tax only (like traditional IRA)
        #   401k: before 59.5 (age 60) = income tax + 10% penalty; after = income tax only
        #
        # Withdrawal priority:
        #   1. Taxable (minimal tax drag — only gains taxed at ~15%)
        #   2. HSA for medical (tax-free; assume ~40% of healthcare spending can come from HSA)
        #   3. Roth (tax-free)
        #   4. HSA remainder (taxed, possibly penalized)
        #   5. 401k (taxed, possibly penalized)

        active_retired = fired & ~failed
        draw = np.where(active_retired, np.maximum(ret_base_spend - ss_inc, 0), 0)

        # 1. Taxable brokerage — assume 50% cost basis, so only ~50% is gains taxed at 15%
        d_taxable = np.minimum(draw, np.maximum(taxable, 0))
        taxable_tax = d_taxable * 0.50 * 0.15
        taxable -= d_taxable
        rem = draw - d_taxable

        # 2. HSA for medical expenses (tax-free at any age)
        #    Approximate: retirees can cover ~40% of healthcare costs from HSA tax-free
        hsa_medical = np.where(active_retired, hc * 0.40, 0)
        d_hsa_medical = np.minimum(np.minimum(hsa_medical, rem), np.maximum(hsa_bal, 0))
        hsa_bal -= d_hsa_medical
        rem -= d_hsa_medical

        # 3. Roth — always tax-free
        d_roth = np.minimum(rem, np.maximum(roth, 0))
        roth -= d_roth
        roth_basis = np.minimum(roth_basis, roth)
        rem -= d_roth

        # 4. HSA non-medical remainder
        d_hsa_nonmed = np.minimum(rem * 0.5, np.maximum(hsa_bal, 0))
        hsa_penalty = np.where(age < 65, d_hsa_nonmed * 0.20, 0)
        hsa_income_tax = calc_taxes_vec(d_hsa_nonmed, cfg.retirement_state_tax, inf=inf)
        hsa_bal -= d_hsa_nonmed
        rem -= d_hsa_nonmed

        # 5. 401k — income tax always; 10% penalty before age 60 (proxy for 59.5)
        d_401k = np.minimum(rem, np.maximum(t401k, 0))
        pen_401k = np.where(age < 60, d_401k * 0.10, 0)
        tax_401k = calc_taxes_vec(d_401k, cfg.retirement_state_tax, inf=inf)
        t401k -= d_401k

        # Total tax drag from withdrawals (deducted from portfolio)
        total_wd_tax = taxable_tax + hsa_penalty + hsa_income_tax + pen_401k + tax_401k
        # Pay taxes from taxable first, then 401k
        tax_from_taxable = np.minimum(total_wd_tax, np.maximum(taxable, 0))
        taxable -= tax_from_taxable
        tax_from_401k = total_wd_tax - tax_from_taxable
        t401k -= np.where(active_retired, tax_from_401k, 0)

        # Clamp all liquid accounts at zero — overdrafts are not real money
        taxable = np.maximum(taxable, 0)
        t401k = np.maximum(t401k, 0)
        roth = np.maximum(roth, 0)
        roth_basis = np.minimum(roth_basis, roth)  # basis can't exceed balance
        hsa_bal = np.maximum(hsa_bal, 0)

        # Apply market returns (skip failed portfolios to avoid overflow)
        hr = (1+mr)**0.5-1; wm = ~fired
        # DCA: only the portion that went into taxable gets half-year return
        taxable_new_contrib = np.where(wp, net_inc - a401 - ar - ah, 0)
        taxable_new_contrib = np.maximum(taxable_new_contrib, 0)
        active = ~failed
        taxable = np.where(active & wm, (taxable - taxable_new_contrib)*(1+mr) + taxable_new_contrib*(1+hr),
                           np.where(active, taxable*(1+mr), taxable))
        t401k = np.where(active, t401k*(1+mr), t401k)
        roth = np.where(active, roth*(1+mr), roth)
        hsa_bal = np.where(active, hsa_bal*(1+mr*0.8), hsa_bal)

        # Post-return safety clamp
        taxable = np.maximum(taxable, 0)
        t401k = np.maximum(t401k, 0)
        roth = np.maximum(roth, 0)
        hsa_bal = np.maximum(hsa_bal, 0)
        roth_basis = np.minimum(roth_basis, roth)

        total_liq = taxable + t401k + roth + hsa_bal + c529_1 + c529_2
        accessible = taxable + roth_basis + hsa_bal

        # Record trajectory data
        if return_trajectories:
            traj_incomes[i] = year_inc
            # After FIRE, actual spending is ret_base_spend (dynamic), not formula-based total_spend
            actual_spend = np.where(fired & ~failed, ret_base_spend, np.where(~fired, total_spend, 0))
            traj_spending[i] = actual_spend
            traj_taxable[i] = taxable.copy()
            traj_t401k[i] = t401k.copy()
            traj_roth[i] = roth.copy()
            traj_hsa[i] = hsa_bal.copy()
            traj_home_equity[i] = np.where(owns_home, home_val - mortgage, 0)
            traj_net_worth[i] = total_liq + np.where(owns_home, home_val - mortgage, 0)
            traj_fired[i] = fired.copy()
            # Spending breakdown: in retirement, approximate category split
            is_retired_ok = fired & ~failed
            traj_spending_housing[i] = np.where(is_retired_ok, actual_spend * 0.35, np.where(~fired, housing, 0))
            traj_spending_disc[i] = np.where(is_retired_ok, actual_spend * 0.35, np.where(~fired, disc, 0))
            traj_spending_kids[i] = np.where(fired, 0, kids)
            traj_spending_education[i] = np.where(fired, 0, c529c)
            traj_spending_healthcare[i] = np.where(is_retired_ok, actual_spend * 0.30, np.where(~fired, hc, 0))
            traj_spending_one_time[i] = np.where(fired, 0, ot)
            # Cash flow breakdown
            traj_taxes[i] = taxes + np.where(active_retired, total_wd_tax, 0)
            traj_savings_401k[i] = a401
            traj_savings_roth[i] = ar
            traj_savings_hsa[i] = ah
            traj_savings_taxable[i] = np.where(wp, net_inc - a401 - ar - ah, 0)
            traj_ss_income[i] = ss_inc

        # Check for FIRE eligibility only during working years (up to FIRE_HORIZON)
        if 30 <= age <= fire_horizon:
            if has_home and age >= 33:
                years_owned_fire = np.where(owns_home, age - home_purchase_age, 0)
                rh_own = home_val*(cfg.property_tax_rate+cfg.home_maintenance_pct)
                rh_own += cfg.insurance_premium*((1+cfg.insurance_inflation)**years_owned_fire)
                rh_own += cfg.utility_premium*inf
                rh_rent = cfg.family_rent*12*inf + cfg.utility_premium*inf
                rh = np.where(owns_home, rh_own, rh_rent)
            else:
                rh = cfg.family_rent*12*inf + cfg.utility_premium*inf
            rd = 45000*inf; rhc = 24000*inf + HEALTH_SHOCK_PROB*HEALTH_SHOCK_COST*inf
            r529 = np.zeros(N)
            if kid1_born <= age < kid1_college: r529 += contrib_per_kid*inf
            if kid2_born <= age < kid2_college: r529 += contrib_per_kid*inf
            rt = rh + rd + rhc + r529 + kids
            # Dynamic SWR based on how long money needs to last
            swr = calc_swr(age, life_expectancy)

            # Prorate SS offset by the fraction of retirement it actually covers.
            # Someone FIREing at 35 with SS at 67 only gets SS for 23 of 55 years,
            # not the full retirement period.
            ss_future_annual = np.zeros(N)
            if ss_config.enabled:
                total_retirement_years = max(life_expectancy - age, 1)
                if ss_config.claiming_age <= life_expectancy:
                    years_with_ss = max(0, life_expectancy - max(ss_config.claiming_age, age))
                    ss_fraction = years_with_ss / total_retirement_years
                    ss_future_annual += ss_primary_annual * inf * ss_fraction
                if ss_config.spouse_claiming_age <= life_expectancy and family_config.marriage_age <= age:
                    years_with_spouse_ss = max(0, life_expectancy - max(ss_config.spouse_claiming_age, age))
                    spouse_ss_fraction = years_with_spouse_ss / total_retirement_years
                    ss_future_annual += ss_spouse_annual * inf * spouse_ss_fraction
                rt_net = np.maximum(rt - ss_future_annual, rt * 0.3)
            else:
                rt_net = rt

            fn = rt_net / swr
            bridge = rt_net * max(0, 60-age)
            can_fire = (~fired) & (total_liq >= fn) & (accessible >= bridge)
            fire_ages = np.where(can_fire, age, fire_ages)
            fire_swr = np.where(can_fire, swr, fire_swr)
            fired = fired | can_fire
            ret_base_spend = np.where(can_fire, rt, ret_base_spend)

        # Force retirement at FIRE_HORIZON for those who didn't FIRE voluntarily.
        # Without this, non-FIRE'd people have $0 income but never withdraw from
        # retirement accounts, making the simulation unrealistically optimistic.
        if age == fire_horizon:
            not_yet_fired = ~fired
            if not_yet_fired.any():
                swr_forced = calc_swr(age, life_expectancy)
                if has_home:
                    years_owned_forced = np.where(owns_home, age - home_purchase_age, 0)
                    rh_own = home_val*(cfg.property_tax_rate+cfg.home_maintenance_pct)
                    rh_own += cfg.insurance_premium*((1+cfg.insurance_inflation)**years_owned_forced)
                    rh_own += cfg.utility_premium*inf
                    rh_rent = cfg.family_rent*12*inf + cfg.utility_premium*inf
                    rh_forced = np.where(owns_home, rh_own, rh_rent)
                else:
                    rh_forced = cfg.family_rent*12*inf + cfg.utility_premium*inf
                rd_forced = 45000*inf
                rhc_forced = 24000*inf + HEALTH_SHOCK_PROB*HEALTH_SHOCK_COST*inf
                rt_forced = rh_forced + rd_forced + rhc_forced
                fire_ages = np.where(not_yet_fired, age, fire_ages)
                fire_swr = np.where(not_yet_fired, swr_forced, fire_swr)
                ret_base_spend = np.where(not_yet_fired, rt_forced, ret_base_spend)
                fired = fired | not_yet_fired

        # Track portfolio failures — use post-withdrawal, post-market liquid total
        # (total_port was computed before withdrawals and market returns; total_liq is current)
        just_failed = fired & ~failed & (total_liq <= 0)
        failure_ages = np.where(just_failed, age, failure_ages)
        failed = failed | just_failed

    if return_trajectories:
        return SimulationResults(
            fire_ages=fire_ages,
            ages=np.arange(current_age, life_expectancy + 1),
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
            ss_income=traj_ss_income,
        )
    return fire_ages, failed, failure_ages

def find_min_tc(city, target_age, conf_pct, seed_amounts=None, family_config=None,
                career_config=None, ss_config=None, lo=100000, hi=700000, tol=5000):
    while hi - lo > tol:
        mid = round((lo + hi) / 2 / 5000) * 5000
        rng = np.random.default_rng(42)
        fire_ages, _, _ = run_vectorized(mid, city, N_SIMS, rng, seed_amounts=seed_amounts,
                                          family_config=family_config, career_config=career_config,
                                          ss_config=ss_config)
        pct = (fire_ages <= target_age).mean() * 100
        if pct >= conf_pct:
            hi = mid
        else:
            lo = mid + 5000
    return hi
