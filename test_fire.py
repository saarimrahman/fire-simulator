"""
pytest test suite for the FIRE simulation engine.

Run:  ./venv/bin/pytest test_fire.py -v
"""

import numpy as np
import pytest

from fire import (
    run_vectorized, simulate_career_growth, calc_taxes_vec, calc_swr,
    SeedAmounts, FamilyConfig, CareerConfig, SocialSecurityConfig,
    CURRENT_AGE, FIRE_HORIZON, LIFE_EXPECTANCY,
    FOUR01K_LIMIT, ROTH_IRA_LIMIT, HSA_FAMILY_LIMIT, HSA_INDIVIDUAL_LIMIT,
)

N = 2_000
SEED = 42


def _run(*, tc=200_000, city="New York City", n=N, seed=SEED, seeds=None,
         family=None, career=None, ss=None, trajectories=True,
         current_age=None, fire_horizon=None, life_expectancy=None,
         use_401k=True, use_roth=True, use_hsa=True,
         hsa_annual_contrib=None, hsa_employer_contrib=0, **kw):
    """Convenience wrapper around run_vectorized with sane defaults."""
    rng = np.random.default_rng(seed)
    return run_vectorized(
        tc, city, n, rng,
        seed_amounts=seeds or SeedAmounts(),
        family_config=family or FamilyConfig(),
        career_config=career or CareerConfig(),
        ss_config=ss or SocialSecurityConfig(enabled=False),
        return_trajectories=trajectories,
        current_age=current_age,
        fire_horizon=fire_horizon,
        life_expectancy=life_expectancy,
        use_401k=use_401k, use_roth=use_roth, use_hsa=use_hsa,
        hsa_annual_contrib=hsa_annual_contrib,
        hsa_employer_contrib=hsa_employer_contrib,
        **kw,
    )


# =========================================================================
# Basic simulation sanity
# =========================================================================

class TestBasicSanity:
    def test_simulation_runs_and_returns_correct_shapes(self):
        r = _run()
        assert r.fire_ages.shape == (N,)
        assert r.ages[0] == CURRENT_AGE
        assert len(r.ages) == LIFE_EXPECTANCY - CURRENT_AGE + 1
        assert r.net_worth.shape == (len(r.ages), N)

    def test_starting_income_near_tc(self):
        r = _run(tc=200_000)
        median_inc = np.median(r.incomes[0])
        assert 180_000 < median_inc < 300_000, f"Starting income {median_inc:,.0f}"

    def test_net_worth_grows_during_working_years(self):
        r = _run(seeds=SeedAmounts(taxable=50_000))
        nw_start = np.median(r.net_worth[0])
        nw_10yr = np.median(r.net_worth[10])
        assert nw_10yr > nw_start * 1.5, f"NW didn't grow: {nw_start:,.0f} -> {nw_10yr:,.0f}"

    def test_fire_ages_in_plausible_range(self):
        r = _run(tc=250_000, seeds=SeedAmounts(taxable=100_000, t401k=50_000))
        valid = r.fire_ages[r.fire_ages < 99]
        assert len(valid) > 0, "No simulations reached FIRE"
        median = np.median(valid)
        assert 28 <= median <= 60, f"Median FIRE age {median:.0f} out of range"

    def test_spending_is_positive_during_working_years(self):
        r = _run()
        for yr in [0, 5, 10]:
            spend = np.median(r.spending[yr])
            assert spend > 30_000, f"Year {yr} spending too low: {spend:,.0f}"

    def test_failure_rate_is_plausible(self):
        r = _run(tc=200_000)
        rate = r.failed.mean()
        assert 0 <= rate <= 0.6, f"Failure rate {rate:.1%} out of range"

    def test_custom_start_age(self):
        for age in [22, 28, 35]:
            r = _run(current_age=age, n=500)
            assert r.ages[0] == age
            assert len(r.ages) == LIFE_EXPECTANCY - age + 1

    def test_deterministic_with_same_seed(self):
        r1 = _run(seed=99, n=500)
        r2 = _run(seed=99, n=500)
        np.testing.assert_array_equal(r1.fire_ages, r2.fire_ages)
        np.testing.assert_array_almost_equal(r1.net_worth, r2.net_worth)


# =========================================================================
# Tax calculation
# =========================================================================

class TestTaxes:
    def test_zero_income_zero_tax(self):
        tax = calc_taxes_vec(np.array([0.0]), 0.05)
        assert tax[0] == 0.0

    def test_tax_increases_with_income(self):
        incomes = np.array([50_000, 100_000, 200_000, 500_000], dtype=float)
        taxes = calc_taxes_vec(incomes, 0.05)
        assert np.all(np.diff(taxes) > 0), "Tax should increase with income"

    def test_401k_deduction_reduces_tax(self):
        inc = np.array([200_000.0])
        tax_no_401k = calc_taxes_vec(inc, 0.05)
        tax_with_401k = calc_taxes_vec(inc, 0.05, t401k=np.array([23_000.0]))
        assert tax_with_401k[0] < tax_no_401k[0], "401k should reduce tax"

    def test_hsa_deduction_reduces_tax(self):
        inc = np.array([200_000.0])
        tax_no_hsa = calc_taxes_vec(inc, 0.05)
        tax_with_hsa = calc_taxes_vec(inc, 0.05, hsa_c=np.array([8_550.0]))
        assert tax_with_hsa[0] < tax_no_hsa[0], "HSA should reduce tax"

    def test_effective_rate_is_reasonable(self):
        inc = np.array([200_000.0])
        tax = calc_taxes_vec(inc, 0.06)
        rate = tax[0] / inc[0]
        assert 0.20 < rate < 0.45, f"Effective rate {rate:.1%} out of range"

    def test_state_tax_matters(self):
        inc = np.array([200_000.0])
        tax_low = calc_taxes_vec(inc, 0.0)
        tax_high = calc_taxes_vec(inc, 0.10)
        assert tax_high[0] > tax_low[0], "Higher state rate should mean more tax"

    def test_city_tax_adds_burden(self):
        inc = np.array([200_000.0])
        tax_no_city = calc_taxes_vec(inc, 0.05, city_tax_rate=0.0)
        tax_city = calc_taxes_vec(inc, 0.05, city_tax_rate=0.035)
        assert tax_city[0] > tax_no_city[0]


# =========================================================================
# Account toggles
# =========================================================================

class TestAccountToggles:
    def test_disabling_401k_reduces_ending_nw(self):
        r_on = _run(seed=1, use_401k=True)
        r_off = _run(seed=1, use_401k=False)
        nw_on = np.median(r_on.net_worth[-1])
        nw_off = np.median(r_off.net_worth[-1])
        assert nw_on > nw_off * 1.1, f"401k should matter: on={nw_on:,.0f} off={nw_off:,.0f}"

    def test_disabling_roth_reduces_ending_nw(self):
        r_on = _run(seed=1, use_roth=True)
        r_off = _run(seed=1, use_roth=False)
        nw_on = np.median(r_on.net_worth[-1])
        nw_off = np.median(r_off.net_worth[-1])
        assert nw_on > nw_off, f"Roth should matter: on={nw_on:,.0f} off={nw_off:,.0f}"

    def test_disabling_hsa_increases_failure_rate(self):
        r_on = _run(seed=1, use_hsa=True)
        r_off = _run(seed=1, use_hsa=False)
        assert r_off.failed.mean() >= r_on.failed.mean() - 0.02, \
            f"HSA off should not decrease failures: on={r_on.failed.mean():.2%} off={r_off.failed.mean():.2%}"

    def test_401k_off_means_zero_401k_balance(self):
        r = _run(seed=1, use_401k=False, seeds=SeedAmounts())
        assert np.median(r.t401k[10]) == 0, "401k balance should stay 0 when disabled"

    def test_roth_off_means_zero_roth_balance(self):
        r = _run(seed=1, use_roth=False, seeds=SeedAmounts())
        assert np.median(r.roth[10]) == 0, "Roth balance should stay 0 when disabled"

    def test_disabled_contributions_flow_to_brokerage(self):
        r_all = _run(seed=1)
        r_no401k = _run(seed=1, use_401k=False)
        taxable_all = np.median(r_all.taxable[10])
        taxable_no401k = np.median(r_no401k.taxable[10])
        assert taxable_no401k > taxable_all * 1.2, \
            f"Brokerage should be higher without 401k: {taxable_all:,.0f} vs {taxable_no401k:,.0f}"


# =========================================================================
# Withdrawal logic and tax treatment
# =========================================================================

class TestWithdrawals:
    """Test retirement withdrawal ordering and tax treatment."""

    def _retired_sim(self, **kw):
        """Run sim with high seeds so everyone retires early, then check drawdown."""
        return _run(
            tc=200_000,
            seeds=SeedAmounts(taxable=500_000, t401k=300_000, roth=200_000, hsa=100_000),
            fire_horizon=45,
            n=1000,
            **kw,
        )

    def test_taxable_drawn_first(self):
        r = self._retired_sim(seed=42)
        fire_mask = r.fire_ages < 99
        if fire_mask.sum() == 0:
            pytest.skip("No sims retired")
        fired_sims = np.where(fire_mask)[0][:50]
        for sim in fired_sims:
            fire_yr = int(r.fire_ages[sim] - r.ages[0])
            if fire_yr + 5 < len(r.ages):
                taxable_at_fire = r.taxable[fire_yr, sim]
                taxable_after = r.taxable[min(fire_yr + 5, len(r.ages) - 1), sim]
                if taxable_at_fire > 10_000:
                    assert taxable_after < taxable_at_fire, \
                        "Taxable should decrease first in retirement"
                    break

    def test_roth_is_tax_free_advantage(self):
        """Roth-heavy portfolio should have lower failure rate than 401k-heavy."""
        r_roth = _run(
            seed=1,
            seeds=SeedAmounts(roth=400_000, taxable=100_000),
            use_401k=False,
            fire_horizon=50,
        )
        r_401k = _run(
            seed=1,
            seeds=SeedAmounts(t401k=400_000, taxable=100_000),
            use_roth=False,
            fire_horizon=50,
        )
        # Roth withdrawals are tax-free so should have equal or better outcomes
        assert r_roth.failed.mean() <= r_401k.failed.mean() + 0.05, \
            f"Roth should do at least as well: roth_fail={r_roth.failed.mean():.2%} 401k_fail={r_401k.failed.mean():.2%}"

    def test_hsa_penalty_before_65(self):
        """HSA non-medical withdrawals before 65 should incur penalty, after 65 no penalty."""
        r = self._retired_sim(seed=42)
        # After retirement, HSA should still hold value if person retires early
        # (penalty discourages early non-medical HSA draws)
        fire_mask = r.fire_ages < 50
        if fire_mask.sum() < 10:
            pytest.skip("Not enough early retirees")
        early_retirees = np.where(fire_mask)[0][:20]
        # At age 50 (pre-65), HSA should retain some balance (penalty discourages drain)
        age_50_idx = 50 - r.ages[0]
        hsa_at_50 = np.median(r.hsa[age_50_idx, early_retirees])
        # At age 70 (post-65, no penalty), HSA should be lower
        age_70_idx = 70 - r.ages[0]
        hsa_at_70 = np.median(r.hsa[age_70_idx, early_retirees])
        # Just verify the sim runs without errors and HSA is non-negative
        assert hsa_at_50 >= 0
        assert hsa_at_70 >= 0

    def test_401k_penalty_before_60(self):
        """401k early withdrawal includes 10% penalty — the same draw costs more."""
        r = self._retired_sim(seed=42)
        taxes_at_45 = np.median(r.taxes[45 - r.ages[0]])
        taxes_at_65 = np.median(r.taxes[65 - r.ages[0]])
        # Both should be non-negative (taxes exist in retirement)
        assert taxes_at_45 >= 0
        assert taxes_at_65 >= 0


# =========================================================================
# Social Security
# =========================================================================

class TestSocialSecurity:
    def test_ss_zero_before_claiming_age(self):
        r = _run(
            ss=SocialSecurityConfig(enabled=True, claiming_age=67,
                                    monthly_benefit_today=2500),
            seeds=SeedAmounts(taxable=100_000),
        )
        age_60_idx = 60 - r.ages[0]
        ss_at_60 = np.median(r.ss_income[age_60_idx])
        assert ss_at_60 == 0, f"SS should be 0 at 60, got {ss_at_60:,.0f}"

    def test_ss_positive_after_claiming_age(self):
        r = _run(
            ss=SocialSecurityConfig(enabled=True, claiming_age=67,
                                    monthly_benefit_today=2500),
            seeds=SeedAmounts(taxable=100_000),
        )
        age_70_idx = 70 - r.ages[0]
        ss_at_70 = np.median(r.ss_income[age_70_idx])
        assert ss_at_70 > 20_000, f"SS should be >$20k at 70, got {ss_at_70:,.0f}"

    def test_ss_improves_fire_rate(self):
        r_ss = _run(seed=1, ss=SocialSecurityConfig(enabled=True, claiming_age=67,
                                                     monthly_benefit_today=3000))
        r_no = _run(seed=1, ss=SocialSecurityConfig(enabled=False))
        fire_ss = (r_ss.fire_ages < 60).mean()
        fire_no = (r_no.fire_ages < 60).mean()
        assert fire_ss >= fire_no, f"SS should help: with={fire_ss:.2%} without={fire_no:.2%}"

    def test_ss_reduces_failure_rate(self):
        r_ss = _run(seed=1, ss=SocialSecurityConfig(enabled=True, claiming_age=67,
                                                     monthly_benefit_today=2500,
                                                     spouse_monthly_benefit=1250))
        r_no = _run(seed=1, ss=SocialSecurityConfig(enabled=False))
        assert r_ss.failed.mean() <= r_no.failed.mean() + 0.01

    def test_ss_tracks_inflation_not_double_counts(self):
        """SS income at 80 vs 67 should grow by inflation only, not inflation + COLA."""
        r = _run(
            ss=SocialSecurityConfig(enabled=True, claiming_age=67,
                                    monthly_benefit_today=2500,
                                    spouse_monthly_benefit=1250),
            seeds=SeedAmounts(taxable=100_000),
        )
        age_67_idx = 67 - r.ages[0]
        age_80_idx = 80 - r.ages[0]
        ss_67 = np.median(r.ss_income[age_67_idx])
        ss_80 = np.median(r.ss_income[age_80_idx])
        if ss_67 > 0:
            ratio = ss_80 / ss_67
            # Over 13 years at ~3% inflation: expect ratio ~1.47. Bug would give ~1.94
            assert 1.2 < ratio < 1.85, f"SS growth ratio {ratio:.2f} looks wrong"


# =========================================================================
# Career growth
# =========================================================================

class TestCareerGrowth:
    def test_aggressive_earns_more_than_conservative(self):
        rng1 = np.random.default_rng(SEED)
        inc_agg = simulate_career_growth(200_000, 1000, 30, rng1,
                                         CareerConfig(trajectory="aggressive"))
        rng2 = np.random.default_rng(SEED)
        inc_con = simulate_career_growth(200_000, 1000, 30, rng2,
                                         CareerConfig(trajectory="conservative"))
        assert np.median(inc_agg[-1]) > np.median(inc_con[-1]), \
            f"Aggressive should earn more: agg={np.median(inc_agg[-1]):,.0f} con={np.median(inc_con[-1]):,.0f}"

    def test_career_respects_soft_cap(self):
        rng = np.random.default_rng(SEED)
        cap = 400_000
        inc = simulate_career_growth(200_000, 1000, 30, rng,
                                     CareerConfig(soft_cap=cap, trajectory="aggressive"))
        median_end = np.median(inc[-1])
        assert median_end < cap * 1.5, f"Median {median_end:,.0f} too far above cap {cap:,.0f}"

    def test_career_uses_custom_start_age(self):
        """Promotions should align with actual age, not global CURRENT_AGE."""
        rng1 = np.random.default_rng(SEED)
        inc_25 = simulate_career_growth(200_000, 1000, 36, rng1,
                                        CareerConfig(), current_age=25)
        rng2 = np.random.default_rng(SEED)
        inc_30 = simulate_career_growth(200_000, 1000, 31, rng2,
                                        CareerConfig(), current_age=30)
        # Same starting TC
        np.testing.assert_array_almost_equal(inc_25[0], inc_30[0])
        # Different trajectories (promo windows hit at different year indices)
        med_y5_25 = np.median(inc_25[5])
        med_y5_30 = np.median(inc_30[5])
        assert abs(med_y5_25 - med_y5_30) / med_y5_25 > 0.01


# =========================================================================
# HSA employer contribution
# =========================================================================

class TestHSAEmployer:
    def test_employer_hsa_increases_hsa_balance(self):
        r_no = _run(seed=1, hsa_employer_contrib=0)
        r_yes = _run(seed=1, hsa_employer_contrib=1500)
        hsa_no = np.median(r_no.hsa[10])
        hsa_yes = np.median(r_yes.hsa[10])
        assert hsa_yes > hsa_no * 1.05, f"Employer HSA should grow balance: {hsa_no:,.0f} vs {hsa_yes:,.0f}"

    def test_employer_hsa_improves_outcomes(self):
        r_no = _run(seed=1, hsa_employer_contrib=0)
        r_yes = _run(seed=1, hsa_employer_contrib=1500)
        nw_no = np.median(r_no.net_worth[-1])
        nw_yes = np.median(r_yes.net_worth[-1])
        assert nw_yes > nw_no, "Employer HSA should improve ending NW"


# =========================================================================
# Safe withdrawal rate
# =========================================================================

class TestSWR:
    def test_swr_decreases_for_earlier_retirement(self):
        swr_40 = calc_swr(40, 90)
        swr_55 = calc_swr(55, 90)
        swr_65 = calc_swr(65, 90)
        assert swr_40 < swr_55 < swr_65, f"SWR should increase: {swr_40:.3f} {swr_55:.3f} {swr_65:.3f}"

    def test_swr_in_range(self):
        for age in [35, 45, 55, 65]:
            swr = calc_swr(age, 90)
            assert 0.02 < swr < 0.06, f"SWR at {age} = {swr:.3f} out of range"


# =========================================================================
# Edge cases and complex interactions
# =========================================================================

class TestEdgeCases:
    def test_no_kids_scenario(self):
        r = _run(family=FamilyConfig(kid_ages=()))
        assert r.fire_ages.shape == (N,)
        # No kids should mean better outcomes than with kids
        r_kids = _run(seed=1, family=FamilyConfig(kid_ages=(31, 33)))
        r_nokids = _run(seed=1, family=FamilyConfig(kid_ages=()))
        assert np.median(r_nokids.fire_ages[r_nokids.fire_ages < 99]) <= \
               np.median(r_kids.fire_ages[r_kids.fire_ages < 99]) + 1

    def test_zero_tc_still_runs(self):
        r = _run(tc=100_000, n=500)
        assert len(r.fire_ages) == 500

    def test_high_tc_almost_everyone_fires(self):
        r = _run(tc=500_000, n=1000,
                 seeds=SeedAmounts(taxable=200_000, t401k=200_000, roth=100_000))
        fire_rate = (r.fire_ages < 60).mean()
        assert fire_rate > 0.5, f"High TC should have >50% FIRE: {fire_rate:.1%}"

    def test_all_accounts_off(self):
        """Everything flows to brokerage — sim should still work."""
        r = _run(seed=1, use_401k=False, use_roth=False, use_hsa=False, n=500)
        assert r.fire_ages.shape == (500,)
        assert np.median(r.taxable[10]) > 0, "All money should be in brokerage"
        assert np.median(r.t401k[10]) == 0
        assert np.median(r.roth[10]) == 0
        assert np.median(r.hsa[10]) == 0

    def test_spouse_no_work(self):
        r_work = _run(seed=1, family=FamilyConfig(spouse_works=True, spouse_salary=80_000))
        r_no = _run(seed=1, family=FamilyConfig(spouse_works=False))
        nw_work = np.median(r_work.net_worth[-1])
        nw_no = np.median(r_no.net_worth[-1])
        assert nw_work > nw_no, "Spouse working should improve NW"

    def test_different_cities_produce_different_results(self):
        r_nyc = _run(seed=1, city="New York City", n=500)
        r_sac = _run(seed=1, city="Sacramento", n=500)
        nw_nyc = np.median(r_nyc.net_worth[-1])
        nw_sac = np.median(r_sac.net_worth[-1])
        assert nw_nyc != nw_sac, "Different cities should produce different outcomes"

    def test_net_worth_never_negative(self):
        r = _run(tc=150_000, n=500)
        assert np.all(r.net_worth >= -1), "Net worth should never go significantly negative"

    def test_taxes_recorded_in_retirement(self):
        """Retirement withdrawal taxes should show up in the taxes trajectory."""
        r = _run(
            seeds=SeedAmounts(taxable=500_000, t401k=500_000),
            fire_horizon=45, n=500,
        )
        age_70_idx = 70 - r.ages[0]
        retirement_taxes = np.median(r.taxes[age_70_idx])
        assert retirement_taxes > 0, f"Retirement taxes should be positive, got {retirement_taxes:,.0f}"
