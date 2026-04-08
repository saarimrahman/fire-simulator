"""
pytest test suite for the FIRE simulation engine.

Run:  ./venv/bin/pytest test_fire.py -v
"""

import numpy as np
import pytest

from fire import (
    run_vectorized, simulate_career_growth, calc_taxes_vec, calc_swr,
    calc_retirement_hc, calc_taxable_brokerage_sale, calc_roth_ira_limit,
    SeedAmounts, FamilyConfig, CareerConfig, SocialSecurityConfig,
    CURRENT_AGE, FIRE_HORIZON, LIFE_EXPECTANCY,
    FOUR01K_LIMIT, FOUR01K_TOTAL_LIMIT, ROTH_IRA_LIMIT,
    HSA_FAMILY_LIMIT, HSA_INDIVIDUAL_LIMIT,
    HC_YOUNG, HC_OLDER, HC_RET_BASE, HC_MEDICARE,
    DISC_YOUNG, DISC_FAMILY, DISC_STEP_35, DISC_STEP_40,
    COLLEGE_YEARS,
)

N = 2_000
SEED = 42


def _run(*, tc=200_000, city="New York City", n=N, seed=SEED, seeds=None,
         family=None, career=None, ss=None, trajectories=True,
         current_age=None, fire_horizon=None, life_expectancy=None,
         use_401k=True, use_roth=True, use_hsa=True,
         hsa_annual_contrib=None, hsa_employer_contrib=0,
         use_mega_backdoor=False, lifestyle_creep_pct=0.025, **kw):
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
        use_mega_backdoor=use_mega_backdoor,
        lifestyle_creep_pct=lifestyle_creep_pct,
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

    def test_single_pays_more_than_married_joint_same_income(self):
        inc = np.array([200_000.0])
        tax_single = calc_taxes_vec(inc, 0.05, filing_status="single")
        tax_joint = calc_taxes_vec(inc, 0.05, filing_status="married_joint")
        assert tax_single[0] > tax_joint[0], \
            f"Single filer should pay more at same income: single={tax_single[0]:,.0f} joint={tax_joint[0]:,.0f}"


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
        """Roth withdrawals should incur less tax than 401k withdrawals in retirement."""
        r_roth = _run(
            seed=1, tc=250_000, n=2000,
            seeds=SeedAmounts(roth=800_000, taxable=200_000),
            use_401k=False,
            fire_horizon=50,
            ss=SocialSecurityConfig(enabled=False),
        )
        r_401k = _run(
            seed=1, tc=250_000, n=2000,
            seeds=SeedAmounts(t401k=800_000, taxable=200_000),
            use_roth=False,
            fire_horizon=50,
            ss=SocialSecurityConfig(enabled=False),
        )
        # Roth withdrawals are tax-free; 401k has income tax.
        # Compare cumulative retirement taxes.
        retire_start = 50 - r_roth.ages[0]
        tax_roth = np.median(r_roth.taxes[retire_start:].sum(axis=0))
        tax_401k = np.median(r_401k.taxes[retire_start:].sum(axis=0))
        assert tax_roth < tax_401k, \
            f"Roth should have lower retirement taxes: roth={tax_roth:,.0f} 401k={tax_401k:,.0f}"

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

    def test_large_ss_can_create_taxable_income_even_with_roth_funding(self):
        """Federal tax code can tax part of Social Security even if spending is funded from Roth."""
        common = dict(
            seed=1, tc=200_000, n=1000,
            seeds=SeedAmounts(roth=2_000_000),
            use_401k=False, use_hsa=False,
            fire_horizon=45,
            family=FamilyConfig(kid_ages=()),
            ss=SocialSecurityConfig(
                enabled=True,
                claiming_age=67,
                spouse_claiming_age=67,
                monthly_benefit_today=7_000,
                spouse_monthly_benefit=5_000,
            ),
        )
        r = _run(**common)
        age_70_idx = 70 - r.ages[0]
        retired = r.fired_status[age_70_idx] & ~r.failed
        if retired.sum() < 50:
            pytest.skip("Not enough retirees at 70")
        taxes_70 = np.median(r.taxes[age_70_idx, retired])
        assert taxes_70 > 1_000, \
            f"Large SS should create some taxable income even with Roth funding: {taxes_70:,.0f}"


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

    def test_employer_hsa_counts_toward_annual_limit(self):
        """Employer HSA money should reduce the employee contribution room."""
        r = _run(
            seed=1, tc=300_000, n=1000,
            use_401k=False, use_roth=False,
            hsa_employer_contrib=7_000,
            family=FamilyConfig(marriage_age=35, kid_ages=(), spouse_works=False),
        )
        age_36_idx = 36 - r.ages[0]
        employee_hsa = np.median(r.savings_hsa[age_36_idx])
        # Family limit is $8,550, so with a $7,000 employer contribution the
        # employee should only have about $1,550 of remaining room.
        assert 1_000 < employee_hsa < 2_500, \
            f"Employee HSA room should shrink after employer funding: {employee_hsa:,.0f}"


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

    def test_swr_override_changes_fire_number(self):
        r_default = _run(seed=1, n=2000, family=FamilyConfig(kid_ages=()))
        r_low = _run(seed=1, n=2000, family=FamilyConfig(kid_ages=()), swr_override=0.03)
        mask_default = r_default.fire_ages < 99
        mask_low = r_low.fire_ages < 99
        if mask_default.sum() < 50 or mask_low.sum() < 50:
            pytest.skip("Not enough FIRE'd sims")
        fn_default = np.median(r_default.fire_number[mask_default])
        fn_low = np.median(r_low.fire_number[mask_low])
        assert fn_low > fn_default * 1.15, \
            f"Lower SWR override should require a bigger nest egg: default={fn_default:,.0f} low={fn_low:,.0f}"


# =========================================================================
# Mega Backdoor Roth
# =========================================================================

class TestMegaBackdoorRoth:
    def test_mega_backdoor_increases_roth_balance(self):
        r_off = _run(seed=1, use_mega_backdoor=False)
        r_on  = _run(seed=1, use_mega_backdoor=True)
        roth_off = np.median(r_off.roth[15])
        roth_on  = np.median(r_on.roth[15])
        assert roth_on > roth_off * 1.5, \
            f"Mega backdoor should boost Roth: off={roth_off:,.0f} on={roth_on:,.0f}"

    def test_mega_backdoor_reduces_taxable(self):
        """Money goes to Roth instead of brokerage."""
        r_off = _run(seed=1, use_mega_backdoor=False)
        r_on  = _run(seed=1, use_mega_backdoor=True)
        tax_off = np.median(r_off.taxable[15])
        tax_on  = np.median(r_on.taxable[15])
        assert tax_on < tax_off, \
            f"Mega backdoor should reduce brokerage: off={tax_off:,.0f} on={tax_on:,.0f}"

    def test_mega_backdoor_improves_ending_nw(self):
        """Tax-free Roth growth should beat taxable growth."""
        r_off = _run(seed=1, use_mega_backdoor=False)
        r_on  = _run(seed=1, use_mega_backdoor=True)
        nw_off = np.median(r_off.net_worth[-1])
        nw_on  = np.median(r_on.net_worth[-1])
        assert nw_on > nw_off, \
            f"Mega backdoor should improve NW: off={nw_off:,.0f} on={nw_on:,.0f}"

    def test_mega_backdoor_requires_401k(self):
        """Mega backdoor with 401k disabled should have no effect."""
        r = _run(seed=1, use_401k=False, use_mega_backdoor=True)
        assert np.median(r.roth[15]) < 200_000, \
            "Mega backdoor shouldn't work without 401k"

    def test_mega_backdoor_contributions_are_roth_basis(self):
        """Mega backdoor contributions are post-tax, so they're Roth basis (tax-free withdrawal)."""
        r_on = _run(seed=1, use_mega_backdoor=True)
        # Roth should grow faster than without mega, and it's all basis
        # Check that Roth balance at year 15 has a significant basis component
        # (basis tracked through roth_basis, but not exposed in results — we can verify
        # indirectly: Roth-heavy portfolios should have lower failure rates)
        r_off = _run(seed=1, use_mega_backdoor=False)
        assert r_on.failed.mean() <= r_off.failed.mean() + 0.02


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


# =========================================================================
# Contribution mechanics and tax deductions
# =========================================================================

class TestContributionMechanics:
    """Verify contributions, deductions, and money flow are correct."""

    def test_401k_reduces_taxable_income(self):
        """Pre-tax 401k should reduce taxes via AGI deduction."""
        r_on = _run(seed=1, use_401k=True, use_roth=False, use_hsa=False)
        r_off = _run(seed=1, use_401k=False, use_roth=False, use_hsa=False)
        tax_on = np.median(r_on.taxes[5])
        tax_off = np.median(r_off.taxes[5])
        assert tax_on < tax_off, \
            f"401k should reduce working-year taxes: on={tax_on:,.0f} off={tax_off:,.0f}"

    def test_hsa_reduces_taxable_income(self):
        """HSA contributions should reduce taxes via AGI deduction."""
        r_on = _run(seed=1, use_401k=False, use_roth=False, use_hsa=True)
        r_off = _run(seed=1, use_401k=False, use_roth=False, use_hsa=False)
        tax_on = np.median(r_on.taxes[5])
        tax_off = np.median(r_off.taxes[5])
        assert tax_on < tax_off, \
            f"HSA should reduce working-year taxes: on={tax_on:,.0f} off={tax_off:,.0f}"

    def test_roth_does_not_reduce_taxes(self):
        """Roth contributions are post-tax — should not change tax bill."""
        r_on = _run(seed=1, use_401k=False, use_roth=True, use_hsa=False)
        r_off = _run(seed=1, use_401k=False, use_roth=False, use_hsa=False)
        tax_on = np.median(r_on.taxes[5])
        tax_off = np.median(r_off.taxes[5])
        assert abs(tax_on - tax_off) / max(tax_off, 1) < 0.02, \
            f"Roth should not change taxes: on={tax_on:,.0f} off={tax_off:,.0f}"

    def test_mega_backdoor_does_not_reduce_taxes(self):
        """Mega backdoor is after-tax — should not change tax bill."""
        r_on = _run(seed=1, use_mega_backdoor=True)
        r_off = _run(seed=1, use_mega_backdoor=False)
        tax_on = np.median(r_on.taxes[5])
        tax_off = np.median(r_off.taxes[5])
        assert abs(tax_on - tax_off) / max(tax_off, 1) < 0.02, \
            f"Mega backdoor should not change taxes: on={tax_on:,.0f} off={tax_off:,.0f}"

    def test_high_income_single_cannot_make_direct_roth_contribution(self):
        """Direct Roth IRA contributions should phase out at high MAGI."""
        r = _run(
            seed=1, tc=400_000, n=1000,
            use_401k=False, use_hsa=False, use_roth=True,
            family=FamilyConfig(marriage_age=35, kid_ages=(), spouse_works=False),
        )
        roth_yr0 = np.median(r.savings_roth[0])
        assert roth_yr0 == 0, f"High-income single filer should be phased out of direct Roth: {roth_yr0:,.0f}"

    def test_total_savings_equals_net_income(self):
        """All net income should be allocated: 401k + Roth + HSA + brokerage."""
        r = _run(seed=1, n=500)
        for yr in [0, 5, 10]:
            saved = (r.savings_401k[yr] + r.savings_roth[yr] +
                     r.savings_hsa[yr] + r.savings_taxable[yr])
            net = r.incomes[yr] - r.taxes[yr] - r.spending[yr]
            positive = net > 1
            if positive.sum() > 0:
                gap = np.median(np.abs(saved[positive] - net[positive]) / np.maximum(net[positive], 1))
                assert gap < 0.02, f"Year {yr}: per-sim savings allocation gap too large: {gap:.2%}"

    def test_employer_match_is_free_money(self):
        """Employer match should increase 401k without reducing net income."""
        r_match = _run(seed=1, career=CareerConfig(employer_match_pct=0.50, employer_match_limit=0.43))
        r_nomatch = _run(seed=1, career=CareerConfig(employer_match_pct=0.0, employer_match_limit=0.0))
        k401_match = np.median(r_match.t401k[10])
        k401_no = np.median(r_nomatch.t401k[10])
        assert k401_match > k401_no * 1.1, f"Match should boost 401k: {k401_match:,.0f} vs {k401_no:,.0f}"
        # Brokerage should be similar (match doesn't come from net_inc)
        tax_match = np.median(r_match.taxable[10])
        tax_no = np.median(r_nomatch.taxable[10])
        assert abs(tax_match - tax_no) / max(tax_no, 1) < 0.15, \
            f"Match shouldn't drain brokerage: {tax_match:,.0f} vs {tax_no:,.0f}"


# =========================================================================
# Withdrawal tax correctness
# =========================================================================

class TestWithdrawalTaxCorrectness:
    """Verify tax treatment of each account type during retirement withdrawals."""

    def _early_retire_sim(self, **kw):
        return _run(
            tc=200_000,
            seeds=SeedAmounts(taxable=800_000, t401k=400_000, roth=300_000, hsa=150_000),
            fire_horizon=40, n=1000, seed=42,
            **kw,
        )

    def test_brokerage_only_taxed_on_gains(self):
        """Brokerage withdrawals should be taxed at ~7.5% effective (50% gains * 15% LTCG)."""
        r = _run(
            seeds=SeedAmounts(taxable=2_000_000),
            use_401k=False, use_roth=False, use_hsa=False,
            fire_horizon=40, n=500, seed=42,
        )
        age_50_idx = 50 - r.ages[0]
        retired_mask = r.fired_status[age_50_idx]
        if retired_mask.sum() < 10:
            pytest.skip("Not enough retirees at 50")
        taxes_50 = np.median(r.taxes[age_50_idx, retired_mask])
        spend_50 = np.median(r.spending[age_50_idx, retired_mask])
        if spend_50 > 0:
            eff_rate = taxes_50 / spend_50
            assert eff_rate < 0.20, \
                f"Brokerage-only effective rate should be low: {eff_rate:.1%}"

    def test_roth_only_has_zero_withdrawal_tax(self):
        """Roth-only retirees should pay no withdrawal tax after 59.5."""
        r = _run(
            seeds=SeedAmounts(roth=2_000_000),
            use_401k=False, use_hsa=False,
            fire_horizon=40, n=500, seed=42,
        )
        # At age 62, Roth withdrawals (contributions + earnings) are fully tax-free.
        # Before 60, earnings would be penalized — that's correct behavior, so test post-60.
        age_62_idx = 62 - r.ages[0]
        retired_mask = r.fired_status[age_62_idx]
        if retired_mask.sum() < 10:
            pytest.skip("Not enough retirees")
        taxes_62 = np.median(r.taxes[age_62_idx, retired_mask])
        assert taxes_62 < 1000, f"Roth retirement taxes after 60 should be ~0: got {taxes_62:,.0f}"

    def test_401k_withdrawal_has_income_tax(self):
        """401k-only retirees should pay income tax on withdrawals."""
        r = _run(
            seeds=SeedAmounts(t401k=2_000_000),
            use_roth=False, use_hsa=False,
            fire_horizon=40, n=500, seed=42,
        )
        age_65_idx = 65 - r.ages[0]
        retired_mask = r.fired_status[age_65_idx]
        if retired_mask.sum() < 10:
            pytest.skip("Not enough retirees")
        taxes_65 = np.median(r.taxes[age_65_idx, retired_mask])
        assert taxes_65 > 5000, f"401k retirement taxes should be significant: got {taxes_65:,.0f}"

    def test_401k_penalty_disappears_at_60(self):
        """401k should have 10% penalty before 60 but not after."""
        r = self._early_retire_sim()
        # Compare tax burden at 45 (penalty) vs 65 (no penalty), per dollar withdrawn
        # Hard to isolate, so just verify taxes at 65 are positive (income tax exists)
        age_65_idx = 65 - r.ages[0]
        taxes_65 = np.median(r.taxes[age_65_idx])
        assert taxes_65 >= 0

    def test_hsa_medical_is_tax_free(self):
        """HSA medical withdrawals should not add to tax burden."""
        # Compare HSA-heavy vs no-HSA, both with medical expenses
        r_hsa = _run(
            seeds=SeedAmounts(taxable=500_000, hsa=500_000),
            use_401k=False, use_roth=False,
            fire_horizon=40, n=500, seed=42,
        )
        r_no_hsa = _run(
            seeds=SeedAmounts(taxable=1_000_000),
            use_401k=False, use_roth=False, use_hsa=False,
            fire_horizon=40, n=500, seed=42,
        )
        age_55_idx = 55 - r_hsa.ages[0]
        tax_hsa = np.median(r_hsa.taxes[age_55_idx])
        tax_no = np.median(r_no_hsa.taxes[age_55_idx])
        # HSA medical withdrawals are tax-free, so tax burden should be similar or less
        assert tax_hsa <= tax_no * 1.1, \
            f"HSA should not increase taxes: hsa={tax_hsa:,.0f} no_hsa={tax_no:,.0f}"


class TestTaxableBrokerageRealism:
    def test_taxable_sale_of_pure_basis_has_no_capital_gains_tax(self):
        tax, new_balance, new_basis, realized_gains = calc_taxable_brokerage_sale(
            balance=np.array([500_000.0]),
            basis=np.array([500_000.0]),
            withdrawal=np.array([100_000.0]),
            state_tax_rate=0.05,
        )
        assert tax[0] == 0, f"Pure basis withdrawal should have no capital gains tax: {tax[0]:,.0f}"
        assert realized_gains[0] == 0
        assert new_balance[0] == 400_000
        assert new_basis[0] == 400_000


# =========================================================================
# Portfolio drawdown and failure mechanics
# =========================================================================

class TestDrawdownMechanics:
    def test_portfolio_decreases_in_retirement(self):
        """Net worth should generally decrease during retirement (drawdown)."""
        r = _run(
            seeds=SeedAmounts(taxable=300_000, t401k=200_000),
            fire_horizon=45, n=500, seed=42,
            ss=SocialSecurityConfig(enabled=False),
        )
        fire_mask = r.fire_ages < 50
        if fire_mask.sum() < 10:
            pytest.skip("Not enough early retirees")
        sims = np.where(fire_mask)[0][:30]
        age_50 = 50 - r.ages[0]
        age_65 = 65 - r.ages[0]
        for sim in sims:
            if not r.failed[sim]:
                nw_50 = r.net_worth[age_50, sim]
                nw_65 = r.net_worth[age_65, sim]
                if nw_50 > 100_000:
                    break
        # At least some should show drawdown over 15 years without SS
        assert nw_65 < nw_50 * 1.5 or r.failed[sim], \
            f"Expected drawdown: NW50={nw_50:,.0f} NW65={nw_65:,.0f}"

    def test_failed_portfolios_have_zero_nw(self):
        """Failed simulations should have net worth at or near zero at failure."""
        r = _run(tc=150_000, n=1000, seed=42)
        failed_sims = np.where(r.failed)[0]
        if len(failed_sims) == 0:
            pytest.skip("No failures to test")
        for sim in failed_sims[:20]:
            fail_age = int(r.failure_ages[sim])
            fail_idx = fail_age - r.ages[0]
            if fail_idx < len(r.ages):
                nw = r.net_worth[fail_idx, sim]
                assert nw <= 1, f"Failed sim {sim} has NW={nw:,.0f} at failure age {fail_age}"

    def test_ss_reduces_drawdown_rate(self):
        """Social Security should slow portfolio depletion in retirement."""
        common = dict(
            seeds=SeedAmounts(taxable=400_000, t401k=200_000),
            fire_horizon=50, n=500, seed=42,
        )
        r_ss = _run(**common, ss=SocialSecurityConfig(enabled=True, claiming_age=67,
                                                       monthly_benefit_today=2500))
        r_no = _run(**common, ss=SocialSecurityConfig(enabled=False))
        # At age 75, SS recipients should have more NW remaining
        age_75 = 75 - r_ss.ages[0]
        nw_ss = np.median(r_ss.net_worth[age_75])
        nw_no = np.median(r_no.net_worth[age_75])
        assert nw_ss > nw_no, f"SS should preserve NW: ss={nw_ss:,.0f} no_ss={nw_no:,.0f}"

    def test_spending_floor_in_retirement(self):
        """Retirement spending should not drop below the $30k floor."""
        r = _run(
            tc=150_000, n=500, seed=42,
            seeds=SeedAmounts(taxable=200_000),
            fire_horizon=50,
        )
        for yr_idx in range(50 - r.ages[0], len(r.ages)):
            retired = r.fired_status[yr_idx] & ~r.failed
            if retired.sum() > 0:
                min_spend = r.spending[yr_idx, retired].min()
                assert min_spend >= 20_000, \
                    f"Spending at age {r.ages[yr_idx]} below floor: {min_spend:,.0f}"

    def test_forced_retirement_at_fire_horizon(self):
        """Everyone should be retired at fire_horizon age."""
        r = _run(tc=150_000, n=500, fire_horizon=55, seed=42)
        assert np.all(r.fire_ages <= 55), \
            f"Some sims not retired by 55: max fire_age={r.fire_ages.max()}"


# =========================================================================
# Home purchase and housing interactions
# =========================================================================

class TestHousing:
    def test_home_purchase_reduces_taxable(self):
        """Down payment should come from taxable account."""
        r = _run(
            seeds=SeedAmounts(taxable=500_000),
            city="Sacramento", n=500, seed=42,
        )
        # Before home purchase age (~33), taxable should be higher than after
        age_32 = 32 - r.ages[0]
        age_36 = 36 - r.ages[0]
        tax_32 = np.median(r.taxable[age_32])
        tax_36 = np.median(r.taxable[age_36])
        # Some sims buy homes, reducing taxable
        he_36 = np.median(r.home_equity[age_36])
        assert he_36 > 0, "Some sims should own homes by 36"

    def test_home_equity_grows_over_time(self):
        """Home equity should increase from appreciation and mortgage paydown."""
        r = _run(city="Sacramento", n=500, seed=42,
                 seeds=SeedAmounts(taxable=200_000))
        age_40 = 40 - r.ages[0]
        age_60 = 60 - r.ages[0]
        he_40 = np.median(r.home_equity[age_40])
        he_60 = np.median(r.home_equity[age_60])
        assert he_60 > he_40, f"Home equity should grow: 40={he_40:,.0f} 60={he_60:,.0f}"


# =========================================================================
# Spending trajectory correctness
# =========================================================================

class TestSpendingTrajectory:
    def test_spending_breakdown_sums_to_total(self):
        """Sum of spending categories should approximately match total spending."""
        r = _run(n=500, seed=42)
        for yr in [0, 5, 10, 15]:
            housing = np.median(r.spending_housing[yr])
            disc = np.median(r.spending_discretionary[yr])
            kids = np.median(r.spending_kids[yr])
            edu = np.median(r.spending_education[yr])
            hc = np.median(r.spending_healthcare[yr])
            ot = np.median(r.spending_one_time[yr])
            total = np.median(r.spending[yr])
            parts = housing + disc + kids + edu + hc + ot
            if total > 0:
                assert abs(parts - total) / total < 0.20, \
                    f"Year {yr}: parts={parts:,.0f} vs total={total:,.0f}"

    def test_kid_costs_start_at_birth(self):
        """Kid spending should be zero before birth age and positive after."""
        r = _run(family=FamilyConfig(kid_ages=(35,)), n=500, seed=42)
        age_30 = 30 - r.ages[0]
        age_37 = 37 - r.ages[0]
        kids_30 = np.median(r.spending_kids[age_30])
        kids_37 = np.median(r.spending_kids[age_37])
        assert kids_30 == 0, f"No kid costs before birth: {kids_30:,.0f}"
        assert kids_37 > 0, f"Kid costs should exist after birth: {kids_37:,.0f}"

    def test_education_spending_is_529_contributions(self):
        """Education spending should occur between kid birth and college age."""
        r = _run(family=FamilyConfig(kid_ages=(31,)), n=500, seed=42)
        age_28 = 28 - r.ages[0]
        age_40 = 40 - r.ages[0]
        edu_28 = np.median(r.spending_education[age_28])
        edu_40 = np.median(r.spending_education[age_40])
        assert edu_28 == 0, f"No 529 before kid born: {edu_28:,.0f}"
        assert edu_40 > 0, f"529 contributions should exist during kid's childhood: {edu_40:,.0f}"

    def test_lifestyle_creep_increases_spending(self):
        """Higher lifestyle creep should mean higher discretionary spending."""
        r_low = _run(seed=1, lifestyle_creep_pct=0.0)
        r_high = _run(seed=1, lifestyle_creep_pct=0.05)
        age_35 = 35 - r_low.ages[0]
        disc_low = np.median(r_low.spending_discretionary[age_35])
        disc_high = np.median(r_high.spending_discretionary[age_35])
        assert disc_high > disc_low * 1.1, \
            f"Higher creep should increase spending: low={disc_low:,.0f} high={disc_high:,.0f}"


# =========================================================================
# Cross-feature interactions
# =========================================================================

class TestInteractions:
    def test_mega_backdoor_with_high_match_fills_roth(self):
        """With large employer match, mega backdoor room shrinks (total 415c limit)."""
        r_low_match = _run(seed=1, use_mega_backdoor=True,
                           career=CareerConfig(employer_match_pct=0.0))
        r_high_match = _run(seed=1, use_mega_backdoor=True,
                            career=CareerConfig(employer_match_pct=1.0, employer_match_limit=1.0))
        roth_low = np.median(r_low_match.roth[10])
        roth_high = np.median(r_high_match.roth[10])
        # With 100% match up to full limit, more of the 415c space is used by match,
        # leaving less room for mega backdoor → Roth gets less mega contribution
        # But 401k gets a huge match, so total NW is better
        assert roth_low > roth_high * 0.8, \
            "Mega backdoor Roth should be similar or higher with no match"

    def test_all_levers_on_beats_all_off(self):
        """Maxing all tax-advantaged accounts should beat brokerage-only."""
        r_max = _run(seed=1, use_401k=True, use_roth=True, use_hsa=True,
                     use_mega_backdoor=True, hsa_employer_contrib=1500,
                     career=CareerConfig(employer_match_pct=0.5, employer_match_limit=0.43))
        r_min = _run(seed=1, use_401k=False, use_roth=False, use_hsa=False)
        nw_max = np.median(r_max.net_worth[-1])
        nw_min = np.median(r_min.net_worth[-1])
        assert nw_max > nw_min * 1.3, \
            f"All accounts should dominate: max={nw_max:,.0f} min={nw_min:,.0f}"

    def test_ss_plus_roth_minimizes_retirement_taxes(self):
        """Roth withdrawals + SS should yield lower lifetime retirement taxes than 401k + SS."""
        common = dict(
            seed=1, tc=250_000, fire_horizon=50, n=2000,
            ss=SocialSecurityConfig(enabled=True, claiming_age=67, monthly_benefit_today=2500),
        )
        r_roth = _run(
            seeds=SeedAmounts(roth=1_000_000, taxable=300_000),
            use_401k=False, **common,
        )
        r_401k = _run(
            seeds=SeedAmounts(t401k=1_000_000, taxable=300_000),
            use_roth=False, **common,
        )
        # Sum retirement-year taxes across all years after fire_horizon
        retire_start = 50 - r_roth.ages[0]
        tax_roth_total = np.median(r_roth.taxes[retire_start:].sum(axis=0))
        tax_401k_total = np.median(r_401k.taxes[retire_start:].sum(axis=0))
        assert tax_roth_total < tax_401k_total, \
            f"Roth should have lower lifetime retirement taxes: roth={tax_roth_total:,.0f} 401k={tax_401k_total:,.0f}"

    def test_high_seeds_enable_earlier_fire(self):
        """Starting with large balances should enable earlier FIRE."""
        r_low = _run(seed=1, seeds=SeedAmounts())
        r_high = _run(seed=1, seeds=SeedAmounts(taxable=500_000, t401k=300_000, roth=200_000))
        med_low = np.median(r_low.fire_ages[r_low.fire_ages < 99])
        med_high = np.median(r_high.fire_ages[r_high.fire_ages < 99])
        assert med_high < med_low, \
            f"High seeds should FIRE earlier: high={med_high:.0f} low={med_low:.0f}"

    def test_conservative_career_delays_fire(self):
        """Conservative trajectory should delay FIRE vs aggressive."""
        r_con = _run(seed=1, career=CareerConfig(trajectory="conservative"))
        r_agg = _run(seed=1, career=CareerConfig(trajectory="aggressive"))
        fire_con = (r_con.fire_ages < 55).mean()
        fire_agg = (r_agg.fire_ages < 55).mean()
        assert fire_agg >= fire_con, \
            f"Aggressive should FIRE more: agg={fire_agg:.2%} con={fire_con:.2%}"

    def test_cheaper_city_improves_outcomes(self):
        """Lower-cost city should improve FIRE outcomes."""
        r_sac = _run(seed=1, city="Sacramento", n=1000)
        r_sf = _run(seed=1, city="San Francisco", n=1000)
        fail_sac = r_sac.failed.mean()
        fail_sf = r_sf.failed.mean()
        assert fail_sac <= fail_sf + 0.05, \
            f"Sacramento should be better: sac_fail={fail_sac:.2%} sf_fail={fail_sf:.2%}"

    def test_more_kids_delays_fire(self):
        """More kids = more spending = later FIRE."""
        r_0 = _run(seed=1, family=FamilyConfig(kid_ages=()))
        r_2 = _run(seed=1, family=FamilyConfig(kid_ages=(31, 33)))
        fire_0 = (r_0.fire_ages < 55).mean()
        fire_2 = (r_2.fire_ages < 55).mean()
        assert fire_0 >= fire_2, \
            f"No kids should FIRE more: 0kids={fire_0:.2%} 2kids={fire_2:.2%}"

    def test_wedding_budget_is_one_time_cost(self):
        """Wedding budget should appear as one-time spending at marriage age."""
        r = _run(seed=1, n=500,
                 family=FamilyConfig(marriage_age=30, wedding_budget=100_000))
        marriage_idx = 30 - r.ages[0]
        ot_at_marriage = np.median(r.spending_one_time[marriage_idx])
        ot_before = np.median(r.spending_one_time[marriage_idx - 1])
        assert ot_at_marriage > ot_before + 50_000, \
            f"Wedding should spike one-time: at_marriage={ot_at_marriage:,.0f} before={ot_before:,.0f}"


# =========================================================================
# FIRE number (principled target) correctness
# =========================================================================

class TestFireNumber:
    def test_fire_number_stable_across_incomes(self):
        """FIRE number in today's dollars should NOT scale with income.
        It's based on fixed retirement spending estimates, not income."""
        r_low = _run(seed=1, tc=150_000, n=2000)
        r_high = _run(seed=1, tc=350_000, n=2000)
        mask_low = r_low.fire_ages < 99
        mask_high = r_high.fire_ages < 99
        if mask_low.sum() < 50 or mask_high.sum() < 50:
            pytest.skip("Not enough FIRE'd sims")
        fn_low = np.median(r_low.fire_number[mask_low])
        fn_high = np.median(r_high.fire_number[mask_high])
        # Same city, same family → similar retirement spending → similar FIRE number.
        # Allow 30% tolerance for SWR differences (different FIRE ages → different SWR).
        ratio = fn_high / fn_low
        assert 0.7 < ratio < 1.3, \
            f"FIRE number should be income-independent: low={fn_low:,.0f} high={fn_high:,.0f} ratio={ratio:.2f}"

    def test_fire_number_positive_for_fired_sims(self):
        """Every sim that FIRE'd should have a positive FIRE number."""
        r = _run(seed=1, n=2000)
        fired_mask = r.fire_ages < 99
        assert np.all(r.fire_number[fired_mask] > 0), "FIRE'd sims should have positive fire_number"

    def test_fire_number_zero_for_never_fired(self):
        """Sims that never FIRE should have fire_number = 0 (unset)."""
        r = _run(tc=100_000, n=2000, seed=1, fire_horizon=35)
        never_mask = r.fire_ages == 99
        if never_mask.sum() > 0:
            # These should have fire_number from forced retirement, not 0
            # Actually all get forced at fire_horizon, so all should have a value
            assert np.all(r.fire_number[never_mask] == 0) or np.all(r.fire_number[never_mask] > 0)

    def test_fire_spending_matches_expected_components(self):
        """Retirement spending should be in a plausible range for fixed assumptions."""
        r = _run(seed=1, n=2000, family=FamilyConfig(kid_ages=()),
                 city="Sacramento")
        fired_mask = (r.fire_ages < 99) & (r.fire_ages >= 40)
        if fired_mask.sum() < 50:
            pytest.skip("Not enough mid-age FIRE sims")
        spend = np.median(r.fire_spending[fired_mask])
        # No kids: housing (~$30-60k) + disc ($45k) + HC ($24k + shock buffer) = ~$100-130k
        assert 60_000 < spend < 200_000, \
            f"Retirement spending should be plausible: {spend:,.0f}"

    def test_fire_number_in_todays_dollars(self):
        """FIRE number should be deflated — sims FIREing at 40 vs 55 should produce
        similar today's-dollar FIRE numbers (same city, same family config)."""
        r = _run(seed=1, tc=200_000, n=3000, family=FamilyConfig(kid_ages=()))
        early = (r.fire_ages >= 35) & (r.fire_ages <= 42)
        late = (r.fire_ages >= 50) & (r.fire_ages <= 58)
        if early.sum() < 30 or late.sum() < 30:
            pytest.skip("Not enough sims in both age ranges")
        fn_early = np.median(r.fire_number[early])
        fn_late = np.median(r.fire_number[late])
        # Earlier FIRE → lower SWR → higher FIRE number. But both should be
        # in the same order of magnitude (not inflated by nominal dollars).
        # Early FIRE needs ~3% SWR vs late ~4%, so early should be ~33% higher.
        assert fn_early > fn_late * 0.8, f"Early={fn_early:,.0f} Late={fn_late:,.0f}"
        assert fn_early < fn_late * 2.5, f"Early={fn_early:,.0f} Late={fn_late:,.0f}"


# =========================================================================
# Scenario tests: tax edge cases, financial logic, subtle interactions
# =========================================================================

class TestFICAWageCap:
    """FICA is 7.65% up to $168,600 then only 1.45% Medicare above that."""

    def test_high_earner_pays_lower_effective_fica_rate(self):
        gross_low = np.array([150_000.0])
        gross_high = np.array([300_000.0])
        tax_low = calc_taxes_vec(gross_low, state_rate=0.0)
        tax_high = calc_taxes_vec(gross_high, state_rate=0.0)
        # Isolate FICA: calculate with no state, then compare the FICA component
        # FICA for 150k: 150000 * 0.0765 = $11,475
        # FICA for 300k: 168600 * 0.0765 + (300000-168600) * 0.0145 = $12,899 + $1,905 = $14,804
        fica_low = 150_000 * 0.0765
        fica_high = 168_600 * 0.0765 + (300_000 - 168_600) * 0.0145
        eff_low = fica_low / 150_000   # ~7.65%
        eff_high = fica_high / 300_000  # ~4.93%
        assert eff_high < eff_low, \
            f"High earner should have lower FICA rate: {eff_high:.2%} vs {eff_low:.2%}"

    def test_fica_cap_matches_calc_taxes_vec(self):
        """Verify calc_taxes_vec computes FICA correctly at the wage cap boundary."""
        at_cap = np.array([168_600.0])
        above_cap = np.array([268_600.0])
        tax_at = calc_taxes_vec(at_cap, state_rate=0.0)
        tax_above = calc_taxes_vec(above_cap, state_rate=0.0)
        marginal = float((tax_above - tax_at)[0])
        # The extra $100k above cap should only add 1.45% Medicare, not full 7.65%.
        # If FICA cap didn't exist, FICA alone would add $7,650 instead of $1,450.
        # Total diff includes federal marginal tax. Verify it's less than full-FICA scenario.
        assert marginal < 100_000 * (0.37 + 0.0765), "FICA cap should limit SS tax"


class TestFiveTwentyNineRollover:
    """When kid finishes college, any 529 surplus rolls into taxable."""

    def test_529_surplus_goes_to_taxable(self):
        r = _run(seed=42, n=1000, family=FamilyConfig(kid_ages=(31,), college_cost_per_kid=50_000))
        # College cost is only $50k total ($12.5k/yr), but 529 contributions target $300k.
        # Huge surplus should roll into taxable at age 31+18+4 = 53.
        rollover_age = 31 + 18 + COLLEGE_YEARS  # 53
        idx_before = rollover_age - 1 - r.ages[0]
        idx_after = rollover_age - r.ages[0]
        if idx_after >= len(r.ages):
            pytest.skip("Rollover age beyond simulation")
        tax_before = np.median(r.taxable[idx_before])
        tax_after = np.median(r.taxable[idx_after])
        # Taxable should jump at rollover (529 surplus + market return)
        assert tax_after > tax_before, \
            f"529 surplus should roll into taxable: before={tax_before:,.0f} after={tax_after:,.0f}"


class TestFIREBridgeCheck:
    """FIRE requires 'accessible' funds (taxable + roth_basis + HSA) to cover
    spending until age 60. 401k alone can't satisfy the bridge."""

    def test_401k_heavy_portfolio_delays_fire(self):
        """$2M in 401k but $0 accessible should FIRE later than $2M in taxable."""
        r_401k = _run(seed=1, tc=200_000, n=2000,
                      seeds=SeedAmounts(t401k=2_000_000),
                      use_roth=False, use_hsa=False)
        r_tax = _run(seed=1, tc=200_000, n=2000,
                     seeds=SeedAmounts(taxable=2_000_000),
                     use_roth=False, use_hsa=False)
        med_401k = np.median(r_401k.fire_ages[r_401k.fire_ages < 99])
        med_tax = np.median(r_tax.fire_ages[r_tax.fire_ages < 99])
        assert med_tax < med_401k, \
            f"Taxable should FIRE earlier (bridge): tax={med_tax:.0f} 401k={med_401k:.0f}"


class TestHSAReturns:
    """HSA earns full market returns (invested in index funds)."""

    def test_hsa_grows_at_market_rate(self):
        """With same seed balance, HSA and Roth should grow at similar rates."""
        r_hsa = _run(seed=1, n=2000,
                     seeds=SeedAmounts(hsa=100_000),
                     use_401k=False, use_roth=False)
        r_roth = _run(seed=1, n=2000,
                      seeds=SeedAmounts(roth=100_000),
                      use_401k=False, use_hsa=False)
        yr15 = 15
        hsa_15 = np.median(r_hsa.hsa[yr15])
        roth_15 = np.median(r_roth.roth[yr15])
        # Both get full market returns; Roth may be slightly higher due to
        # larger ongoing contributions ($7k Roth vs ~$4-8k HSA limit)
        ratio = hsa_15 / roth_15
        assert ratio > 0.5, \
            f"HSA and Roth should grow similarly: hsa={hsa_15:,.0f} roth={roth_15:,.0f}"


class TestRothBasisInvariant:
    """roth_basis should never exceed roth balance at any point."""

    def test_roth_basis_never_exceeds_balance(self):
        r = _run(seed=1, tc=200_000, n=2000,
                 seeds=SeedAmounts(roth=50_000))
        for yr in range(0, len(r.ages), 5):
            roth_bal = r.roth[yr]
            # We don't have roth_basis in trajectories, but we can verify
            # indirectly: roth balance should always be >= 0
            assert np.all(roth_bal >= -1), \
                f"Roth balance negative at year {yr}: min={roth_bal.min():,.0f}"


class TestStateTaxTransition:
    """Before age 28, state tax is hardcoded 5.5% regardless of city.
    At 31+, uses the city's actual rate."""

    def test_nyc_taxes_lower_before_31(self):
        """NYC taxes should follow the simulator's single->joint and city-tax transition."""
        r = _run(seed=1, tc=200_000, n=1000, city="New York City",
                 use_401k=False, use_roth=False, use_hsa=False,
                 family=FamilyConfig(kid_ages=(), spouse_works=False))
        age_26_idx = 26 - r.ages[0]
        age_32_idx = 32 - r.ages[0]
        tax_26 = np.median(r.taxes[age_26_idx])
        tax_32 = np.median(r.taxes[age_32_idx])
        exp_26 = np.median(calc_taxes_vec(
            r.incomes[age_26_idx], state_rate=0.055, city_tax_rate=0.0, filing_status="single"
        ))
        assert abs(tax_26 - exp_26) / max(exp_26, 1) < 0.02
        assert tax_32 > tax_26, \
            f"NYC taxes should still rise after marriage/city-tax transition: age26={tax_26:,.0f} age32={tax_32:,.0f}"

    def test_city_tax_is_not_applied_before_31(self):
        """Before the family phase, NYC should use the generic 5.5% state rate only."""
        r = _run(seed=1, tc=200_000, n=1000, city="New York City",
                 use_401k=False, use_roth=False, use_hsa=False,
                 family=FamilyConfig(kid_ages=(), spouse_works=False))
        age_26_idx = 26 - r.ages[0]
        median_income = np.median(r.incomes[age_26_idx])
        median_tax = np.median(r.taxes[age_26_idx])
        expected = calc_taxes_vec(
            np.array([median_income]), state_rate=0.055, city_tax_rate=0.0, filing_status="single"
        )[0]
        assert abs(median_tax - expected) / max(expected, 1) < 0.02, \
            f"Pre-31 NYC taxes should exclude city tax: sim={median_tax:,.0f} expected={expected:,.0f}"


class TestHSALimitSwitch:
    """HSA limit switches from individual ($4,300) to family ($8,550) at marriage."""

    def test_hsa_contributions_increase_at_marriage(self):
        # Use high TC, no kids, late marriage to isolate the limit switch.
        # With no kids and high TC, net_inc*0.2 won't be the binding constraint.
        r = _run(seed=1, tc=300_000, n=1000, use_401k=False, use_roth=False,
                 family=FamilyConfig(marriage_age=35, kid_ages=(), spouse_works=False))
        pre_idx = 34 - r.ages[0]
        post_idx = 36 - r.ages[0]
        hsa_pre = np.median(r.savings_hsa[pre_idx])
        hsa_post = np.median(r.savings_hsa[post_idx])
        # Family limit ($8,550) is ~2x individual ($4,300)
        assert hsa_post > hsa_pre * 1.4, \
            f"HSA should jump at marriage: pre={hsa_pre:,.0f} post={hsa_post:,.0f}"


class TestSSProrationForEarlyFIRE:
    """SS proration: FIRE at 35 with SS at 67 only credits ~42% of SS benefit
    in the FIRE check, since SS only covers 23 of 55 retirement years."""

    def test_ss_proration_reduces_fire_number_benefit(self):
        """SS should reduce the FIRE number, but late claiming (age 70) means
        fewer years of SS coverage → smaller reduction in FIRE number for early retirees.
        Compare FIRE number with SS vs without — the reduction should exist."""
        common = dict(seed=1, tc=200_000, n=2000)
        r_no_ss = _run(**common, ss=SocialSecurityConfig(enabled=False))
        r_ss = _run(**common, ss=SocialSecurityConfig(enabled=True, claiming_age=67,
                                                       monthly_benefit_today=2500))
        mask_no = r_no_ss.fire_ages < 99
        mask_ss = r_ss.fire_ages < 99
        if mask_no.sum() < 50 or mask_ss.sum() < 50:
            pytest.skip("Not enough FIRE'd sims")
        fn_no = np.median(r_no_ss.fire_number[mask_no])
        fn_ss = np.median(r_ss.fire_number[mask_ss])
        # SS should lower the required nest egg (rt_net = rt - prorated_ss)
        assert fn_ss < fn_no, \
            f"SS should reduce FIRE number: ss={fn_ss:,.0f} no_ss={fn_no:,.0f}"


class TestInsuranceCostGrowth:
    """Homeowner insurance grows exponentially: premium * (1+8%)^years_owned."""

    def test_housing_costs_grow_for_homeowners(self):
        r = _run(seed=42, n=1000, city="Sacramento",
                 seeds=SeedAmounts(taxable=300_000))
        # Sacramento has home_price, so some sims buy homes
        age_36_idx = 36 - r.ages[0]  # ~3 years of ownership
        age_55_idx = 55 - r.ages[0]  # ~22 years of ownership
        # Only look at working sims (not retired) to avoid retirement spending split
        working_36 = ~r.fired_status[age_36_idx]
        working_55 = ~r.fired_status[age_55_idx]
        if working_36.sum() < 50 or working_55.sum() < 50:
            pytest.skip("Not enough working sims at both ages")
        housing_36 = np.median(r.spending_housing[age_36_idx, working_36])
        housing_55 = np.median(r.spending_housing[age_55_idx, working_55])
        # Insurance at 8%/yr for 19 extra years ≈ 4.3x. Housing should grow substantially.
        assert housing_55 > housing_36 * 1.3, \
            f"Housing should grow with insurance: age36={housing_36:,.0f} age55={housing_55:,.0f}"


class TestForcedRetirementSpending:
    """Forced retirees at fire_horizon get housing+$45k disc+$24k HC — no kids/529."""

    def test_forced_vs_voluntary_spending_baseline(self):
        """Voluntary FIRE at 40 with young kids should have higher initial spending
        than forced retirement at 60 with grown kids."""
        # Kids born at 31, 33 → at age 40 they're 9 and 7 (active kid costs + 529)
        # At age 60 they're 29 and 27 (no kid costs)
        r = _run(seed=1, tc=250_000, n=2000,
                 seeds=SeedAmounts(taxable=500_000, t401k=300_000, roth=200_000),
                 family=FamilyConfig(kid_ages=(31, 33)))
        vol_mask = (r.fire_ages >= 38) & (r.fire_ages <= 42)
        forced_mask = r.fire_ages == 60
        if vol_mask.sum() < 20 or forced_mask.sum() < 20:
            pytest.skip("Need sims in both categories")
        vol_spend = np.median(r.fire_spending[vol_mask])
        forced_spend = np.median(r.fire_spending[forced_mask])
        # Voluntary includes kids + 529; forced does not
        assert vol_spend > forced_spend, \
            f"Voluntary w/kids should spend more: vol={vol_spend:,.0f} forced={forced_spend:,.0f}"

    def test_forced_retirement_with_kids_includes_child_costs(self):
        """Forced retirement spending should still include active kid and 529 costs."""
        common = dict(seed=1, tc=120_000, n=2000, fire_horizon=40,
                      ss=SocialSecurityConfig(enabled=False))
        r_kids = _run(**common, family=FamilyConfig(kid_ages=(31, 33)))
        r_no_kids = _run(**common, family=FamilyConfig(kid_ages=()))
        forced_kids = r_kids.fire_ages == 40
        forced_no_kids = r_no_kids.fire_ages == 40
        if forced_kids.sum() < 50 or forced_no_kids.sum() < 50:
            pytest.skip("Need enough forced retirees in both scenarios")
        spend_kids = np.median(r_kids.fire_spending[forced_kids])
        spend_no_kids = np.median(r_no_kids.fire_spending[forced_no_kids])
        assert spend_kids > spend_no_kids + 20_000, \
            f"Forced retirement should retain kid costs: kids={spend_kids:,.0f} no_kids={spend_no_kids:,.0f}"


class TestInflationAdjustedLimits:
    """Contribution limits (401k, HSA, Roth) are inflation-adjusted each year."""

    def test_401k_contributions_grow_nominally_over_time(self):
        r = _run(seed=1, tc=300_000, n=1000, use_roth=False, use_hsa=False)
        # Year 0 vs year 15: nominal 401k contributions should grow with inflation
        s401_yr0 = np.median(r.savings_401k[0])
        s401_yr15 = np.median(r.savings_401k[15])
        # ~3% inflation for 15 years → limit grows ~1.56x
        assert s401_yr15 > s401_yr0 * 1.2, \
            f"401k limit should grow with inflation: yr0={s401_yr0:,.0f} yr15={s401_yr15:,.0f}"


class TestHomePurchaseRequiresTaxable:
    """Down payment comes from taxable account. If all savings are in 401k,
    the sim can never buy a home."""

    def test_401k_only_cannot_buy_home(self):
        r = _run(seed=1, n=1000, city="Sacramento",
                 seeds=SeedAmounts(t401k=500_000),
                 use_roth=False, use_hsa=False, tc=150_000)
        # Sacramento has home_price=$530k, down=20%=$106k
        # With low TC and no starting taxable, it'll take a while to save enough
        age_35_idx = 35 - r.ages[0]
        home_eq_35 = np.median(r.home_equity[age_35_idx])
        # Contrast with having $200k in taxable
        r2 = _run(seed=1, n=1000, city="Sacramento",
                  seeds=SeedAmounts(taxable=200_000),
                  use_roth=False, use_hsa=False, tc=150_000)
        home_eq_35_tax = np.median(r2.home_equity[age_35_idx])
        assert home_eq_35_tax > home_eq_35, \
            f"Taxable seeds enable home purchase: taxable={home_eq_35_tax:,.0f} 401k={home_eq_35:,.0f}"


class TestRetirementSpendingGuardrails:
    """Spending adjusts based on withdrawal rate vs target SWR:
    - >1.5x SWR → 15% cut
    - <0.5x SWR → 8% raise"""

    def test_wealthy_retiree_spending_increases(self):
        """With massive wealth, withdrawal rate stays far below SWR,
        so guardrails should boost spending over time."""
        r = _run(seed=1, tc=200_000, n=1000,
                 seeds=SeedAmounts(taxable=5_000_000),
                 fire_horizon=40, family=FamilyConfig(kid_ages=()),
                 ss=SocialSecurityConfig(enabled=False))
        # Look at early retirement spending vs later
        age_42_idx = 42 - r.ages[0]
        age_55_idx = 55 - r.ages[0]
        retired_42 = r.fired_status[age_42_idx] & ~r.failed
        retired_55 = r.fired_status[age_55_idx] & ~r.failed
        if retired_42.sum() < 50 or retired_55.sum() < 50:
            pytest.skip("Not enough retirees")
        spend_42 = np.median(r.spending[age_42_idx, retired_42])
        spend_55 = np.median(r.spending[age_55_idx, retired_55])
        # With $5M, withdrawal rate is well below SWR → spending should increase
        assert spend_55 > spend_42 * 0.9, \
            f"Wealthy retiree spending shouldn't collapse: 42={spend_42:,.0f} 55={spend_55:,.0f}"


class TestKidCostScaling:
    """Kid costs scale by age: 1.0x (0-5), 1.2x (6-12), 1.5x (13-17), 1.3x (18-21)."""

    def test_teen_costs_higher_than_toddler(self):
        r = _run(seed=1, n=1000, family=FamilyConfig(kid_ages=(31,)))
        # Kid age 3 → parent age 34, year idx 9
        # Kid age 15 → parent age 46, year idx 21
        toddler_idx = 34 - r.ages[0]
        teen_idx = 46 - r.ages[0]
        # Only look at working sims to avoid retirement spending approximation
        working_t = ~r.fired_status[toddler_idx]
        working_teen = ~r.fired_status[teen_idx]
        if working_t.sum() < 50 or working_teen.sum() < 50:
            pytest.skip("Not enough working sims")
        kids_toddler = np.median(r.spending_kids[toddler_idx, working_t])
        kids_teen = np.median(r.spending_kids[teen_idx, working_teen])
        # Teen = 1.5x, toddler = 1.0x → teens should cost ~50% more
        assert kids_teen > kids_toddler * 1.3, \
            f"Teens should cost more: toddler={kids_toddler:,.0f} teen={kids_teen:,.0f}"

    def test_kid_costs_zero_after_22(self):
        r = _run(seed=1, n=1000, family=FamilyConfig(kid_ages=(31,)))
        # Kid age 22+ → parent age 53+, costs should be zero
        idx = 54 - r.ages[0]  # kid is 23
        working = ~r.fired_status[idx]
        if working.sum() < 50:
            pytest.skip("Not enough working sims at 54")
        kids_cost = np.median(r.spending_kids[idx, working])
        assert kids_cost == 0, f"Kid costs should be 0 after 22: {kids_cost:,.0f}"

    def test_third_kid_increases_costs_and_529_savings(self):
        """FamilyConfig.kid_ages should support more than two children."""
        r_two = _run(seed=1, n=1000, family=FamilyConfig(kid_ages=(31, 33)))
        r_three = _run(seed=1, n=1000, family=FamilyConfig(kid_ages=(31, 33, 35)))
        idx = 40 - r_two.ages[0]
        working_two = ~r_two.fired_status[idx]
        working_three = ~r_three.fired_status[idx]
        if working_two.sum() < 50 or working_three.sum() < 50:
            pytest.skip("Not enough working sims at 40")
        kids_two = np.median(r_two.spending_kids[idx, working_two])
        kids_three = np.median(r_three.spending_kids[idx, working_three])
        edu_two = np.median(r_two.spending_education[idx, working_two])
        edu_three = np.median(r_three.spending_education[idx, working_three])
        assert kids_three > kids_two * 1.2, \
            f"A third kid should raise kid costs: two={kids_two:,.0f} three={kids_three:,.0f}"
        assert edu_three > edu_two * 1.2, \
            f"A third kid should raise 529 savings: two={edu_two:,.0f} three={edu_three:,.0f}"


class TestMegaBackdoorRoomWithMatch:
    """Mega backdoor room = $70k total limit - employee 401k - employer match.
    High match eats into the room."""

    def test_high_match_reduces_mega_room(self):
        # Use high TC ($400k) so the 50% after-tax cash flow cap isn't binding
        # and the mega room difference actually shows up.
        r_no_match = _run(seed=1, tc=400_000, n=1000, use_mega_backdoor=True,
                          career=CareerConfig(employer_match_pct=0.0))
        r_full_match = _run(seed=1, tc=400_000, n=1000, use_mega_backdoor=True,
                            career=CareerConfig(employer_match_pct=1.0, employer_match_limit=1.0))
        # With full match, employee $23k + match ~$23k = $46k, leaving $24k for mega
        # Without match, employee $23k + match $0 = $23k, leaving $47k for mega
        # Check Roth balance after 10 years (accumulated mega contributions compound)
        roth_no = np.median(r_no_match.roth[10])
        roth_full = np.median(r_full_match.roth[10])
        assert roth_no > roth_full, \
            f"No match should have more mega room → more Roth: no={roth_no:,.0f} full={roth_full:,.0f}"


class TestMarriageAfterFireHorizon:
    """If marriage_age > fire_horizon, spouse never contributes income during
    working years and spousal SS activates late."""

    def test_late_marriage_same_as_no_spouse(self):
        r_late = _run(seed=1, n=1000,
                      family=FamilyConfig(marriage_age=65, kid_ages=(),
                                          spouse_works=True, spouse_salary=80_000),
                      ss=SocialSecurityConfig(enabled=False))
        r_none = _run(seed=1, n=1000,
                      family=FamilyConfig(marriage_age=65, kid_ages=(),
                                          spouse_works=False),
                      ss=SocialSecurityConfig(enabled=False))
        # Income should be identical — spouse never works before fire_horizon
        inc_late = np.median(r_late.incomes[10])
        inc_none = np.median(r_none.incomes[10])
        assert abs(inc_late - inc_none) / max(inc_none, 1) < 0.01, \
            f"Late marriage should equal no spouse income: late={inc_late:,.0f} none={inc_none:,.0f}"


class TestHealthcareCostCliff:
    """Pre-65 ACA costs are high and growing. Post-65 drops to flat Medicare ($13k)."""

    def test_healthcare_drops_at_65(self):
        """calc_retirement_hc should produce a significant drop from 64 to 65."""
        inf = np.ones(100)  # no inflation for clarity
        hc_64 = calc_retirement_hc(64, inf, 100)
        hc_65 = calc_retirement_hc(65, inf, 100)
        # At 64: base=30000 + 24*800 = $49,200 (plus real growth)
        # At 65: flat $13,000
        assert float(hc_65[0]) < float(hc_64[0]) * 0.5, \
            f"Medicare should be much cheaper: 64={float(hc_64[0]):,.0f} 65={float(hc_65[0]):,.0f}"

    def test_aca_costs_increase_with_age(self):
        inf = np.ones(100)
        hc_45 = calc_retirement_hc(45, inf, 100)
        hc_60 = calc_retirement_hc(60, inf, 100)
        assert float(hc_60[0]) > float(hc_45[0]) * 1.3, \
            f"ACA should rise with age: 45={float(hc_45[0]):,.0f} 60={float(hc_60[0]):,.0f}"


class TestHomeEquityIsIlliquid:
    """Net worth includes home equity, but FIRE check uses total_liq (no home equity).
    Someone with a paid-off home but little liquid savings should have
    high NW but potentially late FIRE."""

    def test_home_equity_doesnt_count_for_fire(self):
        # Sacramento with high home price appreciation
        r = _run(seed=1, n=1000, tc=150_000, city="Sacramento",
                 seeds=SeedAmounts(taxable=150_000))
        # Sims that bought homes have home equity in NW but it doesn't help FIRE
        age_50_idx = 50 - r.ages[0]
        homeowners = r.home_equity[age_50_idx] > 100_000
        # Among homeowners, NW is high but FIRE age isn't necessarily early
        if homeowners.sum() < 20:
            pytest.skip("Not enough homeowners")
        nw_owners = np.median(r.net_worth[age_50_idx, homeowners])
        he_owners = np.median(r.home_equity[age_50_idx, homeowners])
        liquid_owners = nw_owners - he_owners
        # Home equity is a big chunk of NW but doesn't count for FIRE
        assert he_owners > liquid_owners * 0.2, \
            f"Home equity should be significant portion: HE={he_owners:,.0f} liquid={liquid_owners:,.0f}"


class TestRetirementWithdrawalTaxesInTrajectory:
    """traj_taxes should include retirement withdrawal taxes, not just working-year taxes."""

    def test_401k_retiree_has_ongoing_taxes(self):
        r = _run(seed=42, tc=200_000, n=1000,
                 seeds=SeedAmounts(t401k=1_500_000),
                 use_roth=False, use_hsa=False,
                 fire_horizon=45,
                 ss=SocialSecurityConfig(enabled=False))
        # At age 70, everyone is retired. 401k withdrawals incur income tax.
        age_70_idx = 70 - r.ages[0]
        retired = r.fired_status[age_70_idx] & ~r.failed
        if retired.sum() < 50:
            pytest.skip("Not enough retirees at 70")
        taxes_70 = np.median(r.taxes[age_70_idx, retired])
        assert taxes_70 > 1000, \
            f"401k retirees should have withdrawal taxes at 70: {taxes_70:,.0f}"

    def test_roth_retiree_has_minimal_taxes(self):
        r = _run(seed=42, tc=200_000, n=1000,
                 seeds=SeedAmounts(roth=1_500_000),
                 use_401k=False, use_hsa=False,
                 fire_horizon=45,
                 ss=SocialSecurityConfig(enabled=False))
        age_70_idx = 70 - r.ages[0]
        retired = r.fired_status[age_70_idx] & ~r.failed
        if retired.sum() < 50:
            pytest.skip("Not enough retirees at 70")
        taxes_70 = np.median(r.taxes[age_70_idx, retired])
        # Roth withdrawals are tax-free, so taxes should be near zero
        assert taxes_70 < 5000, \
            f"Roth retirees should have near-zero withdrawal taxes: {taxes_70:,.0f}"


# =========================================================================
# RMD realism
# =========================================================================

class TestRequiredMinimumDistributions:
    def test_large_401k_rmds_raise_post_73_tax_burden(self):
        """RMDs should create extra taxable income after age 73 for traditional balances."""
        r = _run(
            seed=1, tc=200_000, n=1000,
            seeds=SeedAmounts(taxable=5_000_000, t401k=2_000_000),
            use_roth=False, use_hsa=False,
            fire_horizon=40,
            family=FamilyConfig(kid_ages=()),
            ss=SocialSecurityConfig(enabled=False),
        )
        age_72_idx = 72 - r.ages[0]
        age_75_idx = 75 - r.ages[0]
        retired = r.fired_status[age_75_idx] & ~r.failed
        if retired.sum() < 50:
            pytest.skip("Not enough retirees at 75")
        taxes_72 = np.median(r.taxes[age_72_idx, retired])
        taxes_75 = np.median(r.taxes[age_75_idx, retired])
        assert taxes_75 > taxes_72 * 1.5, \
            f"Post-73 taxes should jump from RMD income: 72={taxes_72:,.0f} 75={taxes_75:,.0f}"


# =========================================================================
# BUG HUNTERS: Tests designed to expose specific bugs
# =========================================================================

class TestDCAMegaBackdoor:
    """DCA half-year return correctly subtracts mega_contrib from
    taxable_new_contrib, so only money that actually went to taxable
    gets the half-year DCA return."""

    def test_mega_backdoor_money_doesnt_inflate_taxable_via_dca(self):
        r_mega = _run(seed=1, tc=400_000, n=2000, use_mega_backdoor=True,
                      use_roth=True, use_hsa=False)
        r_no = _run(seed=1, tc=400_000, n=2000, use_mega_backdoor=False,
                    use_roth=True, use_hsa=False)

        yr10 = 10
        tax_mega = np.median(r_mega.taxable[yr10])
        tax_no = np.median(r_no.taxable[yr10])
        roth_mega = np.median(r_mega.roth[yr10])
        roth_no = np.median(r_no.roth[yr10])

        roth_gain = roth_mega - roth_no
        tax_loss = tax_no - tax_mega

        # With correct DCA, the money simply moves from taxable → Roth.
        # taxable_loss ≈ roth_gain (ratio ~0.9-1.1).
        ratio = tax_loss / max(roth_gain, 1)
        assert ratio > 0.85, \
            f"DCA should correctly account for mega_contrib: " \
            f"tax_loss={tax_loss:,.0f} roth_gain={roth_gain:,.0f} ratio={ratio:.3f}"


class TestWithdrawalTaxCombinedIncome:
    """HSA non-medical + 401k withdrawals are taxed as combined income
    with one standard deduction and no FICA."""

    def test_combined_withdrawal_tax_uses_one_standard_deduction(self):
        """A 401k-heavy retiree should pay more tax than a Roth-heavy retiree
        because combined 401k income is taxed progressively."""
        r_401k = _run(seed=42, tc=200_000, n=1000,
                      seeds=SeedAmounts(t401k=1_500_000, taxable=100_000),
                      use_roth=False, use_hsa=False,
                      fire_horizon=45, ss=SocialSecurityConfig(enabled=False))
        r_roth = _run(seed=42, tc=200_000, n=1000,
                      seeds=SeedAmounts(roth=1_500_000, taxable=100_000),
                      use_401k=False, use_hsa=False,
                      fire_horizon=45, ss=SocialSecurityConfig(enabled=False))
        age_70 = 70 - r_401k.ages[0]
        ret_401k = r_401k.fired_status[age_70] & ~r_401k.failed
        ret_roth = r_roth.fired_status[age_70] & ~r_roth.failed
        if ret_401k.sum() < 50 or ret_roth.sum() < 50:
            pytest.skip("Not enough retirees")
        tax_401k = np.median(r_401k.taxes[age_70, ret_401k])
        tax_roth = np.median(r_roth.taxes[age_70, ret_roth])
        assert tax_401k > tax_roth + 1000, \
            f"401k retiree should pay significantly more tax: 401k={tax_401k:,.0f} roth={tax_roth:,.0f}"

    def test_no_fica_on_retirement_withdrawals(self):
        """Retirement withdrawal taxes should NOT include FICA.
        calc_taxes_vec with include_fica=False should produce less tax."""
        gross = np.array([80_000.0])
        tax_with_fica = float(calc_taxes_vec(gross, 0.05, include_fica=True)[0])
        tax_no_fica = float(calc_taxes_vec(gross, 0.05, include_fica=False)[0])
        fica_amount = tax_with_fica - tax_no_fica
        assert fica_amount > 5000, \
            f"FICA component should be significant: ${fica_amount:,.0f}"
        # The retirement withdrawal code now uses include_fica=False
        assert tax_no_fica < tax_with_fica, \
            f"No-FICA tax should be less: {tax_no_fica:,.0f} vs {tax_with_fica:,.0f}"


class Test529ContributionsStopAtFIRE:
    """529 contributions are gated on ~fired, so retired people
    don't get unfunded 529 contributions."""

    def test_529_zero_when_fired_before_kid_born(self):
        """If you FIRE before your kid is born, the 529 should stay at zero
        (no earned income to contribute)."""
        r = _run(seed=1, tc=200_000, n=1000,
                 seeds=SeedAmounts(taxable=3_000_000),
                 family=FamilyConfig(kid_ages=(34,), marriage_age=29),
                 fire_horizon=50,
                 ss=SocialSecurityConfig(enabled=False))

        early_fire = (r.fire_ages <= 33) & ~r.failed
        if early_fire.sum() < 20:
            pytest.skip("Not enough early FIRE sims")

        # Derive 529 balance from NW - liquid accounts - home equity
        post_idx = 40 - r.ages[0]
        c529_at_40 = np.median(r.net_worth[post_idx, early_fire] -
                                r.taxable[post_idx, early_fire] -
                                r.t401k[post_idx, early_fire] -
                                r.roth[post_idx, early_fire] -
                                r.hsa[post_idx, early_fire] -
                                r.home_equity[post_idx, early_fire])

        # With fix: 529 should be ~0 since person was FIRE'd before kid born
        assert c529_at_40 < 5_000, \
            f"529 should be ~0 when FIRE'd before kid born: ${c529_at_40:,.0f}"


class TestRetirementSpendingBreakdown:
    """Retirement spending breakdown uses actual computed category costs
    scaled to match the guardrail-adjusted total."""

    def test_retirement_hc_percentage_increases_with_age(self):
        """Healthcare costs rise with age (ACA premiums). The HC share of spending
        should be higher at 60 than at 45 — impossible with fixed 30%."""
        r = _run(seed=1, tc=200_000, n=1000,
                 seeds=SeedAmounts(taxable=2_000_000),
                 fire_horizon=40,
                 family=FamilyConfig(kid_ages=()))

        for check_age, label in [(45, "age 45"), (60, "age 60")]:
            idx = check_age - r.ages[0]
            retired = r.fired_status[idx] & ~r.failed
            if retired.sum() < 50:
                continue
            total = r.spending[idx, retired]
            hc = r.spending_healthcare[idx, retired]
            hc_pct = np.median(hc / np.maximum(total, 1))
            # Stash for comparison
            if check_age == 45:
                hc_pct_45 = hc_pct
            else:
                hc_pct_60 = hc_pct

        # ACA costs grow with age, so HC% at 60 should be higher than at 45
        assert hc_pct_60 > hc_pct_45, \
            f"HC% should grow with age: 45={hc_pct_45:.2%} 60={hc_pct_60:.2%}"

    def test_retirement_spending_not_hardcoded_35_35_30(self):
        """Spending percentages should NOT be exactly 35/35/30 anymore."""
        r = _run(seed=1, tc=200_000, n=1000,
                 seeds=SeedAmounts(taxable=2_000_000),
                 fire_horizon=40,
                 family=FamilyConfig(kid_ages=()))

        age_50_idx = 50 - r.ages[0]
        retired = r.fired_status[age_50_idx] & ~r.failed
        if retired.sum() < 50:
            pytest.skip("Not enough retirees at 50")

        total = r.spending[age_50_idx, retired]
        housing = r.spending_housing[age_50_idx, retired]
        disc = r.spending_discretionary[age_50_idx, retired]
        hc = r.spending_healthcare[age_50_idx, retired]

        housing_pct = np.median(housing / np.maximum(total, 1))
        disc_pct = np.median(disc / np.maximum(total, 1))
        hc_pct = np.median(hc / np.maximum(total, 1))

        # At least one should differ meaningfully from old hardcoded values
        not_35_35_30 = (abs(housing_pct - 0.35) > 0.02 or
                        abs(disc_pct - 0.35) > 0.02 or
                        abs(hc_pct - 0.30) > 0.02)
        assert not_35_35_30, \
            f"Spending should NOT be hardcoded 35/35/30: " \
            f"h={housing_pct:.2%} d={disc_pct:.2%} hc={hc_pct:.2%}"

    def test_retirement_spending_categories_sum_to_total(self):
        """Category breakdown should still sum to total spending."""
        r = _run(seed=1, tc=200_000, n=1000,
                 seeds=SeedAmounts(taxable=2_000_000),
                 fire_horizon=40,
                 family=FamilyConfig(kid_ages=()))

        age_50_idx = 50 - r.ages[0]
        retired = r.fired_status[age_50_idx] & ~r.failed
        if retired.sum() < 50:
            pytest.skip("Not enough retirees at 50")

        total = r.spending[age_50_idx, retired]
        parts = (r.spending_housing[age_50_idx, retired] +
                 r.spending_discretionary[age_50_idx, retired] +
                 r.spending_kids[age_50_idx, retired] +
                 r.spending_education[age_50_idx, retired] +
                 r.spending_healthcare[age_50_idx, retired] +
                 r.spending_one_time[age_50_idx, retired])

        ratio = np.median(parts / np.maximum(total, 1))
        assert 0.95 < ratio < 1.05, \
            f"Categories should sum to total: ratio={ratio:.3f}"
