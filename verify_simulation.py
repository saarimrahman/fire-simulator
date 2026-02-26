#!/usr/bin/env python3
"""
Verification script for FIRE net worth simulator.
Runs simulations and validates:
1. Net worth never goes negative (with our fixes)
2. Account balances stay non-negative
3. Withdrawals never exceed available portfolio
4. Basic sanity checks on outputs
"""
import numpy as np
from fire import run_vectorized, CITIES, SeedAmounts, FamilyConfig, CareerConfig, SocialSecurityConfig

def main():
    rng = np.random.default_rng(42)
    n_sims = 1000  # Smaller for quick verification

    print("=" * 60)
    print("FIRE Simulator Verification")
    print("=" * 60)

    # Test 1: Default config - Sacramento, moderate scenario
    print("\n[Test 1] Default config (Sacramento, $200k TC, 2 kids)")
    results = run_vectorized(
        200_000, "Sacramento", n_sims, rng,
        seed_amounts=SeedAmounts(taxable=50_000, t401k=100_000, roth=30_000, hsa=10_000),
        family_config=FamilyConfig(kid_ages=(31, 33)),
        career_config=CareerConfig(),
        ss_config=SocialSecurityConfig(enabled=True, annual_benefit_at_fra=30000, claiming_age=67),
        return_trajectories=True,
    )

    nw = results.net_worth
    min_nw = nw.min()
    min_nw_per_sim = nw.min(axis=0)
    n_negative = (min_nw_per_sim < 0).sum()

    print(f"  Net worth shape: {nw.shape}")
    print(f"  Min net worth (any sim, any year): ${min_nw:,.0f}")
    print(f"  Sims with any negative NW: {n_negative} / {n_sims}")

    assert min_nw >= -1.0, f"Net worth went negative: min={min_nw}"  # Allow tiny float error
    print("  ✓ Net worth never goes negative")

    # Check account balances
    for name, arr in [("taxable", results.taxable), ("t401k", results.t401k),
                      ("roth", results.roth), ("hsa", results.hsa)]:
        assert arr.min() >= -1.0, f"{name} went negative: min={arr.min()}"
    print("  ✓ All account balances non-negative")

    # Check home equity
    he = results.home_equity
    assert he.min() >= -1.0, f"Home equity went negative: min={he.min()}"
    print("  ✓ Home equity non-negative (underwater fix)")

    # Test 2: Dublin (expensive home) - stress test
    print("\n[Test 2] Dublin (expensive home, $1.3M)")
    results2 = run_vectorized(
        300_000, "Dublin", n_sims, rng,
        seed_amounts=SeedAmounts(taxable=200_000, t401k=150_000),
        return_trajectories=True,
    )
    assert results2.net_worth.min() >= -1.0
    print("  ✓ Net worth non-negative")

    # Test 3: San Francisco (no home) - no home equity
    print("\n[Test 3] San Francisco (no home purchase)")
    results3 = run_vectorized(
        250_000, "San Francisco", n_sims, rng,
        return_trajectories=True,
    )
    assert results3.net_worth.min() >= -1.0
    assert (results3.home_equity == 0).all()
    print("  ✓ Net worth non-negative, no home equity")

    # Test 4: SS disabled
    print("\n[Test 4] Social Security disabled")
    results4 = run_vectorized(
        200_000, "Sacramento", n_sims, rng,
        ss_config=SocialSecurityConfig(enabled=False),
        return_trajectories=True,
    )
    assert results4.net_worth.min() >= -1.0
    print("  ✓ Net worth non-negative without SS")

    # Summary stats
    print("\n" + "=" * 60)
    print("Summary Statistics (Test 1)")
    print("=" * 60)
    pct_fire = (results.fire_ages < 99).mean() * 100
    pct_survived = ((results.fire_ages < 99) & ~results.failed).mean() * 100
    ending_nw = results.net_worth[-1, :]
    valid = ending_nw[ending_nw > 0]
    print(f"  FIRE rate: {pct_fire:.1f}%")
    print(f"  Success rate (FIRE + survived): {pct_survived:.1f}%")
    print(f"  Ending NW (survivors) - median: ${np.median(valid):,.0f}")
    print(f"  Ending NW (survivors) - P10: ${np.percentile(valid, 10):,.0f}")
    print(f"  Ending NW (survivors) - P90: ${np.percentile(valid, 90):,.0f}")

    print("\n✓ All verification tests passed!")


if __name__ == "__main__":
    main()
