import numpy as np

from fire import (
    run_vectorized,
    SeedAmounts,
    FamilyConfig,
    CareerConfig,
    SocialSecurityConfig,
)


def _run(label: str, *, ss: SocialSecurityConfig) -> None:
    rng = np.random.default_rng(42)
    results = run_vectorized(
        200_000,
        "Sacramento",
        5_000,
        rng,
        seed_amounts=SeedAmounts(taxable=0, t401k=0, roth=0, hsa=0),
        family_config=FamilyConfig(),
        career_config=CareerConfig(),
        social_security_config=ss,
        return_trajectories=True,
        life_expectancy=90,
        current_age=25,
    )

    nw = results.net_worth
    if not np.isfinite(nw).all():
        raise AssertionError(f"{label}: net_worth contains NaN/Inf")
    min_nw = float(np.min(nw))
    p1 = float(np.percentile(nw[-1], 1))
    p50 = float(np.percentile(nw[-1], 50))
    p99 = float(np.percentile(nw[-1], 99))
    print(f"{label}: net_worth min={min_nw:,.2f} ending P1/P50/P99={p1:,.0f}/{p50:,.0f}/{p99:,.0f}")

    # With current model (no debt), net worth should never go meaningfully negative.
    if min_nw < -1e-6:
        raise AssertionError(f"{label}: net_worth went negative (min={min_nw})")


def main() -> None:
    _run("baseline_no_ss", ss=SocialSecurityConfig(enabled=False))
    _run(
        "with_ss",
        ss=SocialSecurityConfig(
            enabled=True,
            claim_age=67,
            full_retirement_age=67,
            annual_benefit_at_fra=30_000,
            spouse_enabled=True,
            spouse_claim_age=67,
            spouse_annual_benefit_at_fra=20_000,
        ),
    )


if __name__ == "__main__":
    main()

