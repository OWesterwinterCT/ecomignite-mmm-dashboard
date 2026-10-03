"""Numerical and constraint checks using synthetic paired posterior draws."""
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from ecomignite.engine import (
    adstock_weights, forward_media_draws, hill_response, nonmedia_draws, optimize_budget,
)


def bundle_factory(zero_recent=False):
    rng = np.random.default_rng(22)
    samples, periods = 36, 48
    channels = ["meta_conversion_spend", "google_search_spend", "google_display_spend"]
    dates = pd.date_range("2024-01-01", periods=periods, freq="7D")
    spends = rng.uniform(45, 150, (periods, 3))
    if zero_recent:
        spends[-8:, 0] = 0
    weekly = pd.DataFrame(spends, columns=channels)
    weekly.insert(0, "date", dates)
    weekly["eligible_orders"] = 300
    weekly["discount_rate"] = rng.uniform(0.01, 0.05, periods)
    posterior = {
        "alpha": rng.beta(2, 3, (samples, 3)),
        "beta": rng.uniform(0.10, 0.40, (samples, 3)),
        "kappa": rng.uniform(0.5, 1.5, (samples, 3)),
        "slope": rng.uniform(0.6, 1.9, (samples, 3)),
        "channel_scale": spends.max(axis=0),
        "target_scale": 600.0,
        "intercept": rng.uniform(0.2, 0.3, samples),
        "gamma_control": np.tile([0.05, 0.01], (samples, 1)),
        "gamma_fourier": np.tile([0.01, 0.02, 0.03, 0.04], (samples, 1)),
    }
    posterior["alpha"][:, 1] = rng.beta(2, 98, samples)
    manifest = {
        "channels": channels,
        "controls": ["discount_rate_scaled", "linear_trend_scaled"],
        "yearly_seasonality": 2,
        "adstock_l_max": 8,
        "adstock_normalized": True,
        "adstock_first": True,
        "control_scaling": {"discount_unit": 0.1, "trend_origin": str(dates[0].date()),
                            "trend_span_days": (dates[-1] - dates[0]).days},
    }
    return SimpleNamespace(manifest=manifest, weekly=weekly, posterior=posterior)


class ForwardMathTests(unittest.TestCase):
    def test_direct_convolution_matches_varying_forward_spend(self):
        bundle = bundle_factory()
        p = bundle.posterior
        count, horizon = len(bundle.manifest["channels"]), 13
        future = np.random.default_rng(44).uniform(0, 180, (horizon, count))
        actual = forward_media_draws(bundle, future)
        past = bundle.weekly[bundle.manifest["channels"]].to_numpy()
        sequence = np.vstack([past, future]) / p["channel_scale"]
        expected = np.empty_like(actual)
        weights = adstock_weights(p["alpha"], 8)
        for sample in range(len(p["alpha"])):
            for channel in range(count):
                x = np.convolve(sequence[:, channel], weights[sample, channel])[:len(sequence)][-horizon:]
                expected[sample, :, channel] = (
                    p["target_scale"] * p["beta"][sample, channel] * x ** p["slope"][sample, channel]
                    / (x ** p["slope"][sample, channel] + p["kappa"][sample, channel] ** p["slope"][sample, channel])
                )
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-10)

    def test_zero_safe_fractional_hill_and_normalization(self):
        result = hill_response(np.array([0, 0.5, 1]), 0.3, 1, 0.6)
        self.assertEqual(result[0], 0)
        self.assertTrue(np.isfinite(result).all())
        weights = adstock_weights(np.array([[0, 0.4, 1.0]]), 8)
        np.testing.assert_allclose(weights.sum(axis=2), 1)
        np.testing.assert_allclose(weights[0, 0], [1, 0, 0, 0, 0, 0, 0, 0])

    def test_nonmedia_scaling_and_fourier_order(self):
        bundle = bundle_factory()
        dates = pd.date_range("2025-01-06", periods=2, freq="7D")
        total, parts = nonmedia_draws(bundle, dates, 0.1)
        p, scaling = bundle.posterior, bundle.manifest["control_scaling"]
        phase = 2 * np.pi * dates.dayofyear.to_numpy()[:, None] / 365.25 * np.array([1, 2])
        modes = np.concatenate([np.sin(phase), np.cos(phase)], axis=1)
        trend = (dates - pd.Timestamp(scaling["trend_origin"])).days / scaling["trend_span_days"]
        expected = 600 * (p["intercept"][:, None] + 0.05 + 0.01 * np.asarray(trend)[None, :]
                          + p["gamma_fourier"] @ modes.T)
        np.testing.assert_allclose(total, expected)
        np.testing.assert_allclose(total, sum(parts.values()))


class OptimizerTests(unittest.TestCase):
    def settings(self, **extra):
        settings = {"horizon_weeks": 8, "n_starts": 3, "optimization_draws": None}
        settings.update(extra)
        return settings

    def validate(self, bundle, result):
        table, summary = result["allocation"], result["summary"]
        recommended = table.set_index("channel")["optimized_weekly_spend"].reindex(bundle.manifest["channels"])
        self.assertTrue((table["optimized_weekly_spend"] >= table["min_weekly_spend"] - 1e-6).all())
        self.assertTrue((table["optimized_weekly_spend"] <= table["max_weekly_spend"] + 1e-6).all())
        self.assertLessEqual(summary["optimized_period_budget"], summary["total_budget_cap"] + 1e-6)
        self.assertAlmostEqual(table["reference_period_spend"].sum(), table["optimized_period_spend"].sum(), places=6)
        self.assertEqual(result["weekly_plan"]["date"].iloc[0], bundle.weekly["date"].max() + pd.Timedelta(days=7))
        direct = forward_media_draws(bundle, np.tile(recommended, (summary["horizon_weeks"], 1))).sum(axis=2).mean(axis=0)
        np.testing.assert_allclose(result["weekly_forecast"]["optimized_media_orders"], direct)
        self.assertAlmostEqual(table["optimized_media_orders"].sum(), summary["optimized_total_media_orders"])
        c = result["posterior_comparison"]
        np.testing.assert_allclose(c["optimized_total_orders"] - c["reference_total_orders"],
                                   c["optimized_media_orders"] - c["reference_media_orders"], atol=1e-10)

    def test_budget_search_and_12_week_horizon(self):
        bundle = bundle_factory()
        result = optimize_budget(bundle, self.settings(horizon_weeks=12))
        self.validate(bundle, result)
        search = result["allocation"].set_index("channel").loc["google_search_spend"]
        self.assertGreaterEqual(search["optimized_weekly_spend"], search["historical_weekly_spend"] - 1e-6)
        self.assertTrue(search["search_protected"])
        self.assertEqual(result["summary"]["reference_weeks"], 12)

    def test_reduced_cap_without_search_protection(self):
        bundle = bundle_factory()
        current = bundle.weekly[bundle.manifest["channels"]].tail(8).sum().sum()
        result = optimize_budget(bundle, self.settings(total_budget_cap=current * 0.65, protect_search=False))
        self.validate(bundle, result)
        self.assertFalse(result["allocation"]["search_protected"].any())
        self.assertNotAlmostEqual(result["summary"]["current_period_budget"], result["summary"]["optimized_period_budget"])

    def test_fixed_channels_and_one_free_channel(self):
        bundle = bundle_factory()
        fixed = optimize_budget(bundle, self.settings(max_spend_change=0))
        self.validate(bundle, fixed)
        np.testing.assert_allclose(fixed["allocation"]["optimized_weekly_spend"], fixed["allocation"]["historical_weekly_spend"])
        one = optimize_budget(bundle, self.settings(max_spend_change=0,
                              channel_change_overrides={"meta_conversion_spend": 0.5}))
        self.validate(bundle, one)
        self.assertIn("Direct monotone", one["summary"]["selected_solution"])

    def test_same_budget_reference_retains_historical_proportions(self):
        bundle = bundle_factory()
        result = optimize_budget(bundle, self.settings(channel_weekly_bounds={"meta_conversion_spend": {"min": 130}},
                                      max_spend_change=1.0, cap_at_historical_max=False))
        table = result["allocation"]
        np.testing.assert_allclose(table["reference_weekly_spend"] / table["reference_weekly_spend"].sum(),
                                   table["historical_weekly_spend"] / table["historical_weekly_spend"].sum())
        self.validate(bundle, result)
        self.assertFalse(result["summary"]["same_budget_reference_feasible"])

    def test_one_week_long_horizon_and_unused_cap(self):
        bundle = bundle_factory()
        for horizon in [1, 20]:
            result = optimize_budget(bundle, self.settings(horizon_weeks=horizon))
            self.validate(bundle, result)
            self.assertEqual(len(result["weekly_plan"]), horizon)
        unlimited = optimize_budget(bundle, self.settings(total_budget_cap=1_000_000))
        self.validate(bundle, unlimited)
        self.assertGreater(unlimited["summary"]["unused_budget_cap"], 0)

    def test_infeasible_cap_and_search_bounds(self):
        bundle = bundle_factory()
        with self.assertRaisesRegex(ValueError, "[Cc]onflict|above the budget cap"):
            optimize_budget(bundle, self.settings(total_budget_cap=1))
        with self.assertRaisesRegex(ValueError, "[Cc]onflict"):
            optimize_budget(bundle, self.settings(channel_weekly_bounds={"google_search_spend": {"max": 1}}))

    def test_zero_reference_stays_zero_and_can_be_reopened(self):
        bundle = bundle_factory(zero_recent=True)
        result = optimize_budget(bundle, self.settings())
        row = result["allocation"].set_index("channel").loc["meta_conversion_spend"]
        self.assertEqual(row["optimized_weekly_spend"], 0)
        reopened = optimize_budget(bundle, self.settings(channel_change_overrides={"meta_conversion_spend": None}))
        self.validate(bundle, reopened)
        self.assertFalse(reopened["summary"]["analytic_gradient"])

    def test_nonpositive_total_predictions_not_clipped(self):
        bundle = bundle_factory()
        bundle.posterior["gamma_control"][:, 1] = -10
        result = optimize_budget(bundle, self.settings())
        self.assertIsNone(result["summary"]["gain_vs_historical_same_budget"]["pct_total"])
        self.assertTrue(result["summary"]["warnings"])
        self.assertLess(result["weekly_forecast"]["optimized_total_orders"].min(), 0)

    def test_future_discount_limits_and_reference_window(self):
        bundle = bundle_factory()
        with self.assertRaisesRegex(ValueError, "reference window"):
            optimize_budget(bundle, self.settings(reference_weeks=100))
        with self.assertRaisesRegex(ValueError, "fraction"):
            optimize_budget(bundle, self.settings(future_discount_rate=10))


if __name__ == "__main__":
    unittest.main()
