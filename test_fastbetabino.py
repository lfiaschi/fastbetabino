"""
Comprehensive test suite for fastbetabino library.

Tests include:
- Parameter estimation correctness
- Convergence behavior
- Posterior distribution functions
- Edge cases and input validation
"""

import unittest
import numpy as np
from array import array
import sys


class TestFitAlphaBeta(unittest.TestCase):
    """Test the fit_alpha_beta function for parameter estimation."""

    def setUp(self):
        """Set up test fixtures before each test."""
        # Import here to allow testing after compilation
        try:
            from fastbetabino import fit_alpha_beta, posterior_mean, \
                posterior_interval_bounds, posterior_cdf, posterior_sf
            self.fit_alpha_beta = fit_alpha_beta
            self.posterior_mean = posterior_mean
            self.posterior_interval_bounds = posterior_interval_bounds
            self.posterior_cdf = posterior_cdf
            self.posterior_sf = posterior_sf
        except ImportError:
            self.skipTest("fastbetabino module not compiled. Run 'python setup.py build_ext --inplace' first.")

    def test_basic_fit(self):
        """Test basic parameter fitting with simple data."""
        # Simple test case: uniform conversion rate around 0.1
        np.random.seed(42)
        imps = [100] * 100
        clicks = [10] * 100  # 10% conversion rate

        alpha, beta = self.fit_alpha_beta(imps, clicks)

        # Check that parameters are positive
        self.assertGreater(alpha, 0, "Alpha should be positive")
        self.assertGreater(beta, 0, "Beta should be positive")

        # Check that the mean of the fitted distribution is reasonable
        # Expected: around 0.1 (10/100)
        fitted_mean = alpha / (alpha + beta)
        self.assertAlmostEqual(fitted_mean, 0.1, delta=0.05,
                              msg="Fitted mean should be close to 0.1")

    def test_convergence_with_varied_data(self):
        """Test convergence with more realistic varied data."""
        np.random.seed(123)
        # Generate data from a beta-binomial with known parameters
        true_alpha = 2.0
        true_beta = 8.0
        n_samples = 1000

        # Generate impressions
        imps = np.random.randint(50, 200, n_samples)

        # Generate clicks from beta-binomial
        clicks = []
        for imp in imps:
            # Sample rate from beta distribution
            rate = np.random.beta(true_alpha, true_beta)
            # Sample clicks from binomial
            click = np.random.binomial(imp, rate)
            clicks.append(click)

        alpha, beta = self.fit_alpha_beta(imps, clicks, niter=10000)

        # Check that fitted parameters are close to true parameters
        self.assertAlmostEqual(alpha, true_alpha, delta=1.0,
                              msg=f"Fitted alpha ({alpha:.2f}) should be close to true alpha ({true_alpha})")
        self.assertAlmostEqual(beta, true_beta, delta=1.0,
                              msg=f"Fitted beta ({beta:.2f}) should be close to true beta ({true_beta})")

    def test_high_conversion_rate(self):
        """Test with high conversion rate data."""
        np.random.seed(42)
        imps = [100] * 50
        clicks = [90] * 50  # 90% conversion rate

        alpha, beta = self.fit_alpha_beta(imps, clicks)

        fitted_mean = alpha / (alpha + beta)
        self.assertGreater(fitted_mean, 0.8,
                          "Fitted mean should reflect high conversion rate")

    def test_low_conversion_rate(self):
        """Test with low conversion rate data."""
        np.random.seed(42)
        imps = [100] * 50
        clicks = [2] * 50  # 2% conversion rate

        alpha, beta = self.fit_alpha_beta(imps, clicks)

        fitted_mean = alpha / (alpha + beta)
        self.assertLess(fitted_mean, 0.05,
                       "Fitted mean should reflect low conversion rate")

    def test_convergence_early_stopping(self):
        """Test that convergence criterion works correctly."""
        np.random.seed(42)
        # Use very homogeneous data to ensure quick convergence
        imps = [100] * 100
        clicks = [10] * 100

        # With the fixed convergence check, this should converge early
        alpha, beta = self.fit_alpha_beta(imps, clicks, niter=10000)

        # Just verify it returns valid results
        self.assertIsNotNone(alpha)
        self.assertIsNotNone(beta)
        self.assertGreater(alpha, 0)
        self.assertGreater(beta, 0)

    def test_with_python_lists(self):
        """Test that function works with Python lists."""
        imps = [100, 200, 150, 80, 120]
        clicks = [10, 25, 18, 9, 15]

        alpha, beta = self.fit_alpha_beta(imps, clicks)

        self.assertGreater(alpha, 0)
        self.assertGreater(beta, 0)

    def test_with_numpy_arrays(self):
        """Test that function works with numpy arrays."""
        imps = np.array([100, 200, 150, 80, 120])
        clicks = np.array([10, 25, 18, 9, 15])

        alpha, beta = self.fit_alpha_beta(imps, clicks)

        self.assertGreater(alpha, 0)
        self.assertGreater(beta, 0)

    def test_custom_initial_parameters(self):
        """Test with custom initial alpha and beta values."""
        np.random.seed(42)
        imps = [100] * 50
        clicks = [10] * 50

        alpha1, beta1 = self.fit_alpha_beta(imps, clicks, alpha0=1.0, beta0=1.0)
        alpha2, beta2 = self.fit_alpha_beta(imps, clicks, alpha0=5.0, beta0=15.0)

        # Results should be similar regardless of initial values (convergence property)
        self.assertAlmostEqual(alpha1, alpha2, delta=0.5,
                              msg="Different initial values should converge to similar alpha")
        self.assertAlmostEqual(beta1, beta2, delta=0.5,
                              msg="Different initial values should converge to similar beta")


class TestPosteriorFunctions(unittest.TestCase):
    """Test posterior distribution analysis functions."""

    def setUp(self):
        """Set up test fixtures."""
        try:
            from fastbetabino import fit_alpha_beta, posterior_mean, \
                posterior_interval_bounds, posterior_cdf, posterior_sf
            self.fit_alpha_beta = fit_alpha_beta
            self.posterior_mean = posterior_mean
            self.posterior_interval_bounds = posterior_interval_bounds
            self.posterior_cdf = posterior_cdf
            self.posterior_sf = posterior_sf
        except ImportError:
            self.skipTest("fastbetabino module not compiled.")

    def test_posterior_mean(self):
        """Test posterior mean calculation."""
        # Simple case: 10 clicks out of 100 impressions
        # With alpha=2, beta=8 prior
        imps = 100
        clicks = 10
        alpha_hat = 2.0
        beta_hat = 8.0

        mean = self.posterior_mean(imps, clicks, alpha_hat, beta_hat)

        # Posterior mean = (clicks + alpha) / (imps + alpha + beta)
        expected_mean = (10 + 2.0) / (100 + 2.0 + 8.0)
        self.assertAlmostEqual(mean, expected_mean, places=5)

    def test_posterior_interval_bounds(self):
        """Test posterior confidence interval calculation."""
        imps = 100
        clicks = 10
        alpha_hat = 2.0
        beta_hat = 8.0

        # Get 95% confidence interval
        lower, upper = self.posterior_interval_bounds(0.95, imps, clicks, alpha_hat, beta_hat)

        # Bounds should be between 0 and 1
        self.assertGreaterEqual(lower, 0.0)
        self.assertLessEqual(upper, 1.0)

        # Lower bound should be less than upper bound
        self.assertLess(lower, upper)

        # Mean should be between bounds
        mean = self.posterior_mean(imps, clicks, alpha_hat, beta_hat)
        self.assertGreater(mean, lower)
        self.assertLess(mean, upper)

    def test_posterior_cdf(self):
        """Test posterior cumulative distribution function."""
        imps = 100
        clicks = 10
        alpha_hat = 2.0
        beta_hat = 8.0

        # Test at several points
        cdf_0 = self.posterior_cdf(0.0, imps, clicks, alpha_hat, beta_hat)
        cdf_05 = self.posterior_cdf(0.5, imps, clicks, alpha_hat, beta_hat)
        cdf_1 = self.posterior_cdf(1.0, imps, clicks, alpha_hat, beta_hat)

        # CDF should be monotonically increasing
        self.assertLess(cdf_0, cdf_05)
        self.assertLess(cdf_05, cdf_1)

        # CDF at 0 should be ~0, at 1 should be ~1
        self.assertAlmostEqual(cdf_0, 0.0, delta=0.01)
        self.assertAlmostEqual(cdf_1, 1.0, delta=0.01)

    def test_posterior_sf(self):
        """Test posterior survival function."""
        imps = 100
        clicks = 10
        alpha_hat = 2.0
        beta_hat = 8.0

        # SF should be complementary to CDF
        x = 0.15
        cdf_x = self.posterior_cdf(x, imps, clicks, alpha_hat, beta_hat)
        sf_x = self.posterior_sf(x, imps, clicks, alpha_hat, beta_hat)

        # CDF + SF should approximately equal 1
        self.assertAlmostEqual(cdf_x + sf_x, 1.0, places=5,
                              msg="CDF + SF should equal 1")

    def test_posterior_functions_integration(self):
        """Test integration of fit and posterior functions."""
        np.random.seed(42)
        imps = np.array([100] * 50)
        clicks = np.array([10] * 50)

        # Fit parameters
        alpha, beta = self.fit_alpha_beta(imps, clicks)

        # Use fitted parameters for posterior analysis on a new observation
        new_imps = 100
        new_clicks = 12

        mean = self.posterior_mean(new_imps, new_clicks, alpha, beta)
        lower, upper = self.posterior_interval_bounds(0.90, new_imps, new_clicks, alpha, beta)

        # Sanity checks
        self.assertGreater(mean, 0)
        self.assertLess(mean, 1)
        self.assertGreater(mean, lower)
        self.assertLess(mean, upper)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""

    def setUp(self):
        """Set up test fixtures."""
        try:
            from fastbetabino import fit_alpha_beta
            self.fit_alpha_beta = fit_alpha_beta
        except ImportError:
            self.skipTest("fastbetabino module not compiled.")

    def test_single_observation(self):
        """Test with single observation."""
        imps = [100]
        clicks = [10]

        alpha, beta = self.fit_alpha_beta(imps, clicks)

        self.assertGreater(alpha, 0)
        self.assertGreater(beta, 0)

    def test_zero_clicks(self):
        """Test with zero clicks."""
        imps = [100, 200, 150]
        clicks = [0, 0, 0]

        alpha, beta = self.fit_alpha_beta(imps, clicks)

        # Should still return valid parameters
        self.assertGreater(alpha, 0)
        self.assertGreater(beta, 0)

        # Beta should be much larger than alpha (low conversion)
        self.assertGreater(beta, alpha)

    def test_all_conversions(self):
        """Test with 100% conversion rate."""
        imps = [100, 200, 150]
        clicks = [100, 200, 150]  # All impressions converted

        alpha, beta = self.fit_alpha_beta(imps, clicks)

        # Should still return valid parameters
        self.assertGreater(alpha, 0)
        self.assertGreater(beta, 0)

        # Alpha should be much larger than beta (high conversion)
        self.assertGreater(alpha, beta)

    def test_large_dataset(self):
        """Test with large dataset for performance."""
        np.random.seed(42)
        n = 10000
        imps = np.random.randint(10, 1000, n)
        clicks = [np.random.randint(0, int(i * 0.15)) for i in imps]

        # This should complete without error
        alpha, beta = self.fit_alpha_beta(imps, clicks, niter=1000)

        self.assertGreater(alpha, 0)
        self.assertGreater(beta, 0)


def run_tests():
    """Run all tests and return results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestFitAlphaBeta))
    suite.addTests(loader.loadTestsFromTestCase(TestPosteriorFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == '__main__':
    print("=" * 70)
    print("FastBetaBino Test Suite")
    print("=" * 70)
    print("\nNote: Make sure to compile the module first:")
    print("  python setup.py build_ext --inplace\n")
    print("=" * 70 + "\n")

    result = run_tests()

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")

    if result.wasSuccessful():
        print("\nAll tests passed!")
        sys.exit(0)
    else:
        print("\nSome tests failed.")
        sys.exit(1)
