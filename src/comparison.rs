//! Rate comparison: spectral gap vs convergence rate.

use crate::laplacian::ObservationLaplacian;
use crate::mdp::MDP;
use crate::policy_gradient::PolicyGradientRunner;
use crate::spectral::SpectralGap;
use serde::{Deserialize, Serialize};

/// Result of comparing spectral gap and convergence rate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateComparison {
    /// MDP name.
    pub mdp_name: String,
    /// MDP size.
    pub mdp_size: usize,
    /// Temperature.
    pub hbar: f64,
    /// Spectral gap.
    pub spectral_gap: f64,
    /// Convergence rate.
    pub convergence_rate: f64,
    /// Relative difference.
    pub relative_diff: f64,
    /// Whether rates match within tolerance.
    pub matches: bool,
    /// Tolerance used.
    pub tolerance: f64,
}

impl RateComparison {
    /// Compare spectral gap and convergence rate for a given MDP.
    pub fn compare(mdp: &dyn MDP, hbar: f64, lr: f64, pg_steps: usize, tolerance: f64) -> Self {
        let lap = ObservationLaplacian::from_mdp(mdp);
        let sg = SpectralGap::compute_with_temperature(&lap, hbar);
        let runner = PolicyGradientRunner::new(hbar, lr, pg_steps);
        let pg_result = runner.run(mdp);

        let spectral_gap = sg.gap;
        let convergence_rate = pg_result.convergence_rate;

        let relative_diff = if spectral_gap.abs() > 1e-15 {
            (convergence_rate - spectral_gap).abs() / spectral_gap
        } else if convergence_rate.abs() > 1e-15 {
            (convergence_rate - spectral_gap).abs() / convergence_rate
        } else {
            0.0
        };

        let matches = relative_diff < tolerance;

        Self {
            mdp_name: mdp.name().to_string(),
            mdp_size: mdp.n_states(),
            hbar,
            spectral_gap,
            convergence_rate,
            relative_diff,
            matches,
            tolerance,
        }
    }

    /// Run comparison across multiple temperatures.
    pub fn sweep_temperature(mdp: &dyn MDP, temperatures: &[f64], lr: f64, pg_steps: usize, tolerance: f64) -> Vec<Self> {
        temperatures
            .iter()
            .map(|&hbar| Self::compare(mdp, hbar, lr, pg_steps, tolerance))
            .collect()
    }
}

/// Summary of all comparisons.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentSummary {
    pub comparisons: Vec<RateComparison>,
    pub n_total: usize,
    pub n_match: usize,
    pub match_fraction: f64,
    pub mean_relative_diff: f64,
}

impl ExperimentSummary {
    pub fn from_comparisons(comparisons: Vec<RateComparison>) -> Self {
        let n_total = comparisons.len();
        let n_match = comparisons.iter().filter(|c| c.matches).count();
        let match_fraction = if n_total > 0 {
            n_match as f64 / n_total as f64
        } else {
            0.0
        };
        let mean_relative_diff = if n_total > 0 {
            comparisons.iter().map(|c| c.relative_diff).sum::<f64>() / n_total as f64
        } else {
            0.0
        };

        Self {
            comparisons,
            n_total,
            n_match,
            match_fraction,
            mean_relative_diff,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mdp::*;

    #[test]
    fn test_compare_chain_3() {
        let chain = ChainMDP::new(3, 0.9);
        let comp = RateComparison::compare(&chain, 1.0, 0.1, 200, 1.0);
        assert_eq!(comp.mdp_name, "chain");
        assert!(comp.spectral_gap > 0.0);
        assert!(comp.convergence_rate >= 0.0);
    }

    #[test]
    fn test_compare_grid_3x3() {
        let grid = GridWorldMDP::new(3, 3, 0.9);
        let comp = RateComparison::compare(&grid, 1.0, 0.05, 200, 1.0);
        assert_eq!(comp.mdp_name, "grid_world");
        assert!(comp.spectral_gap > 0.0);
    }

    #[test]
    fn test_compare_grid_5x5() {
        let grid = GridWorldMDP::new(5, 5, 0.9);
        let comp = RateComparison::compare(&grid, 1.0, 0.05, 200, 1.0);
        assert!(comp.spectral_gap > 0.0);
    }

    #[test]
    fn test_compare_chain_5() {
        let chain = ChainMDP::new(5, 0.9);
        let comp = RateComparison::compare(&chain, 1.0, 0.1, 200, 1.0);
        assert!(comp.spectral_gap > 0.0);
    }

    #[test]
    fn test_relative_diff_bounded() {
        let chain = ChainMDP::new(3, 0.9);
        let comp = RateComparison::compare(&chain, 1.0, 0.1, 200, 1.0);
        assert!(comp.relative_diff >= 0.0);
    }

    #[test]
    fn test_temperature_sweep() {
        let chain = ChainMDP::new(3, 0.9);
        let temps = vec![0.1, 0.5, 1.0, 2.0, 5.0];
        let results = RateComparison::sweep_temperature(&chain, &temps, 0.1, 200, 1.0);
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_summary() {
        let comparisons = vec![
            RateComparison {
                mdp_name: "chain".into(),
                mdp_size: 3,
                hbar: 1.0,
                spectral_gap: 1.0,
                convergence_rate: 0.95,
                relative_diff: 0.05,
                matches: true,
                tolerance: 0.1,
            },
            RateComparison {
                mdp_name: "grid".into(),
                mdp_size: 9,
                hbar: 1.0,
                spectral_gap: 0.5,
                convergence_rate: 0.2,
                relative_diff: 0.6,
                matches: false,
                tolerance: 0.1,
            },
        ];
        let summary = ExperimentSummary::from_comparisons(comparisons);
        assert_eq!(summary.n_total, 2);
        assert_eq!(summary.n_match, 1);
        assert!((summary.match_fraction - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_serialization() {
        let chain = ChainMDP::new(3, 0.9);
        let comp = RateComparison::compare(&chain, 1.0, 0.1, 100, 1.0);
        let json = serde_json::to_string(&comp).unwrap();
        let decoded: RateComparison = serde_json::from_str(&json).unwrap();
        assert!((decoded.spectral_gap - comp.spectral_gap).abs() < 1e-10);
    }

    #[test]
    fn test_random_mdp_comparison() {
        let rmdp = RandomMDP::new(5, 2, 0.9, 42);
        let comp = RateComparison::compare(&rmdp, 1.0, 0.05, 200, 1.0);
        assert_eq!(comp.mdp_name, "random");
    }
}
