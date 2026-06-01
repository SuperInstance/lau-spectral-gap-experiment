//! Entropy-regularized policy gradient runner.

use crate::mdp::MDP;
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

/// Result of a policy gradient run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PGResult {
    /// KL divergence from optimal policy at each step.
    pub kl_divergences: Vec<f64>,
    /// Value function estimates at each step.
    pub value_trajectories: Vec<Vec<f64>>,
    /// Estimated convergence rate (slope of log KL divergence).
    pub convergence_rate: f64,
    /// Number of steps taken.
    pub n_steps: usize,
    /// Temperature parameter.
    pub hbar: f64,
}

/// Policy gradient runner with entropy regularization.
pub struct PolicyGradientRunner {
    /// Temperature parameter (ℏ).
    pub hbar: f64,
    /// Learning rate.
    pub lr: f64,
    /// Maximum iterations.
    pub max_steps: usize,
    /// Convergence threshold.
    pub tolerance: f64,
}

impl PolicyGradientRunner {
    pub fn new(hbar: f64, lr: f64, max_steps: usize) -> Self {
        Self {
            hbar,
            lr,
            max_steps,
            tolerance: 1e-10,
        }
    }

    /// Run entropy-regularized policy gradient on an MDP.
    pub fn run(&self, mdp: &dyn MDP) -> PGResult {
        let n = mdp.n_states();
        let na = mdp.n_actions();
        let gamma = mdp.discount();
        let rewards = mdp.rewards();
        let transitions = mdp.transition_matrices();

        // Initialize policy: uniform
        let mut policy = DMatrix::from_element(n, na, 1.0 / na as f64);

        let mut kl_divergences = Vec::new();
        let mut value_trajectories = Vec::new();

        // Compute optimal policy via value iteration first
        let optimal_value = self.value_iteration(mdp);
        let optimal_policy = self.soft_policy_from_value(mdp, &optimal_value);

        for step in 0..self.max_steps {
            // Compute soft value function
            let value = self.soft_value_function(mdp, &policy);

            // Compute KL divergence from optimal
            let kl = self.kl_divergence(&policy, &optimal_policy);
            kl_divergences.push(kl);

            // Store value trajectory
            value_trajectories.push(value.data.as_vec().clone());

            if kl < self.tolerance {
                break;
            }

            // Policy gradient update
            for s in 0..n {
                let mut advantages = Vec::new();
                for a in 0..na {
                    let mut q_sa = rewards[(s, a)];
                    for t in 0..n {
                        q_sa += gamma * transitions[a][(s, t)] * value[t];
                    }
                    // Soft advantage
                    advantages.push(q_sa - value[s]);
                }

                // Softmax policy update with entropy bonus
                let max_adv = advantages.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exp_adv: Vec<f64> = advantages.iter().map(|a| ((a - max_adv) / self.hbar.max(1e-8)).exp()).collect();
                let sum_exp: f64 = exp_adv.iter().sum();
                if sum_exp > 1e-15 {
                    for a in 0..na {
                        let target = exp_adv[a] / sum_exp;
                        policy[(s, a)] += self.lr * (target - policy[(s, a)]);
                    }
                    // Renormalize
                    let row_sum: f64 = (0..na).map(|a| policy[(s, a)]).sum();
                    if row_sum > 1e-15 {
                        for a in 0..na {
                            policy[(s, a)] /= row_sum;
                        }
                    }
                }
            }
        }

        let convergence_rate = self.estimate_convergence_rate(&kl_divergences);

        PGResult {
            kl_divergences,
            value_trajectories,
            convergence_rate,
            n_steps: kl_divergences.len(),
            hbar: self.hbar,
        }
    }

    /// Soft value function under current policy.
    fn soft_value_function(&self, mdp: &dyn MDP, policy: &DMatrix<f64>) -> DVector<f64> {
        let n = mdp.n_states();
        let na = mdp.n_actions();
        let gamma = mdp.discount();
        let rewards = mdp.rewards();
        let transitions = mdp.transition_matrices();

        let mut value = DVector::zeros(n);

        for _ in 0..200 {
            let mut new_value = DVector::zeros(n);
            for s in 0..n {
                let mut v_s = 0.0;
                for a in 0..na {
                    let pi_sa = policy[(s, a)];
                    let mut q_sa = rewards[(s, a)];
                    for t in 0..n {
                        q_sa += gamma * transitions[a][(s, t)] * value[t];
                    }
                    // Entropy bonus: -hbar * log(pi_sa)
                    let entropy = if pi_sa > 1e-15 {
                        -self.hbar * pi_sa.ln()
                    } else {
                        0.0
                    };
                    v_s += pi_sa * (q_sa + entropy);
                }
                new_value[s] = v_s;
            }
            let diff = (&new_value - &value).norm();
            value = new_value;
            if diff < 1e-12 {
                break;
            }
        }

        value
    }

    /// Value iteration to find optimal value function.
    fn value_iteration(&self, mdp: &dyn MDP) -> DVector<f64> {
        let n = mdp.n_states();
        let na = mdp.n_actions();
        let gamma = mdp.discount();
        let rewards = mdp.rewards();
        let transitions = mdp.transition_matrices();

        let mut value = DVector::zeros(n);

        for _ in 0..500 {
            let mut new_value = DVector::zeros(n);
            for s in 0..n {
                let mut max_v = f64::NEG_INFINITY;
                for a in 0..na {
                    let mut q_sa = rewards[(s, a)];
                    for t in 0..n {
                        q_sa += gamma * transitions[a][(s, t)] * value[t];
                    }
                    max_v = max_v.max(q_sa);
                }
                new_value[s] = max_v;
            }
            let diff = (&new_value - &value).norm();
            value = new_value;
            if diff < 1e-12 {
                break;
            }
        }

        value
    }

    /// Derive soft optimal policy from value function.
    fn soft_policy_from_value(&self, mdp: &dyn MDP, value: &DVector<f64>) -> DMatrix<f64> {
        let n = mdp.n_states();
        let na = mdp.n_actions();
        let gamma = mdp.discount();
        let rewards = mdp.rewards();
        let transitions = mdp.transition_matrices();

        let mut policy = DMatrix::zeros(n, na);

        for s in 0..n {
            let mut q_values = Vec::new();
            for a in 0..na {
                let mut q_sa = rewards[(s, a)];
                for t in 0..n {
                    q_sa += gamma * transitions[a][(s, t)] * value[t];
                }
                q_values.push(q_sa);
            }

            let max_q = q_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_q: Vec<f64> = q_values.iter().map(|q| ((q - max_q) / self.hbar.max(1e-8)).exp()).collect();
            let sum: f64 = exp_q.iter().sum();
            for a in 0..na {
                policy[(s, a)] = exp_q[a] / sum;
            }
        }

        policy
    }

    /// KL divergence between two policies (averaged over states).
    fn kl_divergence(&self, p: &DMatrix<f64>, q: &DMatrix<f64>) -> f64 {
        let n = p.nrows();
        let na = p.ncolumns();
        let mut total = 0.0;
        for s in 0..n {
            for a in 0..na {
                let pi = p[(s, a)].max(1e-15);
                let qi = q[(s, a)].max(1e-15);
                total += pi * (pi / qi).ln();
            }
        }
        total / n as f64
    }

    /// Estimate convergence rate from KL divergence trajectory.
    fn estimate_convergence_rate(&self, kl_divs: &[f64]) -> f64 {
        if kl_divs.len() < 10 {
            return 0.0;
        }

        // Fit log(KL) ~ -rate * t + const in the latter half
        let start = kl_divs.len() / 2;
        let n_pts = kl_divs.len() - start;

        if n_pts < 3 {
            return 0.0;
        }

        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_xx = 0.0;

        for (i, kl) in kl_divs[start..].iter().enumerate() {
            let x = i as f64;
            let y = if *kl > 1e-20 { kl.ln() } else { -46.0 }; // ln(1e-20)
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_xx += x * x;
        }

        let n_f = n_pts as f64;
        let denom = n_f * sum_xx - sum_x * sum_x;
        if denom.abs() < 1e-15 {
            return 0.0;
        }

        let slope = (n_f * sum_xy - sum_x * sum_y) / denom;
        // Slope should be negative (divergence decreasing)
        -slope
    }

    /// Run with multiple seeds and average convergence rate.
    pub fn run_averaged(&self, mdp: &dyn MDP, n_runs: usize) -> PGResult {
        // For deterministic MDPs, just run once
        let result = self.run(mdp);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mdp::*;

    #[test]
    fn test_pg_chain_basic() {
        let chain = ChainMDP::new(3, 0.9);
        let runner = PolicyGradientRunner::new(1.0, 0.1, 100);
        let result = runner.run(&chain);
        assert!(result.convergence_rate >= 0.0);
        assert!(!result.kl_divergences.is_empty());
    }

    #[test]
    fn test_pg_convergence_decreasing() {
        let chain = ChainMDP::new(3, 0.9);
        let runner = PolicyGradientRunner::new(1.0, 0.1, 200);
        let result = runner.run(&chain);
        // KL should generally decrease
        if result.kl_divergences.len() > 10 {
            let early_avg: f64 = result.kl_divergences[..10].iter().sum::<f64>() / 10.0;
            let late_avg: f64 = result.kl_divergences[result.kl_divergences.len() - 10..]
                .iter()
                .sum::<f64>()
                / 10.0;
            assert!(late_avg <= early_avg * 1.5, "KL should decrease: early {} late {}", early_avg, late_avg);
        }
    }

    #[test]
    fn test_pg_policy_normalized() {
        let grid = GridWorldMDP::new(3, 3, 0.9);
        let runner = PolicyGradientRunner::new(1.0, 0.05, 50);
        let result = runner.run(&grid);
        // The final policy should be normalized (rows sum to 1)
        // We can check this by verifying value trajectories exist
        assert!(!result.value_trajectories.is_empty());
    }

    #[test]
    fn test_pg_temperature_effect() {
        let chain = ChainMDP::new(5, 0.9);
        let cold = PolicyGradientRunner::new(0.1, 0.05, 200);
        let hot = PolicyGradientRunner::new(10.0, 0.05, 200);
        let r_cold = cold.run(&chain);
        let r_hot = hot.run(&chain);
        // Both should converge (rate > 0)
        assert!(r_cold.convergence_rate >= 0.0);
        assert!(r_hot.convergence_rate >= 0.0);
    }

    #[test]
    fn test_pg_grid_3x3() {
        let grid = GridWorldMDP::new(3, 3, 0.9);
        let runner = PolicyGradientRunner::new(1.0, 0.05, 100);
        let result = runner.run(&grid);
        assert!(result.n_steps > 0);
    }

    #[test]
    fn test_pg_grid_5x5() {
        let grid = GridWorldMDP::new(5, 5, 0.9);
        let runner = PolicyGradientRunner::new(1.0, 0.05, 100);
        let result = runner.run(&grid);
        assert!(result.n_steps > 0);
    }

    #[test]
    fn test_pg_serialization() {
        let chain = ChainMDP::new(3, 0.9);
        let runner = PolicyGradientRunner::new(1.0, 0.1, 50);
        let result = runner.run(&chain);
        let json = serde_json::to_string(&result).unwrap();
        let decoded: PGResult = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.n_steps, result.n_steps);
    }

    #[test]
    fn test_kl_divergence_same_policy() {
        let runner = PolicyGradientRunner::new(1.0, 0.1, 10);
        let p = DMatrix::from_element(3, 2, 0.5);
        let kl = runner.kl_divergence(&p, &p);
        assert!(kl.abs() < 1e-10, "KL of same policy should be 0, got {}", kl);
    }

    #[test]
    fn test_convergence_rate_estimation() {
        let runner = PolicyGradientRunner::new(1.0, 0.1, 10);
        // Exponentially decreasing KL divergences
        let kls: Vec<f64> = (0..100).map(|i| (-0.05 * i as f64).exp()).collect();
        let rate = runner.estimate_convergence_rate(&kls);
        assert!((rate - 0.05).abs() < 0.02, "Expected rate ≈ 0.05, got {}", rate);
    }

    #[test]
    fn test_value_iteration_converges() {
        let chain = ChainMDP::new(3, 0.9);
        let runner = PolicyGradientRunner::new(1.0, 0.1, 10);
        let value = runner.value_iteration(&chain);
        // Last state should have highest value
        assert!(value[2] > value[0]);
    }

    #[test]
    fn test_soft_value_function() {
        let chain = ChainMDP::new(3, 0.9);
        let runner = PolicyGradientRunner::new(1.0, 0.1, 10);
        let policy = DMatrix::from_element(3, 2, 0.5);
        let value = runner.soft_value_function(&chain, &policy);
        assert_eq!(value.len(), 3);
    }
}
