//! # lau-spectral-gap-experiment
//!
//! Experimental verification that the spectral gap of the observation-Laplacian
//! (computed by kalman-hodge) equals the convergence rate of entropy-regularized
//! policy gradient (computed by thermal-rl).
//!
//! If the rates match, Opus's Emergent Theorem A is experimentally confirmed.

pub mod mdp;
pub mod laplacian;
pub mod spectral;
pub mod policy_gradient;
pub mod comparison;
pub mod varadhan;
pub mod temperature_sweep;
pub mod statistics;

pub use mdp::{GridWorldMDP, ChainMDP, RandomMDP, MDP};
pub use laplacian::ObservationLaplacian;
pub use spectral::SpectralGap;
pub use policy_gradient::PolicyGradientRunner;
pub use comparison::RateComparison;
pub use varadhan::VaradhanVerifier;
pub use temperature_sweep::TemperatureSweep;
pub use statistics::StatisticalTest;
