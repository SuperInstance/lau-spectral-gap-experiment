//! Observation Laplacian construction from MDP state space.

use crate::mdp::MDP;
use nalgebra::DMatrix;

/// Observation Laplacian computed from MDP adjacency structure.
#[derive(Debug, Clone)]
pub struct ObservationLaplacian {
    /// The Laplacian matrix L = D - A.
    pub laplacian: DMatrix<f64>,
    /// Degree matrix D.
    pub degree: DMatrix<f64>,
    /// Adjacency matrix A.
    pub adjacency: DMatrix<f64>,
    /// Number of states.
    pub n: usize,
}

impl ObservationLaplacian {
    /// Construct from an MDP using its adjacency structure.
    pub fn from_mdp(mdp: &dyn MDP) -> Self {
        let n = mdp.n_states();
        let adj_list = mdp.adjacency();

        let mut a = DMatrix::zeros(n, n);
        let mut d = DMatrix::zeros(n, n);

        for (s, neighbors) in adj_list.iter().enumerate() {
            let degree = neighbors.len() as f64;
            d[(s, s)] = degree;
            for &t in neighbors {
                a[(s, t)] = 1.0;
            }
        }

        let laplacian = &d - &a;

        Self {
            laplacian,
            degree: d,
            adjacency: a,
            n,
        }
    }

    /// Construct from explicit adjacency list.
    pub fn from_adjacency(n: usize, adj: &[Vec<usize>]) -> Self {
        let mut a = DMatrix::zeros(n, n);
        let mut d = DMatrix::zeros(n, n);

        for (s, neighbors) in adj.iter().enumerate() {
            let degree = neighbors.len() as f64;
            d[(s, s)] = degree;
            for &t in neighbors {
                a[(s, t)] = 1.0;
            }
        }

        let laplacian = &d - &a;

        Self {
            laplacian,
            degree: d,
            adjacency: a,
            n,
        }
    }

    /// Construct normalized Laplacian: L_norm = I - D^{-1/2} A D^{-1/2}.
    pub fn normalized(&self) -> DMatrix<f64> {
        let n = self.n;
        let mut d_inv_sqrt = DMatrix::zeros(n, n);
        for i in 0..n {
            if self.degree[(i, i)] > 0.0 {
                d_inv_sqrt[(i, i)] = 1.0 / self.degree[(i, i)].sqrt();
            }
        }
        let identity = DMatrix::identity(n, n);
        &identity - &d_inv_sqrt * &self.adjacency * &d_inv_sqrt
    }

    /// Weighted Laplacian with temperature parameter.
    /// Higher temperature smooths the graph, reducing the spectral gap.
    pub fn with_temperature(&self, hbar: f64) -> DMatrix<f64> {
        // L_hbar = L / (1 + hbar)
        // This models thermal noise: higher hbar → more mixing → smaller effective gap
        &self.laplacian / (1.0 + hbar)
    }

    /// Verify Laplacian properties: symmetric, rows sum to zero, positive semi-definite.
    pub fn verify(&self) -> LaplacianProperties {
        let n = self.n;
        let mut symmetric = true;
        let mut row_sum_zero = true;
        let mut diag_nonneg = true;

        for i in 0..n {
            let row_sum: f64 = (0..n).map(|j| self.laplacian[(i, j)]).sum();
            if row_sum.abs() > 1e-10 {
                row_sum_zero = false;
            }
            if self.laplacian[(i, i)] < -1e-10 {
                diag_nonneg = false;
            }
            for j in 0..n {
                if (self.laplacian[(i, j)] - self.laplacian[(j, i)).abs() > 1e-10 {
                    symmetric = false;
                }
            }
        }

        LaplacianProperties {
            symmetric,
            row_sum_zero,
            diag_nonneg,
            n_states: n,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LaplacianProperties {
    pub symmetric: bool,
    pub row_sum_zero: bool,
    pub diag_nonneg: bool,
    pub n_states: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mdp::*;

    #[test]
    fn test_laplacian_from_chain() {
        let chain = ChainMDP::new(4, 0.99);
        let lap = ObservationLaplacian::from_mdp(&chain);
        assert_eq!(lap.n, 4);
        // Chain 0-1-2-3: L should be:
        // [ 1 -1  0  0]
        // [-1  2 -1  0]
        // [ 0 -1  2 -1]
        // [ 0  0 -1  1]
        assert!((lap.laplacian[(0, 0)] - 1.0).abs() < 1e-10);
        assert!((lap.laplacian[(1, 1)] - 2.0).abs() < 1e-10);
        assert!((lap.laplacian[(0, 1)] - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_laplacian_row_sums_zero() {
        let grid = GridWorldMDP::new(3, 3, 0.99);
        let lap = ObservationLaplacian::from_mdp(&grid);
        for i in 0..lap.n {
            let row_sum: f64 = (0..lap.n).map(|j| lap.laplacian[(i, j)]).sum();
            assert!(row_sum.abs() < 1e-10, "Row {} sum: {}", i, row_sum);
        }
    }

    #[test]
    fn test_laplacian_symmetric() {
        let grid = GridWorldMDP::new(5, 5, 0.99);
        let lap = ObservationLaplacian::from_mdp(&grid);
        for i in 0..lap.n {
            for j in 0..lap.n {
                assert!(
                    (lap.laplacian[(i, j)] - lap.laplacian[(j, i)]).abs() < 1e-10,
                    "Asymmetric at ({}, {})", i, j
                );
            }
        }
    }

    #[test]
    fn test_laplacian_verify() {
        let chain = ChainMDP::new(5, 0.99);
        let lap = ObservationLaplacian::from_mdp(&chain);
        let props = lap.verify();
        assert!(props.symmetric);
        assert!(props.row_sum_zero);
        assert!(props.diag_nonneg);
    }

    #[test]
    fn test_normalized_laplacian() {
        let chain = ChainMDP::new(4, 0.99);
        let lap = ObservationLaplacian::from_mdp(&chain);
        let norm = lap.normalized();
        // Diagonal entries of normalized Laplacian should be 1 for connected nodes
        assert!((norm[(1, 1)] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_temperature_laplacian() {
        let chain = ChainMDP::new(4, 0.99);
        let lap = ObservationLaplacian::from_mdp(&chain);
        let cold = lap.with_temperature(0.1);
        let hot = lap.with_temperature(10.0);
        // Hotter temperature → smaller effective Laplacian
        assert!(cold.norm() > hot.norm());
    }

    #[test]
    fn test_laplacian_from_grid_3x3() {
        let grid = GridWorldMDP::new(3, 3, 0.99);
        let lap = ObservationLaplacian::from_mdp(&grid);
        // Corner has degree 2
        assert!((lap.degree[(0, 0)] - 2.0).abs() < 1e-10);
        // Center has degree 4
        assert!((lap.degree[(4, 4)] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_laplacian_from_adjacency() {
        let adj = vec![vec![1], vec![0, 2], vec![1]];
        let lap = ObservationLaplacian::from_adjacency(3, &adj);
        assert_eq!(lap.n, 3);
        assert!((lap.laplacian[(0, 0)] - 1.0).abs() < 1e-10);
        assert!((lap.laplacian[(1, 1)] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_laplacian_degree_matches_adjacency() {
        let grid = GridWorldMDP::new(5, 5, 0.99);
        let lap = ObservationLaplacian::from_mdp(&grid);
        for i in 0..lap.n {
            let adj_count = (0..lap.n).filter(|&j| lap.adjacency[(i, j)] > 0.5).count();
            assert!((lap.degree[(i, i)] - adj_count as f64).abs() < 1e-10);
        }
    }

    #[test]
    fn test_laplacian_grid_10x10() {
        let grid = GridWorldMDP::new(10, 10, 0.99);
        let lap = ObservationLaplacian::from_mdp(&grid);
        assert_eq!(lap.n, 100);
        let props = lap.verify();
        assert!(props.symmetric);
        assert!(props.row_sum_zero);
    }
}
