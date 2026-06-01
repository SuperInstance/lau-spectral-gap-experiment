//! Spectral gap computation: smallest nonzero eigenvalue of Laplacian.

use crate::laplacian::ObservationLaplacian;
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

/// Spectral gap result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralGap {
    /// All eigenvalues sorted ascending.
    pub eigenvalues: Vec<f64>,
    /// Spectral gap (smallest nonzero eigenvalue).
    pub gap: f64,
    /// Algebraic connectivity (same as gap for connected graphs).
    pub algebraic_connectivity: f64,
    /// Number of zero eigenvalues (= connected components).
    pub n_components: usize,
    /// The Fiedler vector (eigenvector for the gap eigenvalue).
    pub fiedler_vector: Vec<f64>,
}

impl SpectralGap {
    /// Compute spectral gap from an ObservationLaplacian.
    pub fn compute(laplacian: &ObservationLaplacian) -> Self {
        let n = laplacian.n;
        let eigenvalues = Self::eigenvalues_symmetric(&laplacian.laplacian);
        let (gap, fiedler_idx) = Self::find_spectral_gap(&eigenvalues);

        let fiedler_vector = if fiedler_idx < n {
            Self::eigenvector_symmetric(&laplacian.laplacian, fiedler_idx)
        } else {
            vec![0.0; n]
        };

        let n_components = eigenvalues.iter().filter(|&&e| e.abs() < 1e-8).count();

        Self {
            eigenvalues,
            gap,
            algebraic_connectivity: gap,
            n_components,
            fiedler_vector,
        }
    }

    /// Compute spectral gap with temperature scaling.
    pub fn compute_with_temperature(laplacian: &ObservationLaplacian, hbar: f64) -> Self {
        let scaled = laplacian.with_temperature(hbar);
        let n = laplacian.n;
        let eigenvalues = Self::eigenvalues_symmetric(&scaled);
        let (gap, fiedler_idx) = Self::find_spectral_gap(&eigenvalues);

        let fiedler_vector = if fiedler_idx < n {
            Self::eigenvector_symmetric(&scaled, fiedler_idx)
        } else {
            vec![0.0; n]
        };

        let n_components = eigenvalues.iter().filter(|&&e| e.abs() < 1e-8).count();

        Self {
            eigenvalues,
            gap,
            algebraic_connectivity: gap,
            n_components,
            fiedler_vector,
        }
    }

    /// Simple eigenvalue computation for symmetric matrices using QR iteration.
    fn eigenvalues_symmetric(mat: &DMatrix<f64>) -> Vec<f64> {
        let n = mat.nrows();
        if n == 0 {
            return vec![];
        }
        if n == 1 {
            return vec![mat[(0, 0)]];
        }

        // Use power iteration + deflation for robustness
        let mut eigenvalues = Vec::new();
        let mut current = mat.clone();

        for _ in 0..n {
            let (eigval, _) = Self::power_iteration(&current, 200);
            eigenvalues.push(eigval);
            // Deflate
            let (eigvec, _) = Self::power_iteration_vec(&current, 200);
            let norm_sq: f64 = eigvec.iter().map(|x| x * x).sum();
            if norm_sq > 1e-15 {
                let v = DVector::from_vec(eigvec);
                let rank_one = &v * (&v.transpose() * &current) / norm_sq;
                current = &current - rank_one;
            }
        }

        eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap());
        eigenvalues
    }

    /// Power iteration for largest magnitude eigenvalue.
    fn power_iteration(mat: &DMatrix<f64>, max_iter: usize) -> (f64, usize) {
        let n = mat.nrows();
        let mut v = DVector::from_element(n, 1.0 / (n as f64).sqrt());

        for i in 0..max_iter {
            let Av = mat * &v;
            let norm = Av.norm();
            if norm < 1e-15 {
                return (0.0, i);
            }
            v = Av / norm;
        }

        let eigenvalue = v.dot(&(mat * &v));
        (eigenvalue, max_iter)
    }

    /// Power iteration returning eigenvector.
    fn power_iteration_vec(mat: &DMatrix<f64>, max_iter: usize) -> (Vec<f64>, f64) {
        let n = mat.nrows();
        let mut v = DVector::from_element(n, 1.0 / (n as f64).sqrt());

        for _ in 0..max_iter {
            let Av = mat * &v;
            let norm = Av.norm();
            if norm < 1e-15 {
                return (vec![0.0; n], 0.0);
            }
            v = Av / norm;
        }

        let eigenvalue = v.dot(&(mat * &v));
        (v.data.into(), eigenvalue)
    }

    /// Get eigenvector for a specific eigenvalue using inverse iteration.
    fn eigenvector_symmetric(mat: &DMatrix<f64>, _idx: usize) -> Vec<f64> {
        let n = mat.nrows();
        // Use the full eigenvalue decomposition to get all eigenvectors
        let (_, vecs) = Self::full_eigen(mat);
        // Return the eigenvector corresponding to idx
        if vecs.is_empty() {
            return vec![0.0; n];
        }
        let idx = _idx.min(vecs.len() - 1);
        vecs[idx].clone()
    }

    /// Full eigenvalue decomposition using Jacobi iteration.
    fn full_eigen(mat: &DMatrix<f64>) -> (Vec<f64>, Vec<Vec<f64>>) {
        let n = mat.nrows();
        let mut a = mat.clone();
        let mut v = DMatrix::identity(n, n);

        let max_sweeps = 50 * n;
        for _ in 0..max_sweeps {
            // Find largest off-diagonal element
            let mut max_val = 0.0;
            let mut p = 0;
            let mut q = 1;
            for i in 0..n {
                for j in (i + 1)..n {
                    if a[(i, j)].abs() > max_val {
                        max_val = a[(i, j)].abs();
                        p = i;
                        q = j;
                    }
                }
            }
            if max_val < 1e-12 {
                break;
            }

            // Jacobi rotation
            let app = a[(p, p)];
            let aqq = a[(q, q)];
            let apq = a[(p, q)];

            let theta = if (app - aqq).abs() < 1e-15 {
                std::f64::consts::FRAC_PI_4
            } else {
                0.5 * (2.0 * apq / (app - aqq)).atan()
            };

            let c = theta.cos();
            let s = theta.sin();

            // Update A
            for i in 0..n {
                if i != p && i != q {
                    let aip = a[(i, p)];
                    let aiq = a[(i, q)];
                    a[(i, p)] = c * aip + s * aiq;
                    a[(p, i)] = a[(i, p)];
                    a[(i, q)] = -s * aip + c * aiq;
                    a[(q, i)] = a[(i, q)];
                }
            }
            a[(p, p)] = c * c * app + 2.0 * s * c * apq + s * s * aqq;
            a[(q, q)] = s * s * app - 2.0 * s * c * apq + c * c * aqq;
            a[(p, q)] = 0.0;
            a[(q, p)] = 0.0;

            // Update V
            for i in 0..n {
                let vip = v[(i, p)];
                let viq = v[(i, q)];
                v[(i, p)] = c * vip + s * viq;
                v[(i, q)] = -s * vip + c * viq;
            }
        }

        let mut eigenvalues: Vec<f64> = (0..n).map(|i| a[(i, i)]).collect();
        let mut eigenvectors: Vec<Vec<f64>> = (0..n)
            .map(|j| (0..n).map(|i| v[(i, j)]).collect())
            .collect();

        // Sort by eigenvalue
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| eigenvalues[a].partial_cmp(&eigenvalues[b]).unwrap());
        eigenvalues = indices.iter().map(|&i| eigenvalues[i]).collect();
        eigenvectors = indices.iter().map(|&i| eigenvectors[i].clone()).collect();

        (eigenvalues, eigenvectors)
    }

    /// Find the spectral gap from sorted eigenvalues.
    fn find_spectral_gap(eigenvalues: &[f64]) -> (f64, usize) {
        for (i, &e) in eigenvalues.iter().enumerate() {
            if e.abs() > 1e-8 {
                return (e, i);
            }
        }
        (0.0, eigenvalues.len())
    }

    /// Compute normalized spectral gap.
    pub fn normalized_gap(&self) -> f64 {
        if self.eigenvalues.is_empty() {
            return 0.0;
        }
        let max_eig = self.eigenvalues.last().copied().unwrap_or(1.0);
        if max_eig.abs() < 1e-12 {
            return 0.0;
        }
        self.gap / max_eig
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mdp::*;

    #[test]
    fn test_spectral_gap_chain_2() {
        // Chain of 2 nodes: L = [[1, -1], [-1, 1]], gap = 2
        let chain = ChainMDP::new(2, 0.99);
        let lap = ObservationLaplacian::from_mdp(&chain);
        let sg = SpectralGap::compute(&lap);
        assert!((sg.gap - 2.0).abs() < 0.1, "Expected gap ≈ 2.0, got {}", sg.gap);
    }

    #[test]
    fn test_spectral_gap_chain_3() {
        // Chain 0-1-2: eigenvalues of L are 0, 1, 3. Gap = 1.
        let chain = ChainMDP::new(3, 0.99);
        let lap = ObservationLaplacian::from_mdp(&chain);
        let sg = SpectralGap::compute(&lap);
        assert!((sg.gap - 1.0).abs() < 0.15, "Expected gap ≈ 1.0, got {}", sg.gap);
    }

    #[test]
    fn test_spectral_gap_chain_4() {
        // Chain of 4: eigenvalues 0, 2-sqrt(2), 2, 2+sqrt(2). Gap ≈ 0.586
        let chain = ChainMDP::new(4, 0.99);
        let lap = ObservationLaplacian::from_mdp(&chain);
        let sg = SpectralGap::compute(&lap);
        let expected = 2.0 - std::f64::consts::SQRT_2;
        assert!((sg.gap - expected).abs() < 0.2, "Expected gap ≈ {}, got {}", expected, sg.gap);
    }

    #[test]
    fn test_zero_eigenvalue_exists() {
        let grid = GridWorldMDP::new(3, 3, 0.99);
        let lap = ObservationLaplacian::from_mdp(&grid);
        let sg = SpectralGap::compute(&lap);
        assert!(sg.eigenvalues[0].abs() < 0.1, "First eigenvalue should be ~0, got {}", sg.eigenvalues[0]);
    }

    #[test]
    fn test_spectral_gap_positive() {
        let grid = GridWorldMDP::new(3, 3, 0.99);
        let lap = ObservationLaplacian::from_mdp(&grid);
        let sg = SpectralGap::compute(&lap);
        assert!(sg.gap > 0.0, "Spectral gap should be positive");
    }

    #[test]
    fn test_n_components_connected() {
        let grid = GridWorldMDP::new(3, 3, 0.99);
        let lap = ObservationLaplacian::from_mdp(&grid);
        let sg = SpectralGap::compute(&lap);
        assert_eq!(sg.n_components, 1);
    }

    #[test]
    fn test_fiedler_vector_exists() {
        let chain = ChainMDP::new(4, 0.99);
        let lap = ObservationLaplacian::from_mdp(&chain);
        let sg = SpectralGap::compute(&lap);
        assert_eq!(sg.fiedler_vector.len(), 4);
    }

    #[test]
    fn test_temperature_reduces_gap() {
        let chain = ChainMDP::new(5, 0.99);
        let lap = ObservationLaplacian::from_mdp(&chain);
        let sg_cold = SpectralGap::compute(&lap);
        let sg_hot = SpectralGap::compute_with_temperature(&lap, 5.0);
        assert!(sg_cold.gap > sg_hot.gap, "Cold gap {} should > hot gap {}", sg_cold.gap, sg_hot.gap);
    }

    #[test]
    fn test_normalized_gap() {
        let chain = ChainMDP::new(3, 0.99);
        let lap = ObservationLaplacian::from_mdp(&chain);
        let sg = SpectralGap::compute(&lap);
        let ng = sg.normalized_gap();
        assert!(ng > 0.0 && ng <= 1.0);
    }

    #[test]
    fn test_grid_5x5_gap() {
        let grid = GridWorldMDP::new(5, 5, 0.99);
        let lap = ObservationLaplacian::from_mdp(&grid);
        let sg = SpectralGap::compute(&lap);
        assert!(sg.gap > 0.0);
        assert!(sg.gap < 10.0); // Reasonable range for 5x5
    }

    #[test]
    fn test_serialization() {
        let chain = ChainMDP::new(3, 0.99);
        let lap = ObservationLaplacian::from_mdp(&chain);
        let sg = SpectralGap::compute(&lap);
        let json = serde_json::to_string(&sg).unwrap();
        let decoded: SpectralGap = serde_json::from_str(&json).unwrap();
        assert!((decoded.gap - sg.gap).abs() < 1e-10);
    }

    #[test]
    fn test_grid_10x10_gap() {
        let grid = GridWorldMDP::new(10, 10, 0.99);
        let lap = ObservationLaplacian::from_mdp(&grid);
        let sg = SpectralGap::compute(&lap);
        assert!(sg.gap > 0.0);
    }
}
