//! Linearly-solvable MDP testbed: grid worlds, chain MDPs, random MDPs.

use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Core MDP trait.
pub trait MDP: Send + Sync {
    /// Number of states.
    fn n_states(&self) -> usize;
    /// Number of actions.
    fn n_actions(&self) -> usize;
    /// Transition matrix: P[a] is n_states x n_states.
    fn transition_matrices(&self) -> Vec<DMatrix<f64>>;
    /// Reward function: R[s][a].
    fn rewards(&self) -> DMatrix<f64>;
    /// Discount factor.
    fn discount(&self) -> f64;
    /// Adjacency list for graph construction.
    fn adjacency(&self) -> Vec<Vec<usize>>;
    /// Name of the MDP.
    fn name(&self) -> &str;
}

/// Grid world MDP.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridWorldMDP {
    pub rows: usize,
    pub cols: usize,
    pub discount: f64,
    pub goal_states: Vec<usize>,
    pub obstacles: Vec<usize>,
}

impl GridWorldMDP {
    pub fn new(rows: usize, cols: usize, discount: f64) -> Self {
        let goal_states = vec![rows * cols - 1]; // bottom-right
        Self {
            rows,
            cols,
            discount,
            goal_states,
            obstacles: Vec::new(),
        }
    }

    pub fn with_obstacles(mut self, obstacles: Vec<usize>) -> Self {
        self.obstacles = obstacles;
        self
    }

    fn state_to_rc(&self, s: usize) -> (usize, usize) {
        (s / self.cols, s % self.cols)
    }

    fn rc_to_state(&self, r: usize, c: usize) -> usize {
        r * self.cols + c
    }

    /// 4 actions: up, down, left, right
    pub fn actions(&self) -> &'static [(i64, i64)] {
        &[(-1, 0), (1, 0), (0, -1), (0, 1)]
    }
}

impl MDP for GridWorldMDP {
    fn n_states(&self) -> usize {
        self.rows * self.cols
    }

    fn n_actions(&self) -> usize {
        4
    }

    fn transition_matrices(&self) -> Vec<DMatrix<f64>> {
        let n = self.n_states();
        let actions = self.actions();
        let mut result = Vec::new();

        for &(_, _) in actions.iter() {
            let mut p = DMatrix::zeros(n, n);
            for s in 0..n {
                if self.obstacles.contains(&s) {
                    p[(s, s)] = 1.0;
                    continue;
                }
                if self.goal_states.contains(&s) {
                    p[(s, s)] = 1.0;
                    continue;
                }
                let (sr, sc) = self.state_to_rc(s);
                // Deterministic transition for this action
                let mut targets = Vec::new();
                for &(dr, dc) in actions {
                    let nr = (sr as i64 + dr) as usize;
                    let nc = (sc as i64 + dc) as usize;
                    if nr < self.rows && nc < self.cols {
                        let ns = self.rc_to_state(nr, nc);
                        if !self.obstacles.contains(&ns) {
                            targets.push(ns);
                        } else {
                            targets.push(s);
                        }
                    } else {
                        targets.push(s);
                    }
                }
                // Uniform over valid transitions for this action
                // Actually, each action index corresponds to one direction
                // Let's make it deterministic per action
            }
            result.push(p);
        }

        // Rebuild properly: action i → deterministic move in direction i
        for (ai, &(dr, dc)) in actions.iter().enumerate() {
            let p = &mut result[ai];
            for s in 0..n {
                if self.obstacles.contains(&s) || self.goal_states.contains(&s) {
                    p[(s, s)] = 1.0;
                    continue;
                }
                let (sr, sc) = self.state_to_rc(s);
                let nr = (sr as i64 + dr) as usize;
                let nc = (sc as i64 + dc) as usize;
                if nr < self.rows && nc < self.cols {
                    let ns = self.rc_to_state(nr, nc);
                    if !self.obstacles.contains(&ns) {
                        p[(s, ns)] = 1.0;
                    } else {
                        p[(s, s)] = 1.0;
                    }
                } else {
                    p[(s, s)] = 1.0;
                }
            }
        }

        result
    }

    fn rewards(&self) -> DMatrix<f64> {
        let n = self.n_states();
        let na = self.n_actions();
        let mut r = DMatrix::zeros(n, na);
        for &g in &self.goal_states {
            for a in 0..na {
                r[(g, a)] = 1.0;
            }
        }
        r
    }

    fn discount(&self) -> f64 {
        self.discount
    }

    fn adjacency(&self) -> Vec<Vec<usize>> {
        let n = self.n_states();
        let actions = self.actions();
        let mut adj = vec![Vec::new(); n];
        for s in 0..n {
            if self.obstacles.contains(&s) {
                continue;
            }
            let (sr, sc) = self.state_to_rc(s);
            for &(dr, dc) in actions {
                let nr = (sr as i64 + dr) as usize;
                let nc = (sc as i64 + dc) as usize;
                if nr < self.rows && nc < self.cols {
                    let ns = self.rc_to_state(nr, nc);
                    if !self.obstacles.contains(&ns) && ns != s && !adj[s].contains(&ns) {
                        adj[s].push(ns);
                        if !adj[ns].contains(&s) {
                            adj[ns].push(s);
                        }
                    }
                }
            }
        }
        adj
    }

    fn name(&self) -> &str {
        "grid_world"
    }
}

/// Chain MDP (1D).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainMDP {
    pub length: usize,
    pub discount: f64,
}

impl ChainMDP {
    pub fn new(length: usize, discount: f64) -> Self {
        Self { length, discount }
    }
}

impl MDP for ChainMDP {
    fn n_states(&self) -> usize {
        self.length
    }

    fn n_actions(&self) -> usize {
        2 // left, right
    }

    fn transition_matrices(&self) -> Vec<DMatrix<f64>> {
        let n = self.n_states();
        // Action 0: left
        let mut p_left = DMatrix::zeros(n, n);
        for s in 0..n {
            if s == 0 {
                p_left[(s, s)] = 1.0;
            } else {
                p_left[(s, s - 1)] = 1.0;
            }
        }
        // Action 1: right
        let mut p_right = DMatrix::zeros(n, n);
        for s in 0..n {
            if s == n - 1 {
                p_right[(s, s)] = 1.0;
            } else {
                p_right[(s, s + 1)] = 1.0;
            }
        }
        vec![p_left, p_right]
    }

    fn rewards(&self) -> DMatrix<f64> {
        let n = self.n_states();
        let mut r = DMatrix::zeros(n, 2);
        r[(n - 1, 0)] = 1.0;
        r[(n - 1, 1)] = 1.0;
        r
    }

    fn discount(&self) -> f64 {
        self.discount
    }

    fn adjacency(&self) -> Vec<Vec<usize>> {
        let n = self.n_states();
        let mut adj = vec![Vec::new(); n];
        for s in 0..n {
            if s > 0 {
                adj[s].push(s - 1);
            }
            if s < n - 1 {
                adj[s].push(s + 1);
            }
        }
        adj
    }

    fn name(&self) -> &str {
        "chain"
    }
}

/// Random MDP with stochastic transitions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomMDP {
    pub n_states: usize,
    pub n_actions: usize,
    pub discount: f64,
    #[serde(with = "matrix_serde")]
    pub transitions: Vec<DMatrix<f64>>,
    #[serde(with = "matrix_serde_vec")]
    pub reward_matrix: DMatrix<f64>,
    pub seed: u64,
}

mod matrix_serde {
    use nalgebra::DMatrix;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(mats: &Vec<DMatrix<f64>>, s: S) -> Result<S::Ok, S::Error> {
        let data: Vec<Vec<Vec<f64>>> = mats
            .iter()
            .map(|m| m.row_iter().map(|r| r.iter().cloned().collect()).collect())
            .collect();
        s.serialize_some(&data)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<DMatrix<f64>>, D::Error> {
        let data: Vec<Vec<Vec<f64>>> = Deserialize::deserialize(d)?;
        Ok(data
            .iter()
            .map(|rows| {
                let nrows = rows.len();
                let ncols = rows.get(0).map(|r| r.len()).unwrap_or(0);
                let flat: Vec<f64> = rows.iter().flat_map(|r| r.iter().cloned()).collect();
                DMatrix::from_row_slice(nrows, ncols, &flat)
            })
            .collect())
    }
}

mod matrix_serde_vec {
    use nalgebra::DMatrix;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(m: &DMatrix<f64>, s: S) -> Result<S::Ok, S::Error> {
        let data: Vec<Vec<f64>> = m.row_iter().map(|r| r.iter().cloned().collect()).collect();
        s.serialize_some(&data)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<DMatrix<f64>, D::Error> {
        let data: Vec<Vec<f64>> = Deserialize::deserialize(d)?;
        let nrows = data.len();
        let ncols = data.get(0).map(|r| r.len()).unwrap_or(0);
        let flat: Vec<f64> = data.iter().flat_map(|r| r.iter().cloned()).collect();
        Ok(DMatrix::from_row_slice(nrows, ncols, &flat))
    }
}

impl RandomMDP {
    pub fn new(n_states: usize, n_actions: usize, discount: f64, seed: u64) -> Self {
        use rand::Rng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut transitions = Vec::new();
        for _ in 0..n_actions {
            let mut p = DMatrix::zeros(n_states, n_states);
            for s in 0..n_states {
                // Pick 2-4 random successors
                let n_succ = 2 + rng.gen_range(0..3).min(n_states - 1);
                let mut probs = Vec::new();
                for _ in 0..n_succ {
                    probs.push(0.1 + rng.gen::<f64>());
                }
                let total: f64 = probs.iter().sum();
                for (i, pr) in probs.iter().enumerate() {
                    let t = i % n_states;
                    p[(s, t)] += pr / total;
                }
                // Normalize row
                let row_sum: f64 = (0..n_states).map(|j| p[(s, j)]).sum();
                for j in 0..n_states {
                    p[(s, j)] /= row_sum;
                }
            }
            transitions.push(p);
        }
        let mut reward_matrix = DMatrix::zeros(n_states, n_actions);
        for s in 0..n_states {
            for a in 0..n_actions {
                reward_matrix[(s, a)] = rng.gen::<f64>();
            }
        }
        Self {
            n_states,
            n_actions,
            discount,
            transitions,
            reward_matrix,
            seed,
        }
    }
}

impl MDP for RandomMDP {
    fn n_states(&self) -> usize {
        self.n_states
    }
    fn n_actions(&self) -> usize {
        self.n_actions
    }
    fn transition_matrices(&self) -> Vec<DMatrix<f64>> {
        self.transitions.clone()
    }
    fn rewards(&self) -> DMatrix<f64> {
        self.reward_matrix.clone()
    }
    fn discount(&self) -> f64 {
        self.discount
    }
    fn adjacency(&self) -> Vec<Vec<usize>> {
        let n = self.n_states;
        let mut adj = vec![Vec::new(); n];
        for pa in &self.transitions {
            for s in 0..n {
                for t in 0..n {
                    if pa[(s, t)] > 1e-10 && s != t && !adj[s].contains(&t) {
                        adj[s].push(t);
                    }
                }
            }
        }
        adj
    }
    fn name(&self) -> &str {
        "random"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_3x3_basic() {
        let grid = GridWorldMDP::new(3, 3, 0.99);
        assert_eq!(grid.n_states(), 9);
        assert_eq!(grid.n_actions(), 4);
        assert_eq!(grid.discount(), 0.99);
    }

    #[test]
    fn test_grid_transitions_deterministic() {
        let grid = GridWorldMDP::new(3, 3, 0.99);
        let ps = grid.transition_matrices();
        assert_eq!(ps.len(), 4);
        // Each row should sum to 1
        for p in &ps {
            for s in 0..grid.n_states() {
                let row_sum: f64 = (0..grid.n_states()).map(|j| p[(s, j)]).sum();
                assert!((row_sum - 1.0).abs() < 1e-10, "Row {} doesn't sum to 1: {}", s, row_sum);
            }
        }
    }

    #[test]
    fn test_grid_center_moves() {
        let grid = GridWorldMDP::new(3, 3, 0.99);
        let ps = grid.transition_matrices();
        // Center state is 4 (row 1, col 1)
        // Action 0 (up) -> state 1
        assert!((ps[0][(4, 1)] - 1.0).abs() < 1e-10);
        // Action 1 (down) -> state 7
        assert!((ps[1][(4, 7)] - 1.0).abs() < 1e-10);
        // Action 2 (left) -> state 3
        assert!((ps[2][(4, 3)] - 1.0).abs() < 1e-10);
        // Action 3 (right) -> state 5
        assert!((ps[3][(4, 5)] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_grid_corner_stays() {
        let grid = GridWorldMDP::new(3, 3, 0.99);
        let ps = grid.transition_matrices();
        // State 0 (top-left): up and left should stay
        assert!((ps[0][(0, 0)] - 1.0).abs() < 1e-10); // up -> wall
        assert!((ps[2][(0, 0)] - 1.0).abs() < 1e-10); // left -> wall
    }

    #[test]
    fn test_grid_rewards() {
        let grid = GridWorldMDP::new(3, 3, 0.99);
        let r = grid.rewards();
        // Goal is state 8
        assert!((r[(8, 0)] - 1.0).abs() < 1e-10);
        assert!((r[(8, 1)] - 1.0).abs() < 1e-10);
        // Non-goal should be 0
        assert!((r[(0, 0)]).abs() < 1e-10);
    }

    #[test]
    fn test_grid_adjacency() {
        let grid = GridWorldMDP::new(3, 3, 0.99);
        let adj = grid.adjacency();
        // Center (4) should have 4 neighbors
        assert_eq!(adj[4].len(), 4);
        // Corner (0) should have 2
        assert_eq!(adj[0].len(), 2);
    }

    #[test]
    fn test_grid_obstacles() {
        let grid = GridWorldMDP::new(3, 3, 0.99).with_obstacles(vec![4]);
        let adj = grid.adjacency();
        // Obstacle state has no neighbors
        assert!(adj[4].is_empty());
    }

    #[test]
    fn test_chain_basic() {
        let chain = ChainMDP::new(5, 0.99);
        assert_eq!(chain.n_states(), 5);
        assert_eq!(chain.n_actions(), 2);
    }

    #[test]
    fn test_chain_transitions() {
        let chain = ChainMDP::new(5, 0.99);
        let ps = chain.transition_matrices();
        // Left from state 2 -> state 1
        assert!((ps[0][(2, 1)] - 1.0).abs() < 1e-10);
        // Right from state 2 -> state 3
        assert!((ps[1][(2, 3)] - 1.0).abs() < 1e-10);
        // Left from state 0 -> stays at 0
        assert!((ps[0][(0, 0)] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_chain_rewards() {
        let chain = ChainMDP::new(5, 0.99);
        let r = chain.rewards();
        // Reward at last state
        assert!((r[(4, 0)] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_chain_adjacency() {
        let chain = ChainMDP::new(5, 0.99);
        let adj = chain.adjacency();
        // State 0 has 1 neighbor
        assert_eq!(adj[0].len(), 1);
        // State 2 has 2 neighbors
        assert_eq!(adj[2].len(), 2);
    }

    #[test]
    fn test_random_mdp_basic() {
        let rmdp = RandomMDP::new(5, 3, 0.95, 42);
        assert_eq!(rmdp.n_states(), 5);
        assert_eq!(rmdp.n_actions(), 3);
    }

    #[test]
    fn test_random_mdp_transitions_stochastic() {
        let rmdp = RandomMDP::new(5, 3, 0.95, 42);
        let ps = rmdp.transition_matrices();
        for p in &ps {
            for s in 0..rmdp.n_states() {
                let row_sum: f64 = (0..rmdp.n_states()).map(|j| p[(s, j)]).sum();
                assert!((row_sum - 1.0).abs() < 1e-8, "Row sum: {}", row_sum);
            }
        }
    }

    #[test]
    fn test_random_mdp_deterministic_seed() {
        let r1 = RandomMDP::new(5, 2, 0.95, 42);
        let r2 = RandomMDP::new(5, 2, 0.95, 42);
        let p1 = r1.transition_matrices();
        let p2 = r2.transition_matrices();
        for (a, b) in p1.iter().zip(p2.iter()) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn test_grid_5x5() {
        let grid = GridWorldMDP::new(5, 5, 0.99);
        assert_eq!(grid.n_states(), 25);
        let ps = grid.transition_matrices();
        for p in &ps {
            for s in 0..25 {
                let row_sum: f64 = (0..25).map(|j| p[(s, j)]).sum();
                assert!((row_sum - 1.0).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_grid_10x10() {
        let grid = GridWorldMDP::new(10, 10, 0.99);
        assert_eq!(grid.n_states(), 100);
    }

    #[test]
    fn test_chain_long() {
        let chain = ChainMDP::new(20, 0.99);
        assert_eq!(chain.n_states(), 20);
        let adj = chain.adjacency();
        assert_eq!(adj[10].len(), 2);
    }

    #[test]
    fn test_grid_serialization() {
        let grid = GridWorldMDP::new(3, 3, 0.99);
        let json = serde_json::to_string(&grid).unwrap();
        let decoded: GridWorldMDP = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.rows, 3);
        assert_eq!(decoded.cols, 3);
    }

    #[test]
    fn test_chain_serialization() {
        let chain = ChainMDP::new(5, 0.95);
        let json = serde_json::to_string(&chain).unwrap();
        let decoded: ChainMDP = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.length, 5);
    }
}
