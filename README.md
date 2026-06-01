# lau-spectral-gap-experiment

**Experimental verification: spectral gap = RL convergence rate — testing Opus's Emergent Theorem A.**

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)
[![Rust](https://img.shields.io/badge/rust-2021-orange.svg)](https://www.rust-lang.org/)

## What This Does

This crate experimentally tests a deep prediction: **the spectral gap of the graph Laplacian on an MDP's state space equals the convergence rate of entropy-regularized policy gradient on that MDP**.

If the rates match across diverse MDPs (grid worlds, chains, random graphs), this experimentally confirms Emergent Theorem A from the spectral theory of reinforcement learning: the spectral gap is not just an abstract eigenvalue—it directly predicts how fast your RL algorithm converges.

The crate provides:
- **MDP testbed** — grid worlds, chain MDPs, random MDPs with the `MDP` trait
- **Observation Laplacian** — graph Laplacian construction from MDP state adjacency
- **Spectral gap computation** — eigenvalues, Fiedler vector, algebraic connectivity
- **Policy gradient runner** — entropy-regularized PG with convergence rate estimation
- **Rate comparison** — spectral gap vs. PG convergence rate, match/tolerance checking
- **Composable pipeline** — Laplacian → spectral gap ↔ PG rate → comparison

## Key Idea

```
Graph Laplacian L = D - A
    ↓ eigenvalues
Spectral gap λ₁ = smallest nonzero eigenvalue of L
    ↓ should equal
PG convergence rate = slope of log(KL divergence) over time

If λ₁ ≈ convergence rate ⟹ Emergent Theorem A confirmed
```

The spectral gap determines the slowest-relaxing mode of the diffusion process on the state graph. Policy gradient, under entropy regularization, follows this diffusion. Therefore, the convergence rate of the RL algorithm is **predictable from the graph structure alone**—no need to actually run the algorithm.

## Install

```toml
[dependencies]
lau-spectral-gap-experiment = "0.1"
```

```bash
cargo add lau-spectral-gap-experiment
```

Dependencies: `nalgebra` 0.33, `serde` 1, `rand` 0.8, `rand_distr` 0.4.

> **Note:** The `lib.rs` declares `varadhan`, `temperature_sweep`, and `statistics` modules that are not yet implemented. Only the 6 core modules listed below are functional.

## Quick Start

### Build an MDP and Compute Spectral Gap

```rust
use lau_spectral_gap_experiment::{GridWorldMDP, MDP, ObservationLaplacian, SpectralGap};

// 4×4 grid world with goal at bottom-right
let mdp = GridWorldMDP::new(4, 4, 0.99);

// Build observation Laplacian from MDP adjacency
let laplacian = ObservationLaplacian::from_mdp(&mdp);
println!("Laplacian size: {}×{}", laplacian.n, laplacian.n);

// Compute spectral gap
let sg = SpectralGap::compute(&laplacian);
println!("Spectral gap: {:.6}", sg.gap);
println!("Algebraic connectivity: {:.6}", sg.algebraic_connectivity);
println!("Connected components: {}", sg.n_components);
println!("Fiedler vector: {:?}", sg.fiedler_vector);
```

### Run Policy Gradient and Compare Rates

```rust
use lau_spectral_gap_experiment::{ChainMDP, MDP, RateComparison};

// 10-state chain MDP
let mdp = ChainMDP::new(10, 0.99);

// Compare spectral gap vs PG convergence rate
let comparison = RateComparison::compare(
    &mdp,    // the MDP
    0.1,     // temperature ℏ
    0.01,    // learning rate
    1000,    // PG steps
    0.5,     // tolerance for "match"
);

println!("MDP: {} ({} states)", comparison.mdp_name, comparison.mdp_size);
println!("Spectral gap:     {:.6}", comparison.spectral_gap);
println!("Convergence rate: {:.6}", comparison.convergence_rate);
println!("Relative diff:    {:.4}%", comparison.relative_diff * 100.0);
println!("Matches: {}", comparison.matches);
```

### Temperature Sweep

```rust
use lau_spectral_gap_experiment::{GridWorldMDP, MDP, ObservationLaplacian, SpectralGap};

let mdp = GridWorldMDP::new(5, 5, 0.99);
let lap = ObservationLaplacian::from_mdp(&mdp);

for hbar in [0.01, 0.1, 0.5, 1.0, 5.0] {
    let sg = SpectralGap::compute_with_temperature(&lap, hbar);
    println!("ℏ={:.2}: gap={:.6}", hbar, sg.gap);
}
```

### Random MDP

```rust
use lau_spectral_gap_experiment::{RandomMDP, MDP, RateComparison};

// Random MDP: 20 states, 4 actions, sparsity 0.3
let mdp = RandomMDP::new(20, 4, 0.3, 0.99);
let comparison = RateComparison::compare(&mdp, 0.1, 0.01, 500, 0.5);
```

## API Reference

### MDP Trait and Implementations

| Type | Description |
|------|-------------|
| `MDP` (trait) | Core interface: states, actions, transitions, rewards, adjacency |
| `GridWorldMDP` | Rows × cols grid with 4 actions, optional obstacles |
| `ChainMDP` | N-state chain with 2 actions (left/right) |
| `RandomMDP` | Random sparse transition matrix |

### Laplacian and Spectral

| Type | Description |
|------|-------------|
| `ObservationLaplacian` | L = D - A from MDP adjacency, with temperature scaling |
| `SpectralGap` | Eigenvalues, gap, Fiedler vector, algebraic connectivity |

### Policy Gradient

| Type | Description |
|------|-------------|
| `PolicyGradientRunner` | Entropy-regularized PG with configurable ℏ, lr, steps |
| `PGResult` | KL divergences, value trajectories, convergence rate |

### Comparison

| Type | Description |
|------|-------------|
| `RateComparison` | Spectral gap vs convergence rate, relative diff, match boolean |

## How It Works

### Step 1: Build the Graph Laplacian

From the MDP's adjacency structure, construct the combinatorial Laplacian:

```
L = D - A

D[i,i] = degree of state i (number of neighbors)
A[i,j] = 1 if states i,j are adjacent
```

This encodes the graph structure of the state space.

### Step 2: Compute the Spectral Gap

The spectral gap is the smallest nonzero eigenvalue of L:

```
λ₁ = min{λ : Lv = λv, v ≠ 0, v ⊥ ker(L)}
```

This equals the **algebraic connectivity** of the graph. A large gap means fast mixing; a small gap means bottlenecks.

### Step 3: Run Policy Gradient

Entropy-regularized policy gradient with temperature ℏ:

```
π_{t+1}(a|s) ∝ π_t(a|s) · exp(η · Q^π_t(s,a) / ℏ)
```

Track the KL divergence from the optimal policy at each step. The convergence rate is the slope of log(KL) over time.

### Step 4: Compare Rates

If Emergent Theorem A holds:

```
spectral_gap ≈ convergence_rate
```

The relative difference should be small across different MDPs and temperatures.

## The Math

### Graph Laplacian

For a graph G = (V, E) with adjacency matrix A and degree matrix D:

```
L = D - A

Eigenvalues: 0 = λ₀ ≤ λ₁ ≤ ... ≤ λ_{n-1}
```

- λ₀ = 0 always (constant vector is eigenvector)
- λ₁ > 0 iff the graph is connected (Fiedler's theorem)
- The eigenvector for λ₁ is the **Fiedler vector**, used in spectral clustering

### Cheeger Inequality

The spectral gap relates to the graph's isoperimetric number h(G):

```
λ₁/2 ≤ h(G) ≤ √(2λ₁)
```

This means: narrow bottlenecks → small spectral gap → slow convergence.

### Entropy-Regularized Policy Gradient

Adding entropy regularization with temperature ℏ:

```
J(π) = E[Σ γ^t (r(s_t,a_t) + ℏ · H(π(·|s_t)))]
```

The optimal policy is the **Gibbs distribution**:

```
π*(a|s) ∝ exp(Q*(s,a) / ℏ)
```

Policy gradient converges exponentially with rate determined by the curvature of the objective, which is bounded by the spectral gap.

### Varadhan's Connection

In the continuous limit (WKB/semiclassical):

```
-ℏ log(K_t(x,y)) → d(x,y)² as ℏ → 0
```

The heat kernel connects to geodesic distance, linking spectral theory to geometry.

## Test Coverage

61 tests across 5 modules:
- **MDP** (19 tests): grid world construction, chain construction, random MDP, transition validity, adjacency
- **Laplacian** (10 tests): construction, degree matrix, positive semi-definiteness, temperature scaling
- **Spectral** (12 tests): eigenvalue computation, spectral gap, Fiedler vector, algebraic connectivity, connected components
- **Policy Gradient** (11 tests): convergence, KL divergence tracking, value trajectories, temperature dependence
- **Comparison** (9 tests): rate matching, relative difference, tolerance checking, cross-MDP validation

## License

MIT
