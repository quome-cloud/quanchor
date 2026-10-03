//! equity::metrics — fairness rate estimators with 95% Wilson CIs, Newcombe CIs
//! for the disparity (difference of two proportions), and a seeded permutation
//! test for the null "no group difference". Pure, deterministic.

/// Point estimate of a rate k/n (0 when n == 0).
pub fn rate(k: u64, n: u64) -> f64 {
    if n == 0 {
        0.0
    } else {
        k as f64 / n as f64
    }
}

/// Wilson score interval for a binomial proportion k/n at the given z (1.96 = 95%).
/// Returns (lower, upper), both clamped to [0, 1]. For n == 0, returns (0, 1).
pub fn wilson_ci(k: u64, n: u64, z: f64) -> (f64, f64) {
    if n == 0 {
        return (0.0, 1.0);
    }
    let n = n as f64;
    let p = k as f64 / n;
    let z2 = z * z;
    let denom = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denom;
    let margin = z * ((p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt()) / denom;
    ((center - margin).max(0.0), (center + margin).min(1.0))
}

/// The worst-off pair across groups under the demographic-parity (selection-rate)
/// metric. Each group is `(name, k, n)`. Returns the higher-rate group, the
/// lower-rate group, the gap (>= 0), and a Newcombe 95% CI on the gap. None if
/// fewer than two groups have observations.
pub struct Gap {
    pub group_a: String,
    pub group_b: String,
    pub gap: f64,
    pub ci_lo: f64,
    pub ci_hi: f64,
}

pub fn max_pairwise_gap(groups: &[(String, u64, u64)]) -> Option<Gap> {
    let obs: Vec<&(String, u64, u64)> = groups.iter().filter(|(_, _, n)| *n > 0).collect();
    if obs.len() < 2 {
        return None;
    }
    let z = 1.96;
    let mut best: Option<Gap> = None;
    // Consider each unordered pair once (i < j) and orient so group_a has the
    // higher (or equal) rate, giving a non-negative gap and deterministic output
    // even when groups tie exactly.
    for i in 0..obs.len() {
        for j in (i + 1)..obs.len() {
            let ri = rate(obs[i].1, obs[i].2);
            let rj = rate(obs[j].1, obs[j].2);
            let (hi, lo) = if ri >= rj { (i, j) } else { (j, i) };
            let (na, ka, nai) = (&obs[hi].0, obs[hi].1, obs[hi].2);
            let (nb, kb, nbi) = (&obs[lo].0, obs[lo].1, obs[lo].2);
            let ra = rate(ka, nai);
            let rb = rate(kb, nbi);
            let gap = ra - rb;
            // Newcombe square-and-add CI for the difference of two proportions.
            let (la, ua) = wilson_ci(ka, nai, z);
            let (lb, ub) = wilson_ci(kb, nbi, z);
            let ci_lo = gap - ((ra - la).powi(2) + (ub - rb).powi(2)).sqrt();
            let ci_hi = gap + ((ua - ra).powi(2) + (rb - lb).powi(2)).sqrt();
            if best.as_ref().map_or(true, |g| gap > g.gap) {
                best = Some(Gap {
                    group_a: na.clone(),
                    group_b: nb.clone(),
                    gap,
                    ci_lo,
                    ci_hi,
                });
            }
        }
    }
    best
}

/// Two-sided permutation-test p-value for the difference in means of two boolean
/// samples, shuffling the pooled labels `iters` times under a seeded RNG.
/// p = (#{|perm diff| >= |obs diff|} + 1) / (iters + 1).
pub fn permutation_pvalue(a: &[bool], b: &[bool], iters: usize, seed: u64) -> f64 {
    use rand::rngs::StdRng;
    use rand::seq::SliceRandom;
    use rand::SeedableRng;

    let na = a.len();
    let nb = b.len();
    if na == 0 || nb == 0 {
        return 1.0;
    }
    let mean = |s: &[bool]| s.iter().filter(|&&x| x).count() as f64 / s.len() as f64;
    let obs = (mean(a) - mean(b)).abs();

    let mut pool: Vec<bool> = Vec::with_capacity(na + nb);
    pool.extend_from_slice(a);
    pool.extend_from_slice(b);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut hits = 0usize;
    for _ in 0..iters {
        pool.shuffle(&mut rng);
        let ma = pool[..na].iter().filter(|&&x| x).count() as f64 / na as f64;
        let mb = pool[na..].iter().filter(|&&x| x).count() as f64 / nb as f64;
        if (ma - mb).abs() >= obs - 1e-12 {
            hits += 1;
        }
    }
    (hits as f64 + 1.0) / (iters as f64 + 1.0)
}

/// Holm–Bonferroni step-down correction. Given p-values, return a boolean per
/// input (in input order) marking which are significant at family-wise `alpha`.
/// Used for the intersectional audit (E5), where many cells are tested at once.
pub fn holm_bonferroni(pvalues: &[f64], alpha: f64) -> Vec<bool> {
    let m = pvalues.len();
    let mut idx: Vec<usize> = (0..m).collect();
    idx.sort_by(|&a, &b| pvalues[a].partial_cmp(&pvalues[b]).unwrap());
    let mut sig = vec![false; m];
    for (rank, &i) in idx.iter().enumerate() {
        let threshold = alpha / (m - rank) as f64;
        if pvalues[i] <= threshold {
            sig[i] = true;
        } else {
            break; // step-down: once one fails, all larger p-values fail too
        }
    }
    sig
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(x: f64, y: f64, eps: f64) {
        assert!((x - y).abs() < eps, "expected {y}, got {x}");
    }

    #[test]
    fn rate_handles_zero_n() {
        assert_eq!(rate(0, 0), 0.0);
        approx(rate(3, 4), 0.75, 1e-12);
    }

    #[test]
    fn wilson_ci_zero_successes() {
        let (lo, hi) = wilson_ci(0, 10, 1.96);
        approx(lo, 0.0, 1e-9);
        approx(hi, 0.2776, 1e-3);
    }

    #[test]
    fn wilson_ci_is_symmetric_at_half() {
        let (lo, hi) = wilson_ci(5, 10, 1.96);
        approx(lo, 0.2366, 1e-3);
        approx(hi, 0.7634, 1e-3);
        approx((lo + hi) / 2.0, 0.5, 1e-9);
    }

    #[test]
    fn max_gap_finds_worst_pair() {
        let groups = vec![
            ("A".to_string(), 8, 10), // 0.8
            ("B".to_string(), 2, 10), // 0.2
            ("C".to_string(), 5, 10), // 0.5
        ];
        let g = max_pairwise_gap(&groups).unwrap();
        assert_eq!(g.group_a, "A");
        assert_eq!(g.group_b, "B");
        approx(g.gap, 0.6, 1e-12);
        assert!(g.ci_lo < g.gap && g.gap < g.ci_hi, "CI should bracket the gap");
        assert!(g.ci_lo > 0.0, "a 0.8 vs 0.2 gap at n=10 should exclude 0");
    }

    #[test]
    fn max_gap_none_with_one_group() {
        let groups = vec![("A".to_string(), 8, 10)];
        assert!(max_pairwise_gap(&groups).is_none());
    }

    #[test]
    fn permutation_detects_strong_disparity() {
        let a = vec![true; 18].into_iter().chain(vec![false; 2]).collect::<Vec<_>>();
        let b = vec![true; 2].into_iter().chain(vec![false; 18]).collect::<Vec<_>>();
        let p = permutation_pvalue(&a, &b, 2000, 42);
        assert!(p < 0.01, "strong planted disparity should be significant, got p={p}");
    }

    #[test]
    fn permutation_finds_no_disparity_when_balanced() {
        let a = vec![true; 10].into_iter().chain(vec![false; 10]).collect::<Vec<_>>();
        let b = vec![true; 10].into_iter().chain(vec![false; 10]).collect::<Vec<_>>();
        let p = permutation_pvalue(&a, &b, 2000, 42);
        assert!(p > 0.5, "identical groups should be non-significant, got p={p}");
    }

    #[test]
    fn permutation_is_deterministic_under_seed() {
        let a = vec![true, false, true, true, false];
        let b = vec![false, false, true, false, true];
        let p1 = permutation_pvalue(&a, &b, 500, 7);
        let p2 = permutation_pvalue(&a, &b, 500, 7);
        assert_eq!(p1, p2);
    }

    #[test]
    fn holm_bonferroni_is_more_conservative_than_raw() {
        // Four cells; only the most extreme survives the step-down at 0.05.
        let p = vec![0.001, 0.02, 0.04, 0.5];
        let sig = holm_bonferroni(&p, 0.05);
        // 0.001 <= 0.05/4=0.0125 -> sig; 0.02 <= 0.05/3=0.0167? no -> stop.
        assert_eq!(sig, vec![true, false, false, false]);
    }

    #[test]
    fn holm_bonferroni_all_significant_when_tiny() {
        let p = vec![0.0001, 0.0002, 0.0003];
        assert_eq!(holm_bonferroni(&p, 0.05), vec![true, true, true]);
    }
}
