//! consent::metrics — rate estimators with 95% Wilson CIs, Newcombe CIs for a
//! paired-rate difference, a McNemar exact/asymptotic test for the paired
//! live-vs-gated comparison (E1), Cohen's kappa for the goals-of-care label
//! agreement (E5), and a seeded permutation test. Pure and deterministic.
//!
//! The Wilson / Newcombe / permutation functions mirror `equity::metrics` so the
//! two papers report rates identically; `mcnemar` and `cohens_kappa` are new here.

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

/// Newcombe square-and-add 95% CI for the difference of two independent
/// proportions `a = ka/na` minus `b = kb/nb`. Returns (gap, ci_lo, ci_hi).
pub fn newcombe_diff(ka: u64, na: u64, kb: u64, nb: u64) -> (f64, f64, f64) {
    let z = 1.96;
    let ra = rate(ka, na);
    let rb = rate(kb, nb);
    let gap = ra - rb;
    let (la, ua) = wilson_ci(ka, na, z);
    let (lb, ub) = wilson_ci(kb, nb, z);
    let ci_lo = gap - ((ra - la).powi(2) + (ub - rb).powi(2)).sqrt();
    let ci_hi = gap + ((ua - ra).powi(2) + (rb - lb).powi(2)).sqrt();
    (gap, ci_lo, ci_hi)
}

/// McNemar's test for a paired binary comparison (e.g. the same case, ungated vs
/// gated). `b` = #cases the first arm marked positive but the second did not;
/// `c` = #cases the second arm marked positive but the first did not. Returns the
/// two-sided p-value. Uses the exact binomial tail when `b + c` is small
/// (< 25 discordant pairs) and the continuity-corrected chi-square otherwise.
pub fn mcnemar(b: u64, c: u64) -> f64 {
    let n = b + c;
    if n == 0 {
        return 1.0;
    }
    if n < 25 {
        // Exact: 2 * P(X <= min(b,c)) under Binomial(n, 0.5), capped at 1.
        let k = b.min(c);
        let mut tail = 0.0f64;
        for i in 0..=k {
            tail += binom_pmf(n, i, 0.5);
        }
        (2.0 * tail).min(1.0)
    } else {
        // Continuity-corrected chi-square with 1 df: (|b-c| - 1)^2 / (b+c).
        let bf = b as f64;
        let cf = c as f64;
        let chi2 = ((bf - cf).abs() - 1.0).powi(2) / (bf + cf);
        chi2_sf_1df(chi2)
    }
}

/// Binomial PMF P(X = k) for X ~ Binomial(n, p). Stable via log-gamma.
fn binom_pmf(n: u64, k: u64, p: f64) -> f64 {
    if k > n {
        return 0.0;
    }
    let ln_coeff =
        ln_factorial(n) - ln_factorial(k) - ln_factorial(n - k);
    let ln_p = ln_coeff + (k as f64) * p.ln() + ((n - k) as f64) * (1.0 - p).ln();
    ln_p.exp()
}

fn ln_factorial(n: u64) -> f64 {
    // ln(n!) = ln(Gamma(n+1)); sum of logs is exact enough for our small n.
    let mut s = 0.0;
    for i in 2..=n {
        s += (i as f64).ln();
    }
    s
}

/// Survival function of the chi-square distribution with 1 df: P(X > x) =
/// erfc(sqrt(x/2)). Uses a rational approximation of erfc.
fn chi2_sf_1df(x: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    erfc((x / 2.0).sqrt())
}

/// Complementary error function via Abramowitz & Stegun 7.1.26 (|err| < 1.5e-7).
fn erfc(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let y = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let approx = y * (-x * x).exp();
    if x >= 0.0 {
        approx
    } else {
        2.0 - approx
    }
}

/// Cohen's kappa for two raters over paired binary labels (agreement beyond
/// chance). `a` and `b` are equal-length boolean slices. Returns kappa in
/// [-1, 1]; 1.0 = perfect agreement, 0.0 = chance.
pub fn cohens_kappa(a: &[bool], b: &[bool]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let nf = n as f64;
    let mut agree = 0.0;
    let (mut a_pos, mut b_pos) = (0.0, 0.0);
    for i in 0..n {
        if a[i] == b[i] {
            agree += 1.0;
        }
        if a[i] {
            a_pos += 1.0;
        }
        if b[i] {
            b_pos += 1.0;
        }
    }
    let po = agree / nf;
    let (pa, pb) = (a_pos / nf, b_pos / nf);
    // P(chance agreement) = P(both yes) + P(both no).
    let pe = pa * pb + (1.0 - pa) * (1.0 - pb);
    if (1.0 - pe).abs() < 1e-12 {
        return 1.0;
    }
    (po - pe) / (1.0 - pe)
}

/// Two-sided permutation-test p-value for the difference in means of two boolean
/// samples, shuffling the pooled labels `iters` times under a seeded RNG.
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
    fn wilson_ci_is_symmetric_at_half() {
        let (lo, hi) = wilson_ci(5, 10, 1.96);
        approx((lo + hi) / 2.0, 0.5, 1e-9);
    }

    #[test]
    fn newcombe_brackets_the_gap() {
        let (gap, lo, hi) = newcombe_diff(9, 10, 1, 10);
        approx(gap, 0.8, 1e-12);
        assert!(lo < gap && gap < hi, "CI should bracket the gap");
        assert!(lo > 0.0, "0.9 vs 0.1 at n=10 should exclude 0");
    }

    #[test]
    fn mcnemar_perfect_swing_is_significant() {
        // 20 cases flipped from undisclosed -> disclosed, none the other way.
        let p = mcnemar(20, 0);
        assert!(p < 0.001, "a 20:0 swing should be highly significant, got p={p}");
    }

    #[test]
    fn mcnemar_balanced_is_not_significant() {
        let p = mcnemar(8, 7);
        assert!(p > 0.5, "near-balanced discordance should be non-significant, got p={p}");
    }

    #[test]
    fn mcnemar_no_discordance_is_one() {
        assert_eq!(mcnemar(0, 0), 1.0);
    }

    #[test]
    fn mcnemar_large_n_uses_chisquare() {
        // 40 vs 10 discordant: clearly significant via the asymptotic branch.
        let p = mcnemar(40, 10);
        assert!(p < 0.001, "40 vs 10 should be significant, got p={p}");
    }

    #[test]
    fn kappa_perfect_agreement() {
        let a = vec![true, false, true, false, true];
        approx(cohens_kappa(&a, &a), 1.0, 1e-9);
    }

    #[test]
    fn kappa_chance_agreement_near_zero() {
        // Rater A all-true, rater B alternating: agreement = chance.
        let a = vec![true, true, true, true];
        let b = vec![true, false, true, false];
        let k = cohens_kappa(&a, &b);
        assert!(k.abs() < 1e-9, "all-positive vs anything has pe=po, kappa~0, got {k}");
    }

    #[test]
    fn permutation_detects_strong_disparity() {
        let a = vec![true; 18].into_iter().chain(vec![false; 2]).collect::<Vec<_>>();
        let b = vec![true; 2].into_iter().chain(vec![false; 18]).collect::<Vec<_>>();
        let p = permutation_pvalue(&a, &b, 2000, 42);
        assert!(p < 0.01, "strong planted disparity should be significant, got p={p}");
    }

    #[test]
    fn permutation_is_deterministic_under_seed() {
        let a = vec![true, false, true, true, false];
        let b = vec![false, false, true, false, true];
        assert_eq!(
            permutation_pvalue(&a, &b, 500, 7),
            permutation_pvalue(&a, &b, 500, 7)
        );
    }
}
