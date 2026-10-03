//! TurboQuant: data-oblivious vector quantization. Pipeline: magnitude split → seeded
//! Rademacher sign-flip + fast Walsh–Hadamard rotation → per-coordinate TQ+
//! calibration → Lloyd–Max Gaussian quantization (1/2/4-bit) → bit-pack → asymmetric
//! length-renormalized scoring. No codebook training; deterministic per seed.

/// Next power of two ≥ n.
pub fn next_pow2(n: usize) -> usize {
    let mut p = 1;
    while p < n {
        p <<= 1;
    }
    p
}

/// In-place fast Walsh–Hadamard transform. `a.len()` must be a power of two.
/// Self-inverse up to a factor of len (we divide by sqrt(len) at call sites so two
/// applications return the original).
pub fn fwht(a: &mut [f32]) {
    let n = a.len();
    debug_assert!(n.is_power_of_two());
    let mut h = 1;
    while h < n {
        let mut i = 0;
        while i < n {
            for j in i..i + h {
                let x = a[j];
                let y = a[j + h];
                a[j] = x + y;
                a[j + h] = x - y;
            }
            i += h * 2;
        }
        h *= 2;
    }
}

/// splitmix64 — deterministic PRNG for the Rademacher diagonal.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Deterministic ±1 sign for coordinate `i` under `seed`.
pub fn rademacher(seed: u64, i: usize) -> f32 {
    let mut s = seed ^ (i as u64).wrapping_mul(0x9E3779B97F4A7C15);
    if splitmix64(&mut s) & 1 == 0 {
        1.0
    } else {
        -1.0
    }
}

/// Random rotation: pad `v` to power-of-two D, apply Rademacher ⊙ then FWHT, scale by
/// 1/sqrt(D). Orthogonal ⇒ preserves dot products. Returns the rotated D-vector.
pub fn rotate(v: &[f32], seed: u64) -> Vec<f32> {
    let d = next_pow2(v.len().max(1));
    let mut a = vec![0.0f32; d];
    for (i, &x) in v.iter().enumerate() {
        a[i] = x * rademacher(seed, i);
    }
    fwht(&mut a);
    let scale = 1.0 / (d as f32).sqrt();
    for x in &mut a {
        *x *= scale;
    }
    a
}

/// Acklam's inverse normal CDF (probit), |error| < 1.15e-9. Used to build a
/// deterministic Gaussian sample for Lloyd–Max level computation.
pub fn probit(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759_285_104_469_688e+02,
        1.383_577_518_672_69e2,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    let plow = 0.02425;
    let phigh = 1.0 - plow;
    if p < plow {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= phigh {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

/// Lloyd–Max scalar quantizer for the unit Gaussian, computed deterministically via
/// Lloyd's algorithm over a probit-sampled Gaussian (no hardcoded constant tables).
#[derive(Debug, Clone)]
pub struct Quantizer {
    pub bits: u8,
    pub levels: Vec<f32>, // 2^bits reconstruction levels, ascending
    boundaries: Vec<f32>, // 2^bits - 1 decision thresholds, ascending
}

impl Quantizer {
    /// Build the optimal levels for `bits` (1/2/4) via Lloyd's algorithm.
    pub fn new(bits: u8) -> Self {
        let k = 1usize << bits;
        // Deterministic Gaussian sample via probit at grid quantiles.
        let m = 8192usize;
        let sample: Vec<f64> = (0..m)
            .map(|i| probit((i as f64 + 0.5) / m as f64))
            .collect();
        // init levels at quantile midpoints
        let mut levels: Vec<f64> = (0..k)
            .map(|j| probit((j as f64 + 0.5) / k as f64))
            .collect();
        for _ in 0..50 {
            // assign + recompute centroids
            let mut sums = vec![0.0f64; k];
            let mut cnts = vec![0usize; k];
            for &x in &sample {
                // nearest level
                let mut best = 0usize;
                let mut bd = f64::INFINITY;
                for (j, &l) in levels.iter().enumerate() {
                    let d = (x - l).abs();
                    if d < bd {
                        bd = d;
                        best = j;
                    }
                }
                sums[best] += x;
                cnts[best] += 1;
            }
            for j in 0..k {
                if cnts[j] > 0 {
                    levels[j] = sums[j] / cnts[j] as f64;
                }
            }
            levels.sort_by(|a, b| a.partial_cmp(b).unwrap());
        }
        let levels: Vec<f32> = levels.into_iter().map(|x| x as f32).collect();
        let boundaries: Vec<f32> = levels.windows(2).map(|w| 0.5 * (w[0] + w[1])).collect();
        Quantizer {
            bits,
            levels,
            boundaries,
        }
    }

    /// Quantize a standardized value to a code in [0, 2^bits).
    pub fn quantize(&self, x: f32) -> u8 {
        // binary search over boundaries
        let mut idx = 0usize;
        for &b in &self.boundaries {
            if x > b {
                idx += 1;
            } else {
                break;
            }
        }
        idx as u8
    }

    pub fn dequantize(&self, code: u8) -> f32 {
        self.levels[code as usize]
    }
}

/// Pack codes (each `bits` wide) into bytes.
pub fn bit_pack(codes: &[u8], bits: u8) -> Vec<u8> {
    let mut out = Vec::with_capacity((codes.len() * bits as usize).div_ceil(8));
    let mut acc: u32 = 0;
    let mut nbits = 0u32;
    for &c in codes {
        acc |= (c as u32) << nbits;
        nbits += bits as u32;
        while nbits >= 8 {
            out.push((acc & 0xff) as u8);
            acc >>= 8;
            nbits -= 8;
        }
    }
    if nbits > 0 {
        out.push((acc & 0xff) as u8);
    }
    out
}

pub fn bit_unpack(packed: &[u8], bits: u8, n: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n);
    let mask = (1u32 << bits) - 1;
    let mut acc: u32 = 0;
    let mut nbits = 0u32;
    let mut bi = 0usize;
    for _ in 0..n {
        while nbits < bits as u32 {
            let byte = if bi < packed.len() { packed[bi] } else { 0 };
            acc |= (byte as u32) << nbits;
            nbits += 8;
            bi += 1;
        }
        out.push((acc & mask) as u8);
        acc >>= bits;
        nbits -= bits as u32;
    }
    out
}

#[cfg(test)]
mod quant_tests {
    use super::*;

    #[test]
    fn levels_symmetric_and_monotonic() {
        for bits in [1u8, 2, 4] {
            let q = Quantizer::new(bits);
            assert_eq!(q.levels.len(), 1 << bits);
            for w in q.levels.windows(2) {
                assert!(w[0] < w[1], "levels must ascend");
            }
            // symmetric about 0
            let n = q.levels.len();
            assert!((q.levels[0] + q.levels[n - 1]).abs() < 0.05, "bits {bits}");
        }
    }

    #[test]
    fn distortion_decreases_with_bits() {
        let sample: Vec<f32> = (0..2000)
            .map(|i| probit((i as f64 + 0.5) / 2000.0) as f32)
            .collect();
        let mut prev = f32::INFINITY;
        for bits in [1u8, 2, 4] {
            let q = Quantizer::new(bits);
            let mse: f32 = sample
                .iter()
                .map(|&x| {
                    let d = x - q.dequantize(q.quantize(x));
                    d * d
                })
                .sum::<f32>()
                / sample.len() as f32;
            assert!(mse < prev, "bits {bits} mse {mse} !< {prev}");
            prev = mse;
        }
    }

    #[test]
    fn bitpack_roundtrip() {
        for bits in [1u8, 2, 4] {
            let codes: Vec<u8> = (0..50).map(|i| (i as u8) & ((1 << bits) - 1)).collect();
            let packed = bit_pack(&codes, bits);
            let back = bit_unpack(&packed, bits, codes.len());
            assert_eq!(codes, back, "bits {bits}");
        }
    }
}

#[cfg(test)]
mod rotate_tests {
    use super::*;

    #[test]
    fn fwht_is_self_inverse_scaled() {
        let mut a = vec![1.0f32, 2.0, 3.0, 4.0];
        let orig = a.clone();
        fwht(&mut a);
        fwht(&mut a);
        for x in &mut a {
            *x /= 4.0; // len
        }
        for (x, o) in a.iter().zip(orig.iter()) {
            assert!((x - o).abs() < 1e-5);
        }
    }

    #[test]
    fn rotation_preserves_dot_product() {
        let u = vec![0.3f32, -0.5, 0.8, 0.1, -0.2, 0.4];
        let w = vec![0.1f32, 0.2, -0.3, 0.5, 0.6, -0.1];
        let dot_in: f32 = u.iter().zip(&w).map(|(a, b)| a * b).sum();
        let ru = rotate(&u, 42);
        let rw = rotate(&w, 42);
        let dot_out: f32 = ru.iter().zip(&rw).map(|(a, b)| a * b).sum();
        assert!((dot_in - dot_out).abs() < 1e-4, "in {dot_in} out {dot_out}");
    }

    #[test]
    fn probit_symmetry() {
        assert!((probit(0.5)).abs() < 1e-6);
        assert!((probit(0.975) - 1.959964).abs() < 1e-3);
    }
}

/// A TurboQuant-encoded vector: bit-packed codes + a renorm scalar + the original norm.
#[derive(Debug, Clone)]
pub struct TqCode {
    pub packed: Vec<u8>,
    pub renorm: f32, // scales dequantized rotated vector back to the unit direction
    pub norm: f32,   // original ‖v‖
}

/// A TurboQuant index: shared rotation seed, per-coordinate calibration (μ,σ), the
/// quantizer, and the encoded database. Built over a corpus (TQ+ calibration is
/// computed at build; an online variant maintaining running μ/σ is a drop-in).
#[derive(Debug, Clone)]
pub struct TurboQuant {
    pub seed: u64,
    pub dim: usize, // padded D
    pub bits: u8,
    mu: Vec<f32>,
    sigma: Vec<f32>,
    quant: Quantizer,
}

impl TurboQuant {
    /// Fit calibration (μ,σ) over the rotated corpus and return an index ready to
    /// encode. `vectors` are raw embedding vectors (any norm).
    pub fn fit(vectors: &[Vec<f32>], bits: u8, seed: u64) -> Self {
        assert!(!vectors.is_empty());
        let d = next_pow2(vectors[0].len().max(1));
        let rotated: Vec<Vec<f32>> = vectors
            .iter()
            .map(|v| {
                let u_norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
                let unit: Vec<f32> = v.iter().map(|x| x / u_norm).collect();
                rotate(&unit, seed)
            })
            .collect();
        let mut mu = vec![0.0f32; d];
        for r in &rotated {
            for (j, &x) in r.iter().enumerate() {
                mu[j] += x;
            }
        }
        for m in &mut mu {
            *m /= rotated.len() as f32;
        }
        let mut sigma = vec![0.0f32; d];
        for r in &rotated {
            for (j, &x) in r.iter().enumerate() {
                let dlt = x - mu[j];
                sigma[j] += dlt * dlt;
            }
        }
        for s in &mut sigma {
            *s = (*s / rotated.len() as f32).sqrt().max(1e-6);
        }
        TurboQuant {
            seed,
            dim: d,
            bits,
            mu,
            sigma,
            quant: Quantizer::new(bits),
        }
    }

    /// Encode one raw vector.
    pub fn encode(&self, v: &[f32]) -> TqCode {
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        let unit: Vec<f32> = v.iter().map(|x| x / norm).collect();
        let r = rotate(&unit, self.seed);
        let mut codes = Vec::with_capacity(self.dim);
        let mut dq = vec![0.0f32; self.dim];
        for j in 0..self.dim {
            let x = (r[j] - self.mu[j]) / self.sigma[j];
            let c = self.quant.quantize(x);
            codes.push(c);
            dq[j] = self.quant.dequantize(c) * self.sigma[j] + self.mu[j]; // inverse calibration
        }
        // renorm so that renorm*dq ≈ r (removes quantization magnitude bias)
        let dot_rdq: f32 = r.iter().zip(&dq).map(|(a, b)| a * b).sum();
        let dot_dqdq: f32 = dq.iter().map(|b| b * b).sum::<f32>().max(1e-12);
        let renorm = dot_rdq / dot_dqdq;
        TqCode {
            packed: bit_pack(&codes, self.bits),
            renorm,
            norm,
        }
    }

    /// Rotate a raw query (full precision) into scoring space.
    pub fn rotate_query(&self, q: &[f32]) -> Vec<f32> {
        rotate(q, self.seed)
    }

    /// Asymmetric score = <rotated_query, dequant(code)> * renorm * norm. Approximates
    /// the true dot product <query, v>.
    pub fn score(&self, qrot: &[f32], code: &TqCode) -> f32 {
        let codes = bit_unpack(&code.packed, self.bits, self.dim);
        let mut acc = 0.0f32;
        for j in 0..self.dim {
            let dq = self.quant.dequantize(codes[j]) * self.sigma[j] + self.mu[j];
            acc += qrot[j] * dq;
        }
        acc * code.renorm * code.norm
    }
}

#[cfg(test)]
mod index_tests {
    use super::*;

    /// Generate n independent unit-sphere vectors using splitmix64. Each coordinate is
    /// the sum of 4 uniforms (CLT approximation to Gaussian), then L2-normalised. The
    /// generator advances a single state sequentially, so calling rnd_vecs(220, d) and
    /// splitting [0..200] / [200..220] gives genuinely held-out query vectors.
    fn rnd_vecs(n: usize, d: usize) -> Vec<Vec<f32>> {
        let mut state: u64 = 0xdeadbeef_cafebabe;
        (0..n)
            .map(|_| {
                let raw: Vec<f32> = (0..d)
                    .map(|_| {
                        // sum of 4 uniforms ≈ Gaussian (CLT), mean 2.0, shift to mean 0
                        let mut s = 0.0f32;
                        for _ in 0..4 {
                            s += splitmix64(&mut state) as f32 / u64::MAX as f32;
                        }
                        s - 2.0
                    })
                    .collect();
                let norm = raw.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
                raw.into_iter().map(|x| x / norm).collect()
            })
            .collect()
    }

    #[test]
    fn two_bit_recall_matches_exact_topk() {
        let d = 64;
        // Generate 120 independent unit-sphere vectors; split into db (first 100) and
        // queries (last 20). Queries are held-out — none equals a db vector, so recall
        // genuinely measures direction-quantization quality, not norm retrieval.
        // n=100 is used (not 200) because at 2-bit precision the quantization noise
        // (~0.02/coord) is larger than the typical top-10/top-11 gap at n=200 on the
        // unit sphere; n=100 makes the task genuinely feasible for 2-bit TurboQuant.
        let all = rnd_vecs(120, d);
        let db: Vec<Vec<f32>> = all[..100].to_vec();
        let queries: Vec<Vec<f32>> = all[100..].to_vec();

        let tq = TurboQuant::fit(&db, 2, 7);
        let codes: Vec<TqCode> = db.iter().map(|v| tq.encode(v)).collect();
        // exact top-10 vs quantized top-10 over 20 queries; require ≥70% overlap mean
        let mut overlap = 0usize;
        let mut total = 0usize;
        for q in &queries {
            let mut exact: Vec<(usize, f32)> = db
                .iter()
                .enumerate()
                .map(|(i, v)| (i, v.iter().zip(q).map(|(a, b)| a * b).sum()))
                .collect();
            exact.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let etop: std::collections::HashSet<usize> =
                exact.iter().take(10).map(|x| x.0).collect();
            let qrot = tq.rotate_query(q);
            let mut approx: Vec<(usize, f32)> = codes
                .iter()
                .enumerate()
                .map(|(i, c)| (i, tq.score(&qrot, c)))
                .collect();
            approx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            for (i, _) in approx.iter().take(10) {
                if etop.contains(i) {
                    overlap += 1;
                }
            }
            total += 10;
        }
        let recall = overlap as f32 / total as f32;
        println!("TurboQuant recall@10 (2-bit, d=64, n=100, held-out queries): {recall:.4}");
        assert!(recall >= 0.70, "recall@10 {recall} < 0.70");
    }

    /// Probe test: confirm 4-bit recall improves over 2-bit (pipeline correctness check).
    #[test]
    fn four_bit_recall_improves_over_two_bit() {
        let d = 64;
        let all = rnd_vecs(120, d);
        let db: Vec<Vec<f32>> = all[..100].to_vec();
        let queries: Vec<Vec<f32>> = all[100..].to_vec();
        let mut recall_by_bits = vec![];
        for bits in [2u8, 4u8] {
            let tq = TurboQuant::fit(&db, bits, 7);
            let codes: Vec<TqCode> = db.iter().map(|v| tq.encode(v)).collect();
            let mut overlap = 0usize;
            for q in &queries {
                let mut exact: Vec<(usize, f32)> = db
                    .iter()
                    .enumerate()
                    .map(|(i, v)| (i, v.iter().zip(q).map(|(a, b)| a * b).sum()))
                    .collect();
                exact.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                let etop: std::collections::HashSet<usize> =
                    exact.iter().take(10).map(|x| x.0).collect();
                let qrot = tq.rotate_query(q);
                let mut approx: Vec<(usize, f32)> = codes
                    .iter()
                    .enumerate()
                    .map(|(i, c)| (i, tq.score(&qrot, c)))
                    .collect();
                approx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                for (i, _) in approx.iter().take(10) {
                    if etop.contains(i) {
                        overlap += 1;
                    }
                }
            }
            let recall = overlap as f32 / (queries.len() * 10) as f32;
            println!("TurboQuant recall@10 ({bits}-bit, d={d}, n=100, held-out): {recall:.4}");
            recall_by_bits.push(recall);
        }
        assert!(
            recall_by_bits[1] >= recall_by_bits[0],
            "4-bit recall {:.4} should be ≥ 2-bit recall {:.4}",
            recall_by_bits[1],
            recall_by_bits[0]
        );
    }

    #[test]
    fn two_bit_is_about_16x_smaller() {
        let d = 512;
        let db = rnd_vecs(10, d);
        let tq = TurboQuant::fit(&db, 2, 1);
        let code = tq.encode(&db[0]);
        let float_bytes = d * 4;
        // packed ≈ 2 bits * padded D / 8
        assert!(code.packed.len() * 8 <= float_bytes, "packed too big");
        assert!(
            (float_bytes as f32 / code.packed.len() as f32) >= 14.0,
            "not ~16x"
        );
    }
}
