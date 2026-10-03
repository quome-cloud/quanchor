//! Minimal Merkle tree over entry hashes (hex strings). Leaves are
//! blake3("leaf:" + hash_hex); internal nodes blake3("node:" + left + right).
//! Odd nodes are promoted (not duplicated). Used for batch signing (E2),
//! anchors (E3/TA6) and inclusion proofs (`qfire audit prove`).

use super::chain::blake3_hex;

fn leaf(h: &str) -> String {
    blake3_hex(format!("leaf:{h}").as_bytes())
}

fn node(l: &str, r: &str) -> String {
    blake3_hex(format!("node:{l}{r}").as_bytes())
}

/// Merkle root of entry hashes. Empty input -> blake3("leaf:").
pub fn merkle_root(hashes: &[String]) -> String {
    if hashes.is_empty() {
        return leaf("");
    }
    let mut level: Vec<String> = hashes.iter().map(|h| leaf(h)).collect();
    while level.len() > 1 {
        level = level
            .chunks(2)
            .map(|c| {
                if c.len() == 2 {
                    node(&c[0], &c[1])
                } else {
                    c[0].clone()
                }
            })
            .collect();
    }
    level.pop().unwrap()
}

/// Incremental promote-odd Merkle: push leaves one at a time; root() at any
/// point equals merkle_root(&hashes[..pushed]). Stack of (subtree_size, hash)
/// where sizes are the binary decomposition of the count; equal-size tops
/// merge on push; root = right-to-left fold of the stack.
pub struct IncrementalMerkle {
    stack: Vec<(u64, String)>,
    count: u64,
}

impl Default for IncrementalMerkle {
    fn default() -> Self {
        Self::new()
    }
}

impl IncrementalMerkle {
    pub fn new() -> Self {
        IncrementalMerkle {
            stack: Vec::new(),
            count: 0,
        }
    }

    /// Push one entry hash (raw hex, not yet leaf-hashed).
    pub fn push(&mut self, hash_hex: &str) {
        let mut size = 1u64;
        let mut h = leaf(hash_hex);
        // While the top two entries have equal size, merge them.
        while let Some(&(top_size, _)) = self.stack.last() {
            if top_size == size {
                let (_, top_h) = self.stack.pop().unwrap();
                h = node(&top_h, &h);
                size *= 2;
            } else {
                break;
            }
        }
        self.stack.push((size, h));
        self.count += 1;
    }

    /// Current Merkle root. Equals merkle_root(&hashes[..self.count()]).
    /// Empty tree -> leaf("").
    pub fn root(&self) -> String {
        if self.stack.is_empty() {
            return leaf("");
        }
        // Fold stack right-to-left: iterate in reverse order, accumulating
        // from the rightmost (smallest) element toward the leftmost (largest).
        let mut iter = self.stack.iter().rev();
        let mut acc = iter.next().unwrap().1.clone();
        for (_, h) in iter {
            acc = node(h, &acc);
        }
        acc
    }

    pub fn count(&self) -> u64 {
        self.count
    }
}

/// One step of an inclusion proof: (sibling_hash, sibling_is_left).
pub type ProofStep = (String, bool);

/// Inclusion proof for `index` within `hashes`. None if out of range.
pub fn inclusion_proof(hashes: &[String], index: usize) -> Option<Vec<ProofStep>> {
    if index >= hashes.len() {
        return None;
    }
    let mut proof = Vec::new();
    let mut level: Vec<String> = hashes.iter().map(|h| leaf(h)).collect();
    let mut i = index;
    while level.len() > 1 {
        let sib = if i % 2 == 0 { i + 1 } else { i - 1 };
        if sib < level.len() {
            proof.push((level[sib].clone(), sib < i));
        }
        level = level
            .chunks(2)
            .map(|c| {
                if c.len() == 2 {
                    node(&c[0], &c[1])
                } else {
                    c[0].clone()
                }
            })
            .collect();
        i /= 2;
    }
    Some(proof)
}

/// Verify an inclusion proof: does `hash` at some position fold to `root`?
pub fn verify_inclusion(hash: &str, proof: &[ProofStep], root: &str) -> bool {
    let mut acc = leaf(hash);
    for (sib, sib_is_left) in proof {
        acc = if *sib_is_left {
            node(sib, &acc)
        } else {
            node(&acc, sib)
        };
    }
    acc == root
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hashes(n: usize) -> Vec<String> {
        (0..n)
            .map(|i| blake3_hex(format!("entry{i}").as_bytes()))
            .collect()
    }

    #[test]
    fn root_is_deterministic_and_input_sensitive() {
        let h = hashes(5);
        assert_eq!(merkle_root(&h), merkle_root(&h));
        let mut h2 = h.clone();
        h2[2] = blake3_hex(b"tampered");
        assert_ne!(merkle_root(&h), merkle_root(&h2));
        // Reorder must change the root (TA3 at batch level).
        let mut h3 = h.clone();
        h3.swap(0, 1);
        assert_ne!(merkle_root(&h), merkle_root(&h3));
    }

    #[test]
    fn inclusion_proofs_verify_for_every_index_and_size() {
        for n in [1usize, 2, 3, 4, 5, 8, 9] {
            let h = hashes(n);
            let root = merkle_root(&h);
            for i in 0..n {
                let p = inclusion_proof(&h, i).unwrap();
                assert!(verify_inclusion(&h[i], &p, &root), "n={n} i={i}");
                assert!(!verify_inclusion(&blake3_hex(b"forged"), &p, &root));
            }
        }
        assert!(inclusion_proof(&hashes(3), 3).is_none());
    }

    #[test]
    fn leaf_node_domain_separation() {
        // A leaf value equal to an internal node's concatenation must not
        // collide: "leaf:" vs "node:" prefixes.
        let a = leaf("x");
        let b = node("x", "");
        assert_ne!(a, b);
    }

    #[test]
    fn incremental_merkle_matches_batch_for_all_n_0_to_64() {
        let all_hashes: Vec<String> = (0..=64)
            .map(|i| blake3_hex(format!("entry{i}").as_bytes()))
            .collect();
        // Empty tree must match batch root of empty slice.
        let mut im = IncrementalMerkle::new();
        assert_eq!(
            im.root(),
            merkle_root(&[]),
            "empty incremental root mismatch"
        );
        for n in 0..=64usize {
            im.push(&all_hashes[n]);
            let batch = merkle_root(&all_hashes[..=n]);
            assert_eq!(
                im.root(),
                batch,
                "incremental root mismatch after pushing {n} hashes (n+1={})",
                n + 1
            );
        }
    }
}
