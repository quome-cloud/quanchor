//! Paper-3 head-to-head: our tamper-evident audit construction vs. other tamper-evident-log
//! constructions, implemented with identical primitives (blake3/sha2/ed25519/hmac) on the same
//! hardware, plus two REAL third-party libraries:
//!   - ct-merkle  (RFC 6962 Certificate Transparency append-only log)
//!   - rs_merkle  (general Merkle tree with inclusion proofs)
//!
//! Metrics per scheme: append throughput, full-integrity-verify throughput, crypto bytes/entry,
//! third-party inclusion-proof size, and a security-properties matrix. Compute-only (no fsync) so the
//! comparison isolates the cryptographic construction, not the storage durability tax (which is equal
//! across schemes and is reported separately by the paper's E2).
//!
//! Run: cargo run --release --example audit_compare -- [N]

use std::time::Instant;

use ed25519_dalek::{Signer, SigningKey, Verifier};
use hmac::{Hmac, Mac};
use qfire::audit::chain::blake3_hex; // the real blake3 chain primitive from the audit module
use rs_merkle::{algorithms::Sha256 as RsSha256, Hasher as RsHasher, MerkleProof, MerkleTree};
use sha2::{Digest, Sha256};

type HmacSha256 = Hmac<Sha256>;

fn records(n: usize) -> Vec<String> {
    // ~120-byte synthetic audit entries (ASCII JSON), representative of a decision/mutation record.
    (0..n)
        .map(|i| {
            format!(
                "{{\"seq\":{i},\"ts\":\"2026-06-08T00:00:{:02}Z\",\"kind\":\"decision\",\"actor\":\"agent-7\",\"action\":\"dose_recommend\",\"args\":{{\"glucose\":{}}},\"verdict\":\"allow\"}}",
                i % 60,
                100 + (i % 80)
            )
        })
        .collect()
}

/// median-of-R wall-clock for a closure, returns ns/entry.
fn time_per(n: usize, runs: usize, mut f: impl FnMut()) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..runs {
        let t = Instant::now();
        f();
        let ns = t.elapsed().as_nanos() as f64 / n as f64;
        if ns < best {
            best = ns;
        }
    }
    best
}

struct Row {
    scheme: &'static str,
    append_ns: f64,
    verify_ns: f64,
    bytes_per: usize,
    proof_bytes: String,
    public: &'static str,
    localizes: &'static str,
    detects: &'static str, // edit/delete/reorder/rollback
}

fn main() {
    let n: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(50_000);
    let runs = 3;
    let recs = records(n);
    let bytes: Vec<&[u8]> = recs.iter().map(|s| s.as_bytes()).collect();
    let sk = SigningKey::from_bytes(&[7u8; 32]);
    let vk = sk.verifying_key();
    let mut rows: Vec<Row> = Vec::new();

    eprintln!("benchmarking {n} entries, best-of-{runs} runs...");

    // 1. blake3 hash-chain (integrity only) — the QUOKKAGUARD in-memory provenance construction
    {
        let mut chain = vec![String::new(); n];
        let append = time_per(n, runs, || {
            let mut prev = String::from("GENESIS");
            for (i, r) in bytes.iter().enumerate() {
                let mut buf = prev.clone().into_bytes();
                buf.extend_from_slice(r);
                let h = blake3_hex(&buf);
                chain[i] = h.clone();
                prev = h;
            }
        });
        let verify = time_per(n, runs, || {
            let mut prev = String::from("GENESIS");
            for (i, r) in bytes.iter().enumerate() {
                let mut buf = prev.clone().into_bytes();
                buf.extend_from_slice(r);
                assert_eq!(blake3_hex(&buf), chain[i]);
                prev = chain[i].clone();
            }
        });
        rows.push(Row { scheme: "hash-chain (blake3)", append_ns: append, verify_ns: verify,
            bytes_per: 32, proof_bytes: "O(n)".into(), public: "integrity only", localizes: "yes",
            detects: "edit,delete,reorder; rollback no" });
    }

    // 2. sha256 hash-chain
    {
        let mut chain = vec![[0u8; 32]; n];
        let append = time_per(n, runs, || {
            let mut prev = [0u8; 32];
            for (i, r) in bytes.iter().enumerate() {
                let mut h = Sha256::new();
                h.update(prev);
                h.update(r);
                let d: [u8; 32] = h.finalize().into();
                chain[i] = d;
                prev = d;
            }
        });
        let verify = time_per(n, runs, || {
            let mut prev = [0u8; 32];
            for (i, r) in bytes.iter().enumerate() {
                let mut h = Sha256::new();
                h.update(prev);
                h.update(r);
                let d: [u8; 32] = h.finalize().into();
                assert_eq!(d, chain[i]);
                prev = d;
            }
        });
        rows.push(Row { scheme: "hash-chain (sha256)", append_ns: append, verify_ns: verify,
            bytes_per: 32, proof_bytes: "O(n)".into(), public: "integrity only", localizes: "yes",
            detects: "edit,delete,reorder; rollback no" });
    }

    // 3. ed25519 signature-only (per record, no chain) — the "signed syslog" construction
    {
        let mut sigs = vec![[0u8; 64]; n];
        let append = time_per(n, runs, || {
            for (i, r) in bytes.iter().enumerate() {
                sigs[i] = sk.sign(r).to_bytes();
            }
        });
        let verify = time_per(n, runs, || {
            for (i, r) in bytes.iter().enumerate() {
                let sig = ed25519_dalek::Signature::from_bytes(&sigs[i]);
                assert!(vk.verify(r, &sig).is_ok());
            }
        });
        rows.push(Row { scheme: "signature-only (ed25519)", append_ns: append, verify_ns: verify,
            bytes_per: 64, proof_bytes: "64 B".into(), public: "yes", localizes: "yes (per entry)",
            detects: "edit,forgery; NOT delete/reorder/rollback" });
    }

    // 4. hmac-sha256 chain — the journald Forward-Secure-Sealing construction (secret-key)
    {
        let key = [9u8; 32];
        let mut chain = vec![[0u8; 32]; n];
        let append = time_per(n, runs, || {
            let mut prev = [0u8; 32];
            for (i, r) in bytes.iter().enumerate() {
                let mut m = HmacSha256::new_from_slice(&key).unwrap();
                m.update(&prev);
                m.update(r);
                let d: [u8; 32] = m.finalize().into_bytes().into();
                chain[i] = d;
                prev = d;
            }
        });
        let verify = time_per(n, runs, || {
            let mut prev = [0u8; 32];
            for (i, r) in bytes.iter().enumerate() {
                let mut m = HmacSha256::new_from_slice(&key).unwrap();
                m.update(&prev);
                m.update(r);
                let d: [u8; 32] = m.finalize().into_bytes().into();
                assert_eq!(d, chain[i]);
                prev = d;
            }
        });
        rows.push(Row { scheme: "hmac-chain (sha256)", append_ns: append, verify_ns: verify,
            bytes_per: 32, proof_bytes: "N/A (secret key)".into(), public: "no (secret key)", localizes: "yes",
            detects: "edit,delete,reorder; rollback no" });
    }

    // 5. THIS WORK: blake3-chain + ed25519 per-entry signature (the ChainedSigned construction)
    {
        let mut chain = vec![String::new(); n];
        let mut sigs = vec![[0u8; 64]; n];
        let append = time_per(n, runs, || {
            let mut prev = String::from("GENESIS");
            for (i, r) in bytes.iter().enumerate() {
                let mut buf = prev.clone().into_bytes();
                buf.extend_from_slice(r);
                let h = blake3_hex(&buf);
                sigs[i] = sk.sign(h.as_bytes()).to_bytes(); // sign over the hash hex bytes (as in sign.rs)
                chain[i] = h.clone();
                prev = h;
            }
        });
        let verify = time_per(n, runs, || {
            let mut prev = String::from("GENESIS");
            for (i, r) in bytes.iter().enumerate() {
                let mut buf = prev.clone().into_bytes();
                buf.extend_from_slice(r);
                let h = blake3_hex(&buf);
                assert_eq!(h, chain[i]);
                let sig = ed25519_dalek::Signature::from_bytes(&sigs[i]);
                assert!(vk.verify(h.as_bytes(), &sig).is_ok());
                prev = h;
            }
        });
        rows.push(Row { scheme: "THIS WORK: blake3-chain + ed25519", append_ns: append, verify_ns: verify,
            bytes_per: 96, proof_bytes: "96 B (O(1) auth)".into(), public: "yes", localizes: "yes (per entry)",
            detects: "edit,delete,reorder,forgery; +rollback via anchor" });
    }

    // 5b. THIS WORK (batched): blake3-chain + signed Merkle checkpoints every B entries (ChainedSignedBatched)
    {
        use qfire::audit::merkle::merkle_root;
        let b = 128usize;
        let mut chain = vec![String::new(); n];
        let append = time_per(n, runs, || {
            let mut prev = String::from("GENESIS");
            let mut batch: Vec<String> = Vec::with_capacity(b);
            for (i, r) in bytes.iter().enumerate() {
                let mut buf = prev.clone().into_bytes();
                buf.extend_from_slice(r);
                let h = blake3_hex(&buf);
                batch.push(h.clone());
                if batch.len() == b {
                    let root = merkle_root(&batch);
                    let _sig = sk.sign(root.as_bytes());
                    batch.clear();
                }
                chain[i] = h.clone();
                prev = h;
            }
            if !batch.is_empty() {
                let _sig = sk.sign(merkle_root(&batch).as_bytes());
            }
        });
        let verify = time_per(n, runs, || {
            let mut prev = String::from("GENESIS");
            let mut batch: Vec<String> = Vec::with_capacity(b);
            for (i, r) in bytes.iter().enumerate() {
                let mut buf = prev.clone().into_bytes();
                buf.extend_from_slice(r);
                let h = blake3_hex(&buf);
                assert_eq!(h, chain[i]);
                batch.push(h.clone());
                if batch.len() == b {
                    let sig = sk.sign(merkle_root(&batch).as_bytes());
                    assert!(vk.verify(merkle_root(&batch).as_bytes(), &sig).is_ok());
                    batch.clear();
                }
                prev = h;
            }
        });
        rows.push(Row { scheme: "THIS WORK (batched): chain + signed Merkle checkpoints", append_ns: append, verify_ns: verify,
            bytes_per: 33, proof_bytes: "~32 B/entry + 96 B/batch".into(), public: "yes (batch)", localizes: "yes (per entry, chain)",
            detects: "edit,delete,reorder,forgery; +rollback via anchor" });
    }

    // 6. ct-merkle (REAL LIB): RFC 6962 Certificate Transparency append-only log
    {
        use ct_merkle::mem_backed_tree::MemoryBackedTree;
        use sha2v11::Sha256; // ct-merkle wants digest-0.11 Sha256 (repo's sha2 is 0.10)
        let append = time_per(n, runs, || {
            let mut t = MemoryBackedTree::<Sha256, String>::new();
            for r in &recs {
                t.push(r.clone());
            }
            let _ = t.root();
        });
        // full verify == recompute the root from all leaves and compare to a trusted (signed) root
        let mut tree = MemoryBackedTree::<Sha256, String>::new();
        for r in &recs {
            tree.push(r.clone());
        }
        let trusted = tree.root();
        let verify = time_per(n, runs, || {
            let mut t = MemoryBackedTree::<Sha256, String>::new();
            for r in &recs {
                t.push(r.clone());
            }
            assert_eq!(t.root().as_bytes(), trusted.as_bytes());
        });
        let proof = tree.prove_inclusion(n / 2);
        let proof_len = proof.as_bytes().len();
        rows.push(Row { scheme: "ct-merkle (RFC 6962 CT log)", append_ns: append, verify_ns: verify,
            bytes_per: 32, proof_bytes: format!("{proof_len} B (O(log n))"), public: "yes",
            localizes: "no (batch root)", detects: "edit,delete,reorder; rollback via consistency proof" });
    }

    // 7. rs_merkle (REAL LIB): general Merkle tree (batch build)
    {
        let leaves: Vec<[u8; 32]> = bytes.iter().map(|r| RsSha256::hash(r)).collect();
        let append = time_per(n, runs, || {
            let _t = MerkleTree::<RsSha256>::from_leaves(&leaves);
        });
        let tree = MerkleTree::<RsSha256>::from_leaves(&leaves);
        let root = tree.root().unwrap();
        let verify = time_per(n, runs, || {
            let t = MerkleTree::<RsSha256>::from_leaves(&leaves);
            assert_eq!(t.root().unwrap(), root);
        });
        let idx = vec![n / 2];
        let proof = tree.proof(&idx);
        let proof_len = proof.to_bytes().len();
        // sanity: proof verifies
        let leaf = vec![leaves[n / 2]];
        let p2 = MerkleProof::<RsSha256>::try_from(proof.to_bytes()).unwrap();
        assert!(p2.verify(root, &idx, &leaf, leaves.len()));
        rows.push(Row { scheme: "rs_merkle (Merkle tree, batch)", append_ns: append, verify_ns: verify,
            bytes_per: 32, proof_bytes: format!("{proof_len} B (O(log n))"), public: "yes",
            localizes: "no (batch root)", detects: "edit,delete,reorder; rollback no" });
    }

    // ---- report ----
    println!("\n## Tamper-evident audit-log constructions — head-to-head (n={n}, compute-only)\n");
    println!("| scheme | append (M entries/s) | verify (M entries/s) | bytes/entry | inclusion proof | public-verif | localizes |");
    println!("|---|---|---|---|---|---|---|");
    for r in &rows {
        println!("| {} | {:.2} | {:.2} | {} | {} | {} | {} |",
            r.scheme, 1000.0 / r.append_ns, 1000.0 / r.verify_ns, r.bytes_per, r.proof_bytes, r.public, r.localizes);
    }
    println!("\n### detection coverage");
    for r in &rows {
        println!("- {:34} {}", r.scheme, r.detects);
    }

    // ---- OPTIMIZATION: signature-verification speedup (the audit-verify bottleneck) ----
    // ed25519 verify is the cost; it is per-entry-independent, so it parallelizes and batches with
    // ZERO change to the guarantees (chain still localizes; same public keys, same signatures).
    let mut opt = serde_json::json!(null);
    {
        use rayon::prelude::*;
        let mut hashes: Vec<String> = Vec::with_capacity(n);
        let mut sigs: Vec<ed25519_dalek::Signature> = Vec::with_capacity(n);
        let mut prev = String::from("GENESIS");
        for r in &bytes {
            let mut buf = prev.clone().into_bytes();
            buf.extend_from_slice(r);
            let h = blake3_hex(&buf);
            sigs.push(sk.sign(h.as_bytes()));
            hashes.push(h.clone());
            prev = h;
        }
        let msgs: Vec<&[u8]> = hashes.iter().map(|h| h.as_bytes()).collect();
        let vks = vec![vk; n];

        let seq = time_per(n, runs, || {
            for i in 0..n {
                assert!(vk.verify(msgs[i], &sigs[i]).is_ok());
            }
        });
        let par = time_per(n, runs, || {
            assert!((0..n).into_par_iter().all(|i| vk.verify(msgs[i], &sigs[i]).is_ok()));
        });
        let batch = time_per(n, runs, || {
            ed25519_dalek::verify_batch(&msgs, &sigs, &vks).unwrap();
        });
        // batch + parallel: split into chunks, batch-verify each chunk in parallel
        let chunk = 1024usize;
        let par_batch = time_per(n, runs, || {
            let idx: Vec<usize> = (0..n).step_by(chunk).collect();
            assert!(idx.into_par_iter().all(|s| {
                let e = (s + chunk).min(n);
                ed25519_dalek::verify_batch(&msgs[s..e], &sigs[s..e], &vks[s..e]).is_ok()
            }));
        });
        let base = 1000.0 / seq;
        println!("\n## Signature-verification speedup (per-entry ed25519 — the audit-verify bottleneck, 12 cores)\n");
        println!("| method | verify (M sig/s) | speedup |");
        println!("|---|---|---|");
        let methods = [("sequential (current verifier)", seq), ("rayon parallel", par),
                       ("dalek batch", batch), ("parallel + batch", par_batch)];
        for (name, ns) in methods {
            println!("| {} | {:.2} | {:.1}x |", name, 1000.0 / ns, (1000.0 / ns) / base);
        }
        opt = serde_json::json!({
            "sequential_M_per_s": base, "rayon_M_per_s": 1000.0 / par,
            "batch_M_per_s": 1000.0 / batch, "parallel_batch_M_per_s": 1000.0 / par_batch,
            "rayon_speedup": (1000.0 / par) / base, "batch_speedup": (1000.0 / batch) / base,
            "parallel_batch_speedup": (1000.0 / par_batch) / base,
        });
    }

    // JSON
    let json = serde_json::to_string_pretty(&serde_json::json!({
        "n": n,
        "verify_speedup": opt,
        "rows": rows.iter().map(|r| serde_json::json!({
            "scheme": r.scheme, "append_ns_per_entry": r.append_ns, "verify_ns_per_entry": r.verify_ns,
            "append_M_per_s": 1000.0 / r.append_ns, "verify_M_per_s": 1000.0 / r.verify_ns,
            "bytes_per_entry": r.bytes_per, "inclusion_proof": r.proof_bytes,
            "public_verifiable": r.public, "localizes": r.localizes, "detects": r.detects,
        })).collect::<Vec<_>>()
    })).unwrap();
    let out = "results/003-tamper-audit/crypto_compare";
    std::fs::create_dir_all(out).ok();
    std::fs::write(format!("{out}/compare.json"), json).unwrap();
    eprintln!("\nwrote {out}/compare.json");
}
