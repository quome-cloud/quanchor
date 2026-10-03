//! E4 (paper 003): verification time vs log size.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use qfire::audit::chain::{build_line, EntryKind, GENESIS};
use qfire::audit::sign::AuditSigner;
use qfire::audit::verify::verify_log;
use std::io::Write;

/// Build a clean unsigned chained log file directly via `build_line` (one
/// buffered write, no per-entry fsync). E4 measures VERIFY cost, so the writer's
/// fail-closed fsync tax is irrelevant here; this lets the 1e6 point build in
/// seconds instead of the ~hour a per-entry-fsync store loop would take.
fn build_log_file(path: &std::path::Path, n: u64) {
    let mut prev = GENESIS.to_string();
    let mut buf = String::new();
    for i in 0..n {
        let (line, this) = build_line(
            i,
            "2026-01-01T00:00:00Z",
            EntryKind::Decision,
            &format!("{{\"i\":{i}}}"),
            &prev,
            None,
        )
        .unwrap();
        buf.push_str(&line);
        buf.push('\n');
        prev = this;
    }
    let mut f = std::fs::File::create(path).unwrap();
    f.write_all(buf.as_bytes()).unwrap();
    f.sync_all().unwrap();
}

/// Build a clean PER-ENTRY-SIGNED chained log (header carries mode+pubkey, every entry signed).
/// verify_log auto-detects signed mode and uses the parallel+batch signature fast path.
fn build_signed_log_file(path: &std::path::Path, n: u64, signer: &AuditSigner) {
    let ts = "2026-01-01T00:00:00Z";
    let mut buf = String::new();
    let (hline, mut prev) = build_line(
        0,
        ts,
        EntryKind::Header,
        &format!("{{\"mode\":\"chained_signed\",\"pubkey\":\"{}\"}}", signer.pubkey_hex()),
        GENESIS,
        None,
    )
    .unwrap();
    buf.push_str(&hline);
    buf.push('\n');
    let signc = |h: &str| signer.sign_hash_hex(h);
    for i in 1..n {
        let (line, this) = build_line(
            i,
            ts,
            EntryKind::Decision,
            &format!("{{\"i\":{i}}}"),
            &prev,
            Some(&signc),
        )
        .unwrap();
        buf.push_str(&line);
        buf.push('\n');
        prev = this;
    }
    let mut f = std::fs::File::create(path).unwrap();
    f.write_all(buf.as_bytes()).unwrap();
    f.sync_all().unwrap();
}

fn bench_verify(c: &mut Criterion) {
    let dir0 = tempfile::tempdir().unwrap();
    let signer = AuditSigner::generate_to(&dir0.path().join("key")).unwrap();
    let mut g = c.benchmark_group("audit_verify");
    g.sample_size(10);
    for n in [1_000u64, 10_000, 100_000, 1_000_000] {
        g.throughput(Throughput::Elements(n));
        // unsigned chain (integrity-only baseline)
        let du = tempfile::tempdir().unwrap();
        build_log_file(&du.path().join("audit.jsonl"), n);
        g.bench_with_input(BenchmarkId::new("unsigned", n), &n, |b, _| {
            b.iter(|| {
                assert!(verify_log(&du.path().join("audit.jsonl"), None, None).unwrap().ok);
            })
        });
        // per-entry signed (the security-relevant mode), verified via the parallel+batch default
        let ds = tempfile::tempdir().unwrap();
        build_signed_log_file(&ds.path().join("audit.jsonl"), n, &signer);
        g.bench_with_input(BenchmarkId::new("signed_parallel", n), &n, |b, _| {
            b.iter(|| {
                let r = verify_log(&ds.path().join("audit.jsonl"), None, None).unwrap();
                assert!(r.ok && r.signatures_checked);
            })
        });
    }
    g.finish();
}

criterion_group!(benches, bench_verify);
criterion_main!(benches);
