//! E2 (paper 003): append cost per mode. Targets: chained p99 < 1 ms,
//! >= 50k entries/s/core in batched mode.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use qfire::audit::chain::EntryKind;
use qfire::audit::sign::AuditSigner;
use qfire::audit::store::{Mode, StoreCfg, TamperEvidentLog};
use qfire::audit::{AuditLog, AuditRecord};

const BODY: &str = "{\"event\":\"check\",\"decision\":\"allow\",\"detail\":\"metformin 500mg\"}";

fn open(dir: &std::path::Path, mode: Mode) -> TamperEvidentLog {
    let signer = (!matches!(mode, Mode::Plain | Mode::Chained))
        .then(|| AuditSigner::generate_to(&dir.join("key")).unwrap());
    TamperEvidentLog::open(StoreCfg {
        path: dir.join("audit.jsonl"),
        mode,
        signer,
        anchor: None,
        batch: 64,
        fail_open: false,
    })
    .unwrap()
}

fn bench_write(c: &mut Criterion) {
    let mut g = c.benchmark_group("audit_write");
    g.throughput(Throughput::Elements(1));

    // Baseline: the v1 plain log.
    // Verdict uses #[serde(rename_all = "lowercase")] so "terminal": "allow" (not "Allow").
    g.bench_function("plain_v1", |b| {
        let dir = tempfile::tempdir().unwrap();
        let log = AuditLog::open(dir.path().join("plain.jsonl"));
        let rec: AuditRecord = serde_json::from_str(
            "{\"ts\":\"2026-01-01T00:00:00Z\",\"qfire_version\":\"0\",\"event\":\"check\",\
             \"prompt_hash\":\"p\",\"chain_id\":\"c\",\"chain_version\":\"1\",\
             \"terminal\":\"allow\",\"deciding_rule\":null,\"deciding_node\":null,\
             \"reason\":\"r\",\"wall_clock_ms\":0.1,\"summed_detector_ms\":0.1,\"nodes\":[]}",
        )
        .expect("fixture record parses");
        b.iter(|| log.append(&rec).unwrap());
    });

    for (name, mode) in [
        ("chained", Mode::Chained),
        ("chained_signed", Mode::ChainedSigned),
        ("chained_signed_batched", Mode::ChainedSignedBatched),
    ] {
        g.bench_with_input(BenchmarkId::new("mode", name), &mode, |b, mode| {
            let dir = tempfile::tempdir().unwrap();
            let log = open(dir.path(), *mode);
            b.iter(|| {
                log.append_json(EntryKind::Decision, BODY.to_string())
                    .unwrap()
            });
        });
    }

    // Compute-only (E2 / H2-lower-bound / H4): blake3 chain + ed25519 sign per
    // entry with NO file write and NO fsync, isolating the cryptographic cost
    // from the per-entry-fsync durability cost. The gap between this and
    // `chained_signed` is the fail-closed durability tax.
    g.bench_function("compute_only_signed", |b| {
        use qfire::audit::chain::{build_line, GENESIS};
        let dir = tempfile::tempdir().unwrap();
        let signer = AuditSigner::generate_to(&dir.path().join("key")).unwrap();
        let mut prev = GENESIS.to_string();
        let mut seq = 0u64;
        b.iter(|| {
            let f = |h: &str| signer.sign_hash_hex(h);
            let (_line, this) = build_line(
                seq,
                "2026-01-01T00:00:00Z",
                EntryKind::Decision,
                BODY,
                &prev,
                Some(&f),
            )
            .unwrap();
            prev = this;
            seq += 1;
        });
    });
    g.finish();
}

criterion_group!(benches, bench_write);
criterion_main!(benches);
