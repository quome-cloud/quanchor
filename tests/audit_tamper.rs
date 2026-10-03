//! E1/E5 (paper 003): detection completeness + localization over the
//! TamperBench mutation classes. Expected: 100% detection, exact first-bad-seq.

use qfire::audit::anchor_sink::{AnchorSink, FileAnchorSink};
use qfire::audit::chain::{parse_line, EntryKind};
use qfire::audit::sign::AuditSigner;
use qfire::audit::store::{Mode, StoreCfg, TamperEvidentLog};
use qfire::audit::verify::{verify_log, TamperClass};
use std::path::Path;

fn build_clean_log(dir: &Path, n: usize, anchor_k: u64) {
    let signer = AuditSigner::generate_to(&dir.join("key")).unwrap();
    let sink = Box::new(FileAnchorSink::new(dir.join("anchors.jsonl"))) as Box<dyn AnchorSink>;
    let log = TamperEvidentLog::open(StoreCfg {
        path: dir.join("audit.jsonl"),
        mode: Mode::ChainedSigned,
        signer: Some(signer),
        anchor: Some((sink, anchor_k)),
        batch: 16,
        fail_open: false,
    })
    .unwrap();
    for i in 0..n {
        log.append_json(
            EntryKind::Decision,
            format!("{{\"event_id\":\"e{i}\",\"action\":\"order_medication\",\"detail\":\"metformin 500mg\",\"decision\":\"allow\"}}"),
        )
        .unwrap();
    }
}

fn lines(dir: &Path) -> Vec<String> {
    std::fs::read_to_string(dir.join("audit.jsonl"))
        .unwrap()
        .lines()
        .map(String::from)
        .collect()
}

fn write(dir: &Path, ls: &[String]) {
    std::fs::write(dir.join("audit.jsonl"), ls.join("\n") + "\n").unwrap();
}

#[test]
fn e1_e5_all_tamper_classes_detected_and_localized() {
    let n = 100usize;
    let mut detected = 0u32;
    let mut total = 0u32;

    // TA1: field edits at several positions.
    // The log is ChainedSigned: header=seq0, decisions=seq1..seq100.
    // ls[pos] == seq pos. Replacing "allow" in the body breaks this_hash.
    for pos in [1usize, 25, 50, 99] {
        total += 1;
        let dir = tempfile::tempdir().unwrap();
        build_clean_log(dir.path(), n, 25);
        let mut ls = lines(dir.path());
        ls[pos] = ls[pos].replace("allow", "block");
        write(dir.path(), &ls);
        let r = verify_log(&dir.path().join("audit.jsonl"), None, None).unwrap();
        assert!(!r.ok);
        assert_eq!(r.class, Some(TamperClass::HashMismatch));
        assert_eq!(
            r.first_bad_seq,
            Some(pos as u64),
            "E5 localization at {pos}"
        );
        detected += 1;
    }

    // TA2: deletions.
    for pos in [1usize, 50] {
        total += 1;
        let dir = tempfile::tempdir().unwrap();
        build_clean_log(dir.path(), n, 25);
        let mut ls = lines(dir.path());
        ls.remove(pos);
        write(dir.path(), &ls);
        let r = verify_log(&dir.path().join("audit.jsonl"), None, None).unwrap();
        assert!(!r.ok, "TA2 delete at {pos}");
        detected += 1;
    }

    // TA2: truncation (needs anchors).
    {
        total += 1;
        let dir = tempfile::tempdir().unwrap();
        build_clean_log(dir.path(), n, 25);
        let ls = lines(dir.path());
        write(dir.path(), &ls[..40].to_vec());
        let anchors = FileAnchorSink::read_all(dir.path().join("anchors.jsonl")).unwrap();
        let r = verify_log(&dir.path().join("audit.jsonl"), Some(&anchors), None).unwrap();
        assert!(!r.ok);
        assert_eq!(r.class, Some(TamperClass::RollbackVsAnchor));
        detected += 1;
    }

    // TA3: adjacent swaps.
    for pos in [2usize, 60] {
        total += 1;
        let dir = tempfile::tempdir().unwrap();
        build_clean_log(dir.path(), n, 25);
        let mut ls = lines(dir.path());
        ls.swap(pos, pos + 1);
        write(dir.path(), &ls);
        let r = verify_log(&dir.path().join("audit.jsonl"), None, None).unwrap();
        assert!(!r.ok, "TA3 swap at {pos}");
        detected += 1;
    }

    // TA4: self-consistent forgery (rebuilt hash, no signature).
    // Log is Mode::ChainedSigned. The forged entry has an empty sig, so the
    // verifier catches it at seq 10 via BadSignature (Step 7 fires before the
    // downstream ChainBreak at seq 11 that would follow in an unsigned log).
    {
        total += 1;
        let dir = tempfile::tempdir().unwrap();
        build_clean_log(dir.path(), n, 25);
        let mut ls = lines(dir.path());
        let e = parse_line(&ls[10]).unwrap();
        let (forged, _) = qfire::audit::chain::build_line(
            e.seq,
            &e.ts,
            EntryKind::Decision,
            "{\"event_id\":\"forged\",\"decision\":\"allow\"}",
            &e.prev_hash,
            None,
        )
        .unwrap(); // build_line returns Result<(String, String)>
        ls[10] = forged;
        write(dir.path(), &ls);
        let r = verify_log(&dir.path().join("audit.jsonl"), None, None).unwrap();
        assert!(!r.ok);
        // BadSignature: the forged entry is unsigned but the log is ChainedSigned,
        // so the signature layer catches it at its own seq before ChainBreak at 11.
        assert_eq!(
            r.first_bad_seq,
            Some(10),
            "forged entry caught at its own seq (BadSignature)"
        );
        detected += 1;
    }

    // TA6: rollback with stale anchors.
    {
        total += 1;
        let dir = tempfile::tempdir().unwrap();
        build_clean_log(dir.path(), n, 10);
        let ls = lines(dir.path());
        write(dir.path(), &ls[..50].to_vec());
        let anchors = FileAnchorSink::read_all(dir.path().join("anchors.jsonl")).unwrap();
        let r = verify_log(&dir.path().join("audit.jsonl"), Some(&anchors), None).unwrap();
        assert!(!r.ok);
        assert_eq!(r.class, Some(TamperClass::RollbackVsAnchor));
        detected += 1;
    }

    assert_eq!(detected, total, "E1 detection completeness must be 100%");
}

/// Fixture exporter for tamper.py: `cargo test --test audit_tamper -- --ignored export_fixture`
#[test]
#[ignore]
fn export_fixture() {
    let out = std::path::PathBuf::from("datasets/003-tamper-audit/fixture");
    std::fs::create_dir_all(&out).unwrap();
    build_clean_log(&out, 1000, 100);
    println!("clean fixture at {}", out.display());
}

/// Parameterized exporter for E3: TB_OUT, TB_N, TB_K env vars.
/// Usage: cargo test --release --quiet --test audit_tamper -- --ignored --exact export_fixture_param
#[test]
#[ignore]
fn export_fixture_param() {
    let out = std::path::PathBuf::from(std::env::var("TB_OUT").expect("TB_OUT"));
    let n: usize = std::env::var("TB_N")
        .map(|v| v.parse().unwrap())
        .unwrap_or(1000);
    let k: u64 = std::env::var("TB_K")
        .map(|v| v.parse().unwrap())
        .unwrap_or(100);
    std::fs::create_dir_all(&out).unwrap();
    build_clean_log(&out, n, k);
}
