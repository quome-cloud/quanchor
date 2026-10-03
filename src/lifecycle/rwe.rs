//! Real-world-evidence ingest — the 011→012 loop on the evidence side. Paper 011's
//! `RweExporter` emits per-detector evidence records (drift episodes detected,
//! false-alarm rate, estimated effect size, and — for the autonomy meter — the
//! monitored autonomous effect). `ingest_rwe` reads that bundle and maps it to a
//! lifecycle re-assessment signal: a deployed agent that has accumulated field
//! evidence of drift or whose monitored autonomy approaches the passport's
//! authorized envelope is recommended for re-review (the PCCP's trigger for
//! re-submission). Tolerant parsing (`serde_json::Value`) so it survives 011 schema
//! evolution.

use super::passport::Passport;
use serde::{Deserialize, Serialize};

/// The lifecycle re-assessment recommendation derived from field evidence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RweAssessment {
    pub recommend_rereview: bool,
    pub reasons: Vec<String>,
}

/// Parse an 011 RWE bundle (a JSON array of evidence records, or an object with an
/// `evidence` array) and assess it against the passport's autonomy envelope.
pub fn ingest_rwe(bundle_json: &str, passport: &Passport) -> crate::Result<RweAssessment> {
    let v: serde_json::Value = serde_json::from_str(bundle_json)
        .map_err(|e| crate::error::Error::Config(format!("malformed RWE bundle: {e}")))?;
    let records = match &v {
        serde_json::Value::Array(a) => a.clone(),
        serde_json::Value::Object(o) => o
            .get("evidence")
            .and_then(|e| e.as_array())
            .cloned()
            .ok_or_else(|| crate::error::Error::Config("RWE bundle object lacks `evidence` array".into()))?,
        _ => {
            return Err(crate::error::Error::Config(
                "RWE bundle must be an array or {evidence:[...]}".into(),
            ))
        }
    };

    let envelope_frac = passport.autonomy_envelope.max_autonomous_fraction;
    let mut reasons = Vec::new();
    for r in &records {
        let detector = r.get("detector").and_then(|d| d.as_str()).unwrap_or("?");
        let episodes = r.get("episodes_detected").and_then(|e| e.as_u64()).unwrap_or(0);
        let effect = r.get("estimated_effect").and_then(|e| e.as_f64()).unwrap_or(0.0);

        if episodes > 0 {
            reasons.push(format!(
                "detector `{detector}` reported {episodes} drift/creep episode(s) in the field"
            ));
        }
        // Autonomy creep approaching the passport's authorized envelope.
        if detector.contains("autonomy") && effect >= envelope_frac {
            reasons.push(format!(
                "monitored autonomous fraction {effect:.3} ≥ passport envelope {envelope_frac:.3}"
            ));
        }
    }

    Ok(RweAssessment { recommend_rereview: !reasons.is_empty(), reasons })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lifecycle::fingerprint::Fingerprint;
    use crate::lifecycle::passport::EnvelopeSpec;
    use crate::lifecycle::pccp::Pccp;
    use crate::lifecycle::{AgentMetadata, AutonomyLevel, ClinicalTask, OutputType, Severity};

    fn passport() -> Passport {
        Passport {
            agent_id: "a".into(),
            version: "1".into(),
            ruleset_version: "2026.06".into(),
            metadata: AgentMetadata {
                intended_use: "x".into(),
                clinical_task: ClinicalTask::Drive,
                output_type: OutputType::Recommendation,
                autonomy: AutonomyLevel::Advisory,
                condition_severity: Severity::Serious,
                patient_facing: false,
                tools: vec![],
            },
            classifications: vec![],
            pccp: Pccp { allowed: vec![] },
            deployed_fingerprint: Fingerprint::of(b"w", b"p", b"t", b"d"),
            not_after: 10_000,
            autonomy_envelope: EnvelopeSpec {
                max_autonomous_risk_tier: 2,
                max_autonomous_fraction: 0.30,
                window: 100,
            },
        }
    }

    #[test]
    fn detected_episode_recommends_rereview() {
        let bundle = r#"[{"detector":"cusum","episodes_detected":3,"far":0.01,"estimated_effect":0.4}]"#;
        let a = ingest_rwe(bundle, &passport()).unwrap();
        assert!(a.recommend_rereview);
        assert!(a.reasons.iter().any(|r| r.contains("cusum")));
    }

    #[test]
    fn autonomy_near_envelope_recommends_rereview() {
        let bundle =
            r#"[{"detector":"autonomy_fraction","episodes_detected":0,"estimated_effect":0.35}]"#;
        let a = ingest_rwe(bundle, &passport()).unwrap();
        assert!(a.recommend_rereview);
        assert!(a.reasons.iter().any(|r| r.contains("envelope")));
    }

    #[test]
    fn quiet_field_evidence_no_rereview() {
        let bundle =
            r#"[{"detector":"autonomy_fraction","episodes_detected":0,"estimated_effect":0.10}]"#;
        let a = ingest_rwe(bundle, &passport()).unwrap();
        assert!(!a.recommend_rereview);
        assert!(a.reasons.is_empty());
    }

    #[test]
    fn object_wrapped_evidence_array_ok() {
        let bundle = r#"{"evidence":[{"detector":"adwin","episodes_detected":1}]}"#;
        assert!(ingest_rwe(bundle, &passport()).unwrap().recommend_rereview);
    }

    #[test]
    fn malformed_bundle_errs() {
        assert!(ingest_rwe("not json", &passport()).is_err());
        assert!(ingest_rwe("12", &passport()).is_err());
    }
}
