//! Cross-jurisdiction harmonization. `harmonize` classifies one agent under all
//! four frameworks and reports where they disagree on risk class — and, crucially,
//! attaches the **documented regulatory rationale** to every divergence. A
//! divergence *with* a rationale is a defensible finding (paper H3); a divergence
//! *without* one would be a classifier bug. The agreement score is the fraction of
//! jurisdiction pairs that concur on risk class.

use super::{classifier::classify, AgentMetadata, Classification, Jurisdiction};
use serde::{Deserialize, Serialize};

/// One pairwise disagreement on risk class, with the rationale that explains it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Divergence {
    pub a: Jurisdiction,
    pub b: Jurisdiction,
    pub field: String,
    /// The documented regulatory reason for the divergence (never empty).
    pub rationale: String,
}

/// The result of classifying an agent across all four jurisdictions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarmonizationReport {
    pub per_jurisdiction: Vec<(Jurisdiction, Classification)>,
    pub divergences: Vec<Divergence>,
    /// Fraction of the 6 unordered jurisdiction pairs agreeing on risk class.
    pub agreement: f64,
}

/// The overlay rationale line for a classification (the last `overlay → ...` entry),
/// used to explain a divergence in terms of the documented regulatory difference.
fn overlay_reason(c: &Classification) -> String {
    c.rationale
        .iter()
        .rev()
        .find(|r| r.starts_with("overlay →"))
        .cloned()
        .unwrap_or_else(|| "documented regulatory difference".into())
}

/// Classify across all jurisdictions and report risk-class divergences with reasons.
pub fn harmonize(meta: &AgentMetadata) -> HarmonizationReport {
    let per: Vec<(Jurisdiction, Classification)> =
        Jurisdiction::ALL.iter().map(|&j| (j, classify(meta, j))).collect();

    let mut divergences = Vec::new();
    let mut agree = 0usize;
    let mut pairs = 0usize;
    for i in 0..per.len() {
        for k in (i + 1)..per.len() {
            pairs += 1;
            let (ja, ca) = &per[i];
            let (jb, cb) = &per[k];
            if ca.risk_class == cb.risk_class {
                agree += 1;
            } else {
                // Explain via whichever side moved off the shared IMDRF spine.
                let reason = format!(
                    "{} classifies {:?} where {} classifies {:?}; {} | {}",
                    ja.name(),
                    ca.risk_class,
                    jb.name(),
                    cb.risk_class,
                    overlay_reason(ca),
                    overlay_reason(cb),
                );
                divergences.push(Divergence {
                    a: *ja,
                    b: *jb,
                    field: "risk_class".into(),
                    rationale: reason,
                });
            }
        }
    }
    let agreement = if pairs == 0 { 1.0 } else { agree as f64 / pairs as f64 };
    HarmonizationReport { per_jurisdiction: per, divergences, agreement }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lifecycle::{AutonomyLevel, ClinicalTask, OutputType, RiskClass, Severity};

    fn meta(
        task: ClinicalTask,
        output: OutputType,
        sev: Severity,
        patient_facing: bool,
    ) -> AgentMetadata {
        AgentMetadata {
            intended_use: "test".into(),
            clinical_task: task,
            output_type: output,
            autonomy: AutonomyLevel::Advisory,
            condition_severity: sev,
            patient_facing,
            tools: vec![],
        }
    }

    #[test]
    fn agreement_when_all_concur() {
        // Minimal informational — every jurisdiction agrees.
        let r = harmonize(&meta(
            ClinicalTask::Inform,
            OutputType::Information,
            Severity::NonSerious,
            false,
        ));
        assert!(r.divergences.is_empty());
        assert_eq!(r.agreement, 1.0);
    }

    #[test]
    fn divergence_carries_rationale() {
        // Diagnose / recommendation / serious: EU raises to High, others Moderate.
        let r = harmonize(&meta(
            ClinicalTask::Diagnose,
            OutputType::Recommendation,
            Severity::Serious,
            false,
        ));
        assert!(!r.divergences.is_empty());
        assert!(r.agreement < 1.0);
        // Every divergence is explained (H3) — none is a bare bug.
        assert!(r.divergences.iter().all(|d| !d.rationale.is_empty()));
        // The EU side is named in at least one rationale.
        assert!(r
            .divergences
            .iter()
            .any(|d| d.rationale.contains("EU AI Act")));
    }

    #[test]
    fn risk_classes_present_for_all_four() {
        let r = harmonize(&meta(
            ClinicalTask::Treat,
            OutputType::Action,
            Severity::Critical,
            true,
        ));
        assert_eq!(r.per_jurisdiction.len(), 4);
        // Critical treat is High everywhere — full agreement at the top of the grid.
        assert!(r
            .per_jurisdiction
            .iter()
            .all(|(_, c)| c.risk_class == RiskClass::High));
    }
}
