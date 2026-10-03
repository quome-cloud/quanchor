//! Deterministic risk/autonomy classifier. `classify` is a pure function of
//! [`AgentMetadata`] and a [`Jurisdiction`]; it returns the cleared [`RiskClass`]
//! and [`AutonomyLevel`] plus a citation-tagged `rationale` trace (explainability
//! for the regulatory reviewer).
//!
//! **The rules are author-encoded from published criteria** — not a live expert
//! panel — and each rule cites its source. The base risk spine is the IMDRF SaMD
//! risk-categorization grid (WG/N12); each jurisdiction applies a documented
//! overlay that may move risk up or down. Those overlays are the *intended*,
//! explainable source of cross-jurisdiction divergence (paper H3): FDA's non-device
//! CDS carve-out can lower risk, the EU AI Act's blanket treatment of clinical
//! decision AI can raise it, Health Canada adopts IMDRF verbatim, and the MHRA is
//! IMDRF-aligned but stricter on patient-facing decision support.
//!
//! The rule set is **version-pinned** ([`super::RULESET_VERSION`]) so "regulatory
//! rules evolve" is handled by pinning rather than silent drift.

use super::{
    AgentMetadata, AutonomyLevel, Classification, ClinicalTask, Jurisdiction, OutputType,
    RiskClass, Severity, RULESET_VERSION,
};

/// IMDRF SaMD risk category (I–IV) from the significance-of-information ×
/// state-of-healthcare-situation grid (WG/N12). Returns the framework-agnostic
/// spine risk class plus the firing citation.
fn imdrf_spine(task: ClinicalTask, sev: Severity) -> (RiskClass, &'static str) {
    use ClinicalTask::*;
    use RiskClass::*;
    use Severity::*;
    let cat = match (task, sev) {
        // Treat or diagnose — highest significance row.
        (Treat | Diagnose, Critical) => High,     // IMDRF IV
        (Treat | Diagnose, Serious) => Moderate,  // IMDRF III
        (Treat | Diagnose, NonSerious) => Low,     // IMDRF II
        // Drive clinical management — middle row.
        (Drive, Critical) => Moderate, // IMDRF III
        (Drive, Serious) => Low,        // IMDRF II
        (Drive, NonSerious) => Minimal, // IMDRF I
        // Inform clinical management — lowest significance row.
        (Inform, Critical) => Low,      // IMDRF II
        (Inform, Serious) => Minimal,   // IMDRF I
        (Inform, NonSerious) => Minimal, // IMDRF I
    };
    (cat, "IMDRF SaMD WG/N12 risk categorization (significance × healthcare situation)")
}

/// Apply the jurisdiction overlay to the spine risk class. Returns the adjusted
/// class and a rationale line describing the overlay (empty string = no change).
fn overlay(j: Jurisdiction, base: RiskClass, meta: &AgentMetadata) -> (RiskClass, String) {
    use ClinicalTask::*;
    use RiskClass::*;
    match j {
        // Health Canada adopts the IMDRF N12 categorization directly — no overlay.
        Jurisdiction::HealthCanada => (
            base,
            "Health Canada SaMD guidance adopts IMDRF N12 categorization verbatim".into(),
        ),

        // FDA: a non-device CDS carve-out (21st Century Cures §3060 / FDA CDS
        // guidance 2022) can *lower* risk — software that informs or drives
        // management by *recommending* to a clinician (not acting or diagnosing),
        // is not patient-facing, and addresses a non-time-critical situation so the
        // clinician can independently review the basis is non-device CDS.
        Jurisdiction::Fda => {
            let non_device_cds = matches!(meta.clinical_task, Inform | Drive)
                && meta.output_type != OutputType::Action
                && !meta.patient_facing
                && meta.condition_severity != Severity::Critical;
            if non_device_cds && base > Minimal {
                (Minimal, "FDA non-device CDS carve-out (21st C. Cures §3060 / CDS Guidance 2022): informs only, clinician can independently review".into())
            } else {
                (base, "FDA SaMD: IMDRF spine, no CDS carve-out applies".into())
            }
        }

        // EU AI Act: clinical decision AI that diagnoses, treats, or *acts*, or any
        // AI that is an MDR safety component, is high-risk under Annex III — this can
        // *raise* risk above the spine.
        Jurisdiction::EuAiAct => {
            let annex_iii = matches!(meta.clinical_task, Diagnose | Treat)
                || meta.output_type == OutputType::Action;
            if annex_iii && base < High {
                (High, "EU AI Act Annex III(5)/MDR safety component: clinical decision or actuating AI is high-risk".into())
            } else {
                (base, "EU AI Act: IMDRF spine, Annex III high-risk trigger not met".into())
            }
        }

        // MHRA: IMDRF-aligned (no broad CDS carve-out like the FDA), and stricter on
        // patient-facing decision support — a patient-facing Drive/Diagnose/Treat at
        // the spine's minimal/low band is raised one step.
        Jurisdiction::Mhra => {
            let patient_facing_decision = meta.patient_facing
                && matches!(meta.clinical_task, Drive | Diagnose | Treat);
            if patient_facing_decision && base < Moderate {
                let raised = if base == Minimal { Low } else { Moderate };
                (raised, "MHRA Software & AI as a Medical Device programme: patient-facing decision support raised one band".into())
            } else {
                (base, "MHRA SaMD framework: IMDRF-aligned, no carve-out".into())
            }
        }
    }
}

/// Cap the declared autonomy by clinical consequence: a high-risk *actuating*
/// agent may not run fully autonomously — it is capped at `Supervised` so every
/// action keeps a human reviewer (C6.1). Returns the cleared level + rationale.
fn cap_autonomy(
    declared: AutonomyLevel,
    risk: RiskClass,
    output: OutputType,
) -> (AutonomyLevel, Option<String>) {
    let ceiling = if output == OutputType::Action && risk == RiskClass::High {
        AutonomyLevel::Supervised
    } else {
        AutonomyLevel::Autonomous
    };
    if declared.tier() > ceiling.tier() {
        (
            ceiling,
            Some(format!(
                "autonomy capped {:?}→{:?}: high-risk actuation requires per-action human review (C6.1)",
                declared, ceiling
            )),
        )
    } else {
        (declared, None)
    }
}

/// Classify an agent's metadata under one jurisdiction.
pub fn classify(meta: &AgentMetadata, j: Jurisdiction) -> Classification {
    let mut rationale = vec![format!("ruleset {RULESET_VERSION}, jurisdiction {}", j.name())];

    let (spine, spine_cite) = imdrf_spine(meta.clinical_task, meta.condition_severity);
    rationale.push(format!("spine {:?}: {}", spine, spine_cite));

    let (risk_class, overlay_cite) = overlay(j, spine, meta);
    rationale.push(format!("overlay → {:?}: {}", risk_class, overlay_cite));

    let (autonomy_level, cap_note) = cap_autonomy(meta.autonomy, risk_class, meta.output_type);
    if let Some(n) = cap_note {
        rationale.push(n);
    }

    Classification { risk_class, autonomy_level, rationale }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(
        task: ClinicalTask,
        output: OutputType,
        autonomy: AutonomyLevel,
        sev: Severity,
        patient_facing: bool,
    ) -> AgentMetadata {
        AgentMetadata {
            intended_use: "test".into(),
            clinical_task: task,
            output_type: output,
            autonomy,
            condition_severity: sev,
            patient_facing,
            tools: vec![],
        }
    }

    #[test]
    fn fda_high_risk_treat_critical() {
        let m = meta(
            ClinicalTask::Treat,
            OutputType::Action,
            AutonomyLevel::Autonomous,
            Severity::Critical,
            true,
        );
        let c = classify(&m, Jurisdiction::Fda);
        assert_eq!(c.risk_class, RiskClass::High);
        assert!(c.autonomy_level >= AutonomyLevel::Supervised);
        assert!(!c.rationale.is_empty());
    }

    #[test]
    fn eu_annex_iii_raises_above_fda() {
        // Diagnose / recommendation / serious: spine = Moderate. EU Annex III raises
        // to High; FDA keeps the spine (Moderate) — a documented divergence.
        let m = meta(
            ClinicalTask::Diagnose,
            OutputType::Recommendation,
            AutonomyLevel::Advisory,
            Severity::Serious,
            false,
        );
        let eu = classify(&m, Jurisdiction::EuAiAct);
        let fda = classify(&m, Jurisdiction::Fda);
        assert_eq!(eu.risk_class, RiskClass::High);
        assert_eq!(fda.risk_class, RiskClass::Moderate);
    }

    #[test]
    fn fda_cds_carveout_lowers_below_spine() {
        // Drive / recommendation / serious, not patient-facing: spine = Low (IMDRF
        // II). FDA's non-device CDS carve-out drops it to Minimal because the
        // clinician can independently review a non-time-critical recommendation;
        // Health Canada keeps the pure IMDRF spine (Low).
        let m = meta(
            ClinicalTask::Drive,
            OutputType::Recommendation,
            AutonomyLevel::Advisory,
            Severity::Serious,
            false,
        );
        let fda = classify(&m, Jurisdiction::Fda);
        let hc = classify(&m, Jurisdiction::HealthCanada);
        assert_eq!(fda.risk_class, RiskClass::Minimal); // carve-out
        assert_eq!(hc.risk_class, RiskClass::Low); // pure spine
    }

    #[test]
    fn minimal_informational_agrees_everywhere() {
        let m = meta(
            ClinicalTask::Inform,
            OutputType::Information,
            AutonomyLevel::Informational,
            Severity::NonSerious,
            false,
        );
        for j in Jurisdiction::ALL {
            let c = classify(&m, j);
            assert_eq!(c.risk_class, RiskClass::Minimal, "jurisdiction {:?}", j);
            assert_eq!(c.autonomy_level, AutonomyLevel::Informational);
        }
    }

    #[test]
    fn autonomy_capped_by_action_risk() {
        // Declared Autonomous, but high-risk action ⇒ capped down to Supervised.
        let m = meta(
            ClinicalTask::Treat,
            OutputType::Action,
            AutonomyLevel::Autonomous,
            Severity::Critical,
            false,
        );
        let c = classify(&m, Jurisdiction::Fda);
        assert_eq!(c.autonomy_level, AutonomyLevel::Supervised);
        assert!(c.rationale.iter().any(|r| r.contains("capped")));
    }

    #[test]
    fn health_canada_is_pure_spine() {
        let m = meta(
            ClinicalTask::Drive,
            OutputType::Recommendation,
            AutonomyLevel::Advisory,
            Severity::Critical,
            false,
        );
        // spine for Drive/Critical = Moderate; HC keeps it.
        assert_eq!(classify(&m, Jurisdiction::HealthCanada).risk_class, RiskClass::Moderate);
    }
}
