//! equity::flagger — vulnerable-population protective routing (E4 / H4). Tags a
//! case as belonging to a vulnerable population (pediatric dosing, pregnancy
//! contraindication, geriatric polypharmacy, low-resource default) and, under the
//! protective policy, forces it to human review (the 005 `Escalate` path; real
//! routing is 009). Harm is an unsafe default action that the protective policy
//! would have caught.

use super::{Action, DecisionRecord};

/// The vulnerable-population tags. Stable string identifiers (match the dataset).
pub const TAGS: [&str; 4] = [
    "pediatric_dosing",
    "pregnancy_contraindication",
    "geriatric_polypharmacy",
    "low_resource_default",
];

/// Keyword-flag a free-text clinical request as a vulnerable-population case.
/// Returns the first matching tag, or None.
pub fn flag_text(prompt: &str) -> Option<&'static str> {
    let p = prompt.to_lowercase();
    let has = |needles: &[&str]| needles.iter().any(|n| p.contains(n));
    if has(&["pediatric", "infant", "neonate", "newborn", "child dose", "weight-based dose", "mg/kg"]) {
        Some("pediatric_dosing")
    } else if has(&["pregnan", "gestation", "fetal", "teratogen", "breastfeeding", "lactation"]) {
        Some("pregnancy_contraindication")
    } else if has(&["geriatric", "elderly", "polypharmacy", "beers criteria", "renal dosing in the elderly"]) {
        Some("geriatric_polypharmacy")
    } else if has(&["low-resource", "low resource", "no lab", "without labs", "rural clinic", "resource-limited"]) {
        Some("low_resource_default")
    } else {
        None
    }
}

pub struct VulnerableFlagger;

impl VulnerableFlagger {
    /// A record is vulnerable if it carries a `vulnerable` tag (dataset-supplied)
    /// or its text flags one. Returns the tag.
    pub fn tag(rec: &DecisionRecord, prompt: Option<&str>) -> Option<String> {
        if let Some(t) = &rec.vulnerable {
            return Some(t.clone());
        }
        prompt.and_then(flag_text).map(|s| s.to_string())
    }

    /// Apply the protective policy: a vulnerable case is escalated to human review
    /// regardless of the baseline action; a non-vulnerable case keeps its baseline.
    pub fn protect(rec: &DecisionRecord, baseline: Action, prompt: Option<&str>) -> Action {
        if Self::tag(rec, prompt).is_some() {
            Action::Escalate
        } else {
            baseline
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn vuln(tag: Option<&str>) -> DecisionRecord {
        DecisionRecord {
            case_id: "v".into(),
            group: BTreeMap::new(),
            action: Action::Allow,
            correct: false,
            label: false,
            score: None,
            vulnerable: tag.map(|s| s.to_string()),
        }
    }

    #[test]
    fn flags_each_population_from_text() {
        assert_eq!(flag_text("Calculate the mg/kg pediatric dose"), Some("pediatric_dosing"));
        assert_eq!(flag_text("Patient is 12 weeks pregnant"), Some("pregnancy_contraindication"));
        assert_eq!(flag_text("Geriatric patient with polypharmacy"), Some("geriatric_polypharmacy"));
        assert_eq!(flag_text("Rural clinic with no lab access"), Some("low_resource_default"));
        assert_eq!(flag_text("Order acetaminophen for an adult headache"), None);
    }

    #[test]
    fn protective_policy_escalates_vulnerable_cases() {
        let rec = vuln(Some("pediatric_dosing"));
        assert_eq!(VulnerableFlagger::protect(&rec, Action::Allow, None), Action::Escalate);
    }

    #[test]
    fn protective_policy_passes_through_non_vulnerable() {
        let rec = vuln(None);
        assert_eq!(VulnerableFlagger::protect(&rec, Action::Allow, None), Action::Allow);
        assert_eq!(VulnerableFlagger::protect(&rec, Action::Deny, None), Action::Deny);
    }

    #[test]
    fn dataset_tag_takes_priority_over_text() {
        let rec = vuln(Some("pregnancy_contraindication"));
        assert_eq!(
            VulnerableFlagger::tag(&rec, Some("ordinary adult request")),
            Some("pregnancy_contraindication".to_string())
        );
    }
}
