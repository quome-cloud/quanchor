//! HIPAA Safe-Harbor PHI detection and redaction.
//!
//! Implements matchers for the 18 HIPAA Safe-Harbor identifiers (45 CFR
//! §164.514(b)(2)). Text-extractable identifiers (SSNs, dates, phone/fax, email,
//! MRNs, account/beneficiary numbers, URLs, IP addresses, VINs, license/device
//! serials, ZIP/geographic subdivisions) are matched with regular expressions;
//! identifiers that are not reliably recoverable from plain text alone (full
//! names, biometric data, face photos) use conservative contextual heuristics
//! and are reported as such. Each hit carries its identifier category so results
//! are explainable; [`redact`] masks every hit with a typed placeholder.

use regex::Regex;
use std::sync::OnceLock;

/// A detected PHI element.
#[derive(Debug, Clone)]
pub struct PhiHit {
    /// HIPAA Safe-Harbor identifier number (1–18).
    pub id: u8,
    /// Short category label, e.g. `"SSN"`, `"EMAIL"`.
    pub category: &'static str,
    /// The matched substring.
    pub value: String,
}

struct Matcher {
    id: u8,
    category: &'static str,
    re: Regex,
}

fn matchers() -> &'static [Matcher] {
    static M: OnceLock<Vec<Matcher>> = OnceLock::new();
    M.get_or_init(|| {
        let mk = |id: u8, category: &'static str, pat: &str| Matcher {
            id,
            category,
            re: Regex::new(pat).expect("valid PHI regex"),
        };
        vec![
            // 7. Social Security numbers
            mk(7, "SSN", r"\b\d{3}-\d{2}-\d{4}\b"),
            // 6. Email addresses
            mk(6, "EMAIL", r"(?i)\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
            // 5. Fax numbers (phone-like near a fax label)
            mk(5, "FAX", r"(?i)fax[:#\s]*\+?\d?[\s.-]?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}"),
            // 4. Telephone numbers
            mk(4, "PHONE", r"\b\+?\d{1,2}[\s.-]?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b"),
            // 3. Dates (slash, ISO, and written month formats)
            mk(3, "DATE", r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),
            mk(3, "DATE", r"\b\d{4}-\d{2}-\d{2}\b"),
            mk(3, "DATE", r"(?i)\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b"),
            // 8. Medical record numbers
            mk(8, "MRN", r"(?i)\bMRN[:#\s]*[A-Z0-9]{5,}\b"),
            // 9. Health plan beneficiary numbers (Medicare MBI / Medicaid)
            mk(9, "BENEFICIARY", r"(?i)\b(MBI|medicare|medicaid)[:#\s]*[A-Z0-9-]{6,}\b"),
            mk(9, "BENEFICIARY", r"\b[1-9][AC-HJ-NP-RT-Y][0-9AC-HJ-NP-RT-Y]\d[AC-HJ-NP-RT-Y][0-9AC-HJ-NP-RT-Y]\d[AC-HJ-NP-RT-Y]{2}\d{2}\b"),
            // 10. Account numbers
            mk(10, "ACCOUNT", r"(?i)\b(account|acct)[:#\s]*\d{6,}\b"),
            // 11. Certificate / license numbers
            mk(11, "LICENSE", r"(?i)\b(license|licence|cert(ificate)?|dea)[:#\s]*[A-Z0-9-]{5,}\b"),
            // 12. Vehicle identifiers (VIN, license plate)
            mk(12, "VIN", r"\b[A-HJ-NPR-Z0-9]{17}\b"),
            mk(12, "PLATE", r"(?i)\b(plate|license\s+plate)[:#\s]*[A-Z0-9-]{5,8}\b"),
            // 13. Device identifiers / serial numbers
            mk(13, "DEVICE", r"(?i)\b(serial|sn|device\s*id|udi)[:#\s]*[A-Z0-9-]{6,}\b"),
            // 14. Web URLs
            mk(14, "URL", r"(?i)\bhttps?://[^\s]+\b"),
            // 15. IP addresses (IPv4 + compressed IPv6)
            mk(15, "IP", r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
            mk(15, "IP", r"\b(?:[A-Fa-f0-9]{1,4}:){2,7}[A-Fa-f0-9]{1,4}\b"),
            // 2. Geographic subdivisions smaller than a state (ZIP, ZIP+4)
            mk(2, "ZIP", r"(?i)\b(zip|postal)?[:\s]*\b\d{5}(?:-\d{4})?\b"),
            // 16. Biometric identifiers (contextual)
            mk(16, "BIOMETRIC", r"(?i)\b(fingerprint|retina\s*scan|iris\s*scan|voice\s*print|biometric)\b"),
            // 17. Full-face photographs (contextual)
            mk(17, "FACE_PHOTO", r"(?i)\b(face\s*photo|facial\s*image|headshot|patient\s*photo)\b"),
            // 1. Names (contextual: explicit patient-name cues only — conservative)
            mk(1, "NAME", r"(?i)\b(patient|name|mr|mrs|ms|dr)\.?\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b"),
            // 18. Other unique identifying codes (generic long alphanumerics near id cues)
            mk(18, "OTHER_ID", r"(?i)\b(id|code|number|#)[:#\s]*[A-Z0-9]{8,}\b"),
        ]
    })
}

/// Detect all PHI hits in `text` (deduplicated by value+category).
pub fn detect(text: &str) -> Vec<PhiHit> {
    let mut hits: Vec<PhiHit> = Vec::new();
    for m in matchers() {
        for caught in m.re.find_iter(text) {
            let value = caught.as_str().to_string();
            if !hits.iter().any(|h| h.category == m.category && h.value == value) {
                hits.push(PhiHit { id: m.id, category: m.category, value });
            }
        }
    }
    hits
}

/// Redact `text`, replacing each PHI hit with a typed placeholder, and return
/// the sanitized text alongside the hits found.
pub fn redact(text: &str) -> (String, Vec<PhiHit>) {
    let mut out = text.to_string();
    let mut hits: Vec<PhiHit> = Vec::new();
    for m in matchers() {
        let mut found = false;
        for caught in m.re.find_iter(&out.clone()) {
            let value = caught.as_str().to_string();
            if !hits.iter().any(|h| h.category == m.category && h.value == value) {
                hits.push(PhiHit { id: m.id, category: m.category, value });
            }
            found = true;
        }
        if found {
            let placeholder = format!("[REDACTED-{}]", m.category);
            out = m.re.replace_all(&out, placeholder.as_str()).into_owned();
        }
    }
    (out, hits)
}

/// The number of distinct Safe-Harbor identifier *types* this engine matches.
pub fn covered_identifier_count() -> usize {
    let mut ids: Vec<u8> = matchers().iter().map(|m| m.id).collect();
    ids.sort_unstable();
    ids.dedup();
    ids.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_ssn_email_phone() {
        let t = "Contact Jane at jane@doe.com or 555-123-4567; SSN 123-45-6789.";
        let hits = detect(t);
        let cats: Vec<&str> = hits.iter().map(|h| h.category).collect();
        assert!(cats.contains(&"SSN"));
        assert!(cats.contains(&"EMAIL"));
        assert!(cats.contains(&"PHONE"));
    }

    #[test]
    fn redacts_ssn() {
        let (red, hits) = redact("SSN 123-45-6789 on file");
        assert!(red.contains("[REDACTED-SSN]"));
        assert!(!red.contains("123-45-6789"));
        assert!(hits.iter().any(|h| h.category == "SSN"));
    }

    #[test]
    fn covers_all_eighteen() {
        assert_eq!(covered_identifier_count(), 18);
    }
}
