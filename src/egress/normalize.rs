//! Decoders/normalizers applied to outbound arg strings before taint matching, so encoded PHI
//! (base64/hex/rot13) or lightly obfuscated PHI still matches the tainted source value.

use base64::{engine::general_purpose::STANDARD, Engine as _};

/// Candidate plaintexts derived from `s`: the raw string plus any successful base64/hex decodings,
/// a rot13 form, and a casefolded alphanumeric form. Taint matching tests every candidate.
pub fn candidates(s: &str) -> Vec<String> {
    let mut out = vec![s.to_string()];
    let t = s.trim();
    if t.len() >= 8 && t.len() % 4 == 0 {
        if let Ok(b) = STANDARD.decode(t) {
            if let Ok(d) = String::from_utf8(b) {
                if d.chars().all(|c| !c.is_control()) {
                    out.push(d);
                }
            }
        }
    }
    if let Some(h) = hex_decode(t) {
        out.push(h);
    }
    out.push(rot13(s));
    out.push(casefold_alnum(s));
    out
}

fn hex_decode(t: &str) -> Option<String> {
    if t.len() >= 6 && t.len() % 2 == 0 && t.chars().all(|c| c.is_ascii_hexdigit()) {
        let b: Vec<u8> = (0..t.len())
            .step_by(2)
            .filter_map(|i| u8::from_str_radix(&t[i..i + 2], 16).ok())
            .collect();
        return String::from_utf8(b).ok();
    }
    None
}

fn rot13(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            'a'..='z' => (((c as u8 - b'a' + 13) % 26) + b'a') as char,
            'A'..='Z' => (((c as u8 - b'A' + 13) % 26) + b'A') as char,
            _ => c,
        })
        .collect()
}

fn casefold_alnum(s: &str) -> String {
    s.chars().filter(|c| c.is_alphanumeric()).flat_map(|c| c.to_lowercase()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn decodes_base64_hex_rot13() {
        let secret = "123-45-6789";
        let b64 = base64::engine::general_purpose::STANDARD.encode(secret);
        assert!(candidates(&b64).iter().any(|c| c == secret), "base64 decoded");
        let hexs = secret.as_bytes().iter().map(|b| format!("{b:02x}")).collect::<String>();
        assert!(candidates(&hexs).iter().any(|c| c == secret), "hex decoded");
        assert!(candidates(&rot13(secret)).iter().any(|c| c == secret), "rot13 reversed");
    }
}
