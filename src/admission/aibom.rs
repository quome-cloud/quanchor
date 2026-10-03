//! The AIBOM (AI bill of materials): enumerate every loaded component, hash it,
//! and serialize a CycloneDX 1.5 JSON document with a clinical-agent profile.

use crate::Result;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::Path;

/// The QUOKKAGUARD component class (stamped as a CycloneDX property).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComponentClass {
    Rule,
    Chain,
    Config,
    Model,
    Library,
    Provider,
    Mcp,
    Data,
}

/// One enumerated supply-chain component.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Component {
    pub class: ComponentClass,
    pub name: String,
    /// Version / identity string (e.g. a crate version, a model id, or "sha256:..").
    pub version: String,
    /// Lowercase hex SHA-256 of the artifact bytes, when a digest is computable.
    pub digest: Option<String>,
    /// Whether this component carries a verifiable attestation (digest or signature).
    pub attested: bool,
}

/// A complete AIBOM.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Aibom {
    pub components: Vec<Component>,
}

/// Lowercase hex SHA-256 of a byte slice.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    hex::encode(h.finalize())
}

/// SHA-256 of a file's contents, or None if unreadable.
pub fn sha256_file(path: &Path) -> Option<String> {
    std::fs::read(path).ok().map(|b| sha256_hex(&b))
}

impl Aibom {
    /// Enumerate the gateway's loaded components from configured paths.
    /// `rules_dir`/`chains_dir` are walked for *.yaml; `config_path` and
    /// `onnx_path` (if present) are hashed; `cargo_lock` is parsed for libraries;
    /// `provider_models` are recorded by declared identity.
    pub fn enumerate(
        rules_dir: &Path,
        chains_dir: &Path,
        config_path: Option<&Path>,
        onnx_path: Option<&Path>,
        cargo_lock: Option<&Path>,
        provider_models: &[String],
    ) -> Self {
        let mut components = Vec::new();

        let mut walk_yaml = |dir: &Path, class: ComponentClass| {
            for entry in walkdir::WalkDir::new(dir).into_iter().flatten() {
                let p = entry.path();
                if p.is_file() && p.extension().map(|e| e == "yaml").unwrap_or(false) {
                    let digest = sha256_file(p);
                    components.push(Component {
                        class,
                        name: p.to_string_lossy().to_string(),
                        version: digest
                            .as_ref()
                            .map(|d| format!("sha256:{}", &d[..16]))
                            .unwrap_or_else(|| "unknown".into()),
                        attested: digest.is_some(),
                        digest,
                    });
                }
            }
        };
        walk_yaml(rules_dir, ComponentClass::Rule);
        walk_yaml(chains_dir, ComponentClass::Chain);

        for (opt, class) in [
            (config_path, ComponentClass::Config),
            (onnx_path, ComponentClass::Model),
        ] {
            if let Some(p) = opt {
                if p.exists() {
                    let digest = sha256_file(p);
                    components.push(Component {
                        class,
                        name: p.to_string_lossy().to_string(),
                        version: digest
                            .as_ref()
                            .map(|d| format!("sha256:{}", &d[..16]))
                            .unwrap_or_else(|| "unknown".into()),
                        attested: digest.is_some(),
                        digest,
                    });
                }
            }
        }

        if let Some(lock) = cargo_lock {
            components.extend(enumerate_cargo_lock(lock));
        }

        for m in provider_models {
            // Provider model identity is declared, not file-hashable here (the
            // weights live in the provider runtime, e.g. Ollama). This is the
            // measured provenance gap: attested = false.
            components.push(Component {
                class: ComponentClass::Provider,
                name: m.clone(),
                version: m.clone(),
                digest: None,
                attested: false,
            });
        }

        Aibom { components }
    }

    /// Share of components with no verifiable attestation (the provenance gap).
    pub fn provenance_gap(&self) -> f64 {
        if self.components.is_empty() {
            return 0.0;
        }
        let unattested = self.components.iter().filter(|c| !c.attested).count();
        unattested as f64 / self.components.len() as f64
    }

    /// Serialize to a CycloneDX 1.5 JSON document (without signature; see attest.rs).
    pub fn to_cyclonedx(&self) -> serde_json::Value {
        let components: Vec<serde_json::Value> = self
            .components
            .iter()
            .map(|c| {
                let props = vec![
                    serde_json::json!({"name":"quokkaguard:class",
                    "value": serde_json::to_value(c.class).unwrap()}),
                    serde_json::json!({"name":"quokkaguard:attested",
                    "value": c.attested.to_string()}),
                ];
                let mut obj = serde_json::json!({
                    "type": cyclonedx_type(c.class),
                    "name": c.name,
                    "version": c.version,
                    "properties": props,
                });
                if let Some(d) = &c.digest {
                    obj["hashes"] = serde_json::json!([{"alg":"SHA-256","content":d}]);
                }
                obj
            })
            .collect();
        serde_json::json!({
            "bomFormat": "CycloneDX",
            "specVersion": "1.5",
            "metadata": {
                "component": {"type":"application","name":"quokkaguard"},
                "properties": [{"name":"quokkaguard:profile","value":"clinical-agent"}]
            },
            "components": components,
        })
    }

    /// The canonical digest of the AIBOM document (over the unsigned CycloneDX JSON).
    pub fn document_digest(&self) -> String {
        let doc = self.to_cyclonedx();
        sha256_hex(serde_json::to_string(&doc).unwrap().as_bytes())
    }

    /// Parse an Aibom back from a CycloneDX JSON document.
    pub fn from_cyclonedx(doc: &serde_json::Value) -> Result<Self> {
        let mut components = Vec::new();
        let arr = doc
            .get("components")
            .and_then(|c| c.as_array())
            .ok_or_else(|| crate::error::Error::Config("AIBOM has no components array".into()))?;
        for c in arr {
            let prop = |key: &str| -> Option<String> {
                c.get("properties")?
                    .as_array()?
                    .iter()
                    .find(|p| p.get("name").and_then(|n| n.as_str()) == Some(key))
                    .and_then(|p| p.get("value"))
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
            };
            let class: ComponentClass = serde_json::from_value(serde_json::Value::String(
                prop("quokkaguard:class").unwrap_or_default(),
            ))
            .map_err(|_| crate::error::Error::Config("AIBOM component missing class".into()))?;
            let digest = c
                .get("hashes")
                .and_then(|h| h.as_array())
                .and_then(|a| a.first())
                .and_then(|h| h.get("content"))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            components.push(Component {
                class,
                name: c
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                version: c
                    .get("version")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                attested: prop("quokkaguard:attested").as_deref() == Some("true"),
                digest,
            });
        }
        Ok(Aibom { components })
    }
}

fn cyclonedx_type(class: ComponentClass) -> &'static str {
    match class {
        ComponentClass::Library => "library",
        ComponentClass::Model | ComponentClass::Provider => "machine-learning-model",
        ComponentClass::Mcp => "application",
        _ => "data",
    }
}

/// Parse `Cargo.lock` (TOML) into library components (name + version + checksum).
pub fn enumerate_cargo_lock(path: &Path) -> Vec<Component> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    let Ok(doc) = text.parse::<toml::Value>() else {
        return Vec::new();
    };
    let Some(pkgs) = doc.get("package").and_then(|p| p.as_array()) else {
        return Vec::new();
    };
    pkgs.iter()
        .filter_map(|p| {
            let name = p.get("name")?.as_str()?.to_string();
            let version = p.get("version")?.as_str()?.to_string();
            let checksum = p
                .get("checksum")
                .and_then(|c| c.as_str())
                .map(|s| s.to_string());
            Some(Component {
                class: ComponentClass::Library,
                name,
                version,
                attested: checksum.is_some(), // registry checksum = a pinned digest
                digest: checksum,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::tempdir;

    fn write(dir: &Path, rel: &str, body: &str) -> std::path::PathBuf {
        let p = dir.join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        let mut f = std::fs::File::create(&p).unwrap();
        f.write_all(body.as_bytes()).unwrap();
        p
    }

    #[test]
    fn sha256_is_stable_and_sensitive() {
        assert_eq!(sha256_hex(b"abc"), sha256_hex(b"abc"));
        assert_ne!(sha256_hex(b"abc"), sha256_hex(b"abd"));
        assert_eq!(sha256_hex(b"abc").len(), 64);
    }

    #[test]
    fn enumerate_finds_yaml_and_libs() {
        let dir = tempdir().unwrap();
        write(dir.path(), "rules/a.yaml", "id: a");
        write(dir.path(), "rules/nested/b.yaml", "id: b");
        write(dir.path(), "chains/c.yaml", "name: c");
        let cfg = write(dir.path(), "qfire.toml", "x = 1");
        let lock = write(dir.path(), "Cargo.lock",
            "version = 3\n[[package]]\nname = \"serde\"\nversion = \"1.0\"\nchecksum = \"deadbeef\"\n");
        let a = Aibom::enumerate(
            &dir.path().join("rules"),
            &dir.path().join("chains"),
            Some(&cfg),
            None,
            Some(&lock),
            &["llama3.2".into()],
        );
        let by_class = |c: ComponentClass| a.components.iter().filter(|x| x.class == c).count();
        assert_eq!(by_class(ComponentClass::Rule), 2);
        assert_eq!(by_class(ComponentClass::Chain), 1);
        assert_eq!(by_class(ComponentClass::Config), 1);
        assert_eq!(by_class(ComponentClass::Library), 1);
        assert_eq!(by_class(ComponentClass::Provider), 1);
        // every rule/chain/config carries a digest; the provider does not.
        let rule = a
            .components
            .iter()
            .find(|c| c.class == ComponentClass::Rule)
            .unwrap();
        assert!(rule.digest.is_some() && rule.attested);
        let prov = a
            .components
            .iter()
            .find(|c| c.class == ComponentClass::Provider)
            .unwrap();
        assert!(prov.digest.is_none() && !prov.attested);
    }

    #[test]
    fn provenance_gap_counts_unattested() {
        let dir = tempdir().unwrap();
        write(dir.path(), "rules/a.yaml", "id: a");
        let a = Aibom::enumerate(
            &dir.path().join("rules"),
            &dir.path().join("chains"),
            None,
            None,
            None,
            &["m1".into(), "m2".into()],
        );
        // 1 attested rule + 2 unattested providers -> gap = 2/3.
        assert!((a.provenance_gap() - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn cyclonedx_roundtrips() {
        let dir = tempdir().unwrap();
        write(dir.path(), "rules/a.yaml", "id: a");
        let a = Aibom::enumerate(
            &dir.path().join("rules"),
            &dir.path().join("chains"),
            None,
            None,
            None,
            &["m1".into()],
        );
        let doc = a.to_cyclonedx();
        assert_eq!(doc["bomFormat"], "CycloneDX");
        assert_eq!(doc["metadata"]["properties"][0]["value"], "clinical-agent");
        let back = Aibom::from_cyclonedx(&doc).unwrap();
        assert_eq!(back.components.len(), a.components.len());
        assert_eq!(back.provenance_gap(), a.provenance_gap());
        // digest is stable across the roundtrip
        assert_eq!(
            Aibom::from_cyclonedx(&doc).unwrap().document_digest(),
            a.document_digest()
        );
    }
}
