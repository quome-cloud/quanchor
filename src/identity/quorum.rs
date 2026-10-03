//! k-of-n quorum over distinct, attested identities (defeats manufactured quorum,
//! T4). Approvals only count from agents present in the registry, and duplicates
//! count once — so spawning Sybils cannot satisfy a quorum.

use crate::identity::AgentId;
use std::collections::HashSet;

/// Returns Ok if at least `k` *distinct, registered* approvers signed off.
/// `is_registered` is the registry membership predicate.
pub fn check_quorum<F>(approvers: &[AgentId], k: usize, is_registered: F) -> crate::Result<()>
where
    F: Fn(&AgentId) -> bool,
{
    let distinct: HashSet<&AgentId> = approvers.iter().filter(|a| is_registered(a)).collect();
    if distinct.len() >= k {
        Ok(())
    } else {
        Err(crate::error::Error::Config(format!(
            "quorum not met: {} distinct registered approvers, need {}", distinct.len(), k)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(v: &[&str]) -> Vec<AgentId> { v.iter().map(|s| AgentId(s.to_string())).collect() }

    #[test]
    fn distinct_registered_meets_quorum() {
        let reg = |a: &AgentId| ["a1", "a2", "a3"].contains(&a.0.as_str());
        assert!(check_quorum(&ids(&["a1", "a2"]), 2, reg).is_ok());
    }

    #[test]
    fn duplicates_do_not_count() {
        let reg = |a: &AgentId| ["a1", "a2", "a3"].contains(&a.0.as_str());
        assert!(check_quorum(&ids(&["a1", "a1"]), 2, reg).is_err());
    }

    #[test]
    fn unregistered_approvers_do_not_count() {
        let reg = |a: &AgentId| ["a1"].contains(&a.0.as_str());
        assert!(check_quorum(&ids(&["a1", "ghost", "ghost2"]), 2, reg).is_err());
    }
}
