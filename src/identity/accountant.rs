//! Fleet-global capability accountant (defeats Sybil/collusion, T4). A budget is
//! keyed by *action-class*, not by agent, so an adversary that spawns N agents to
//! split a forbidden action across principals still exhausts the single shared
//! budget. Enforced at the gateway.

use crate::identity::Capability;
use std::collections::HashMap;

pub struct Accountant {
    /// action-class -> (used, limit)
    budgets: HashMap<String, (u32, u32)>,
}

impl Accountant {
    /// Build from action-class -> limit.
    pub fn new(limits: HashMap<String, u32>) -> Self {
        Accountant { budgets: limits.into_iter().map(|(k, v)| (k, (0, v))).collect() }
    }

    /// Charge one use of `action` to the fleet budget. Ok(()) if within budget
    /// (and increments); Err if it would exceed (fail-closed). Unbudgeted
    /// action-classes are unlimited (return Ok without tracking).
    pub fn charge(&mut self, action: &Capability) -> crate::Result<()> {
        match self.budgets.get_mut(&action.0) {
            None => Ok(()),
            Some((used, limit)) => {
                if *used >= *limit {
                    return Err(crate::error::Error::Config(format!(
                        "fleet budget exhausted for action-class '{}'", action.0)));
                }
                *used += 1;
                Ok(())
            }
        }
    }

    pub fn used(&self, action: &str) -> u32 {
        self.budgets.get(action).map(|(u, _)| *u).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn acct(action: &str, limit: u32) -> Accountant {
        Accountant::new(HashMap::from([(action.to_string(), limit)]))
    }

    #[test]
    fn within_budget_allows() {
        let mut a = acct("order_medication", 2);
        assert!(a.charge(&Capability("order_medication".into())).is_ok());
        assert!(a.charge(&Capability("order_medication".into())).is_ok());
    }

    #[test]
    fn sybil_split_hits_shared_cap() {
        let mut a = acct("order_medication", 2);
        a.charge(&Capability("order_medication".into())).unwrap();
        a.charge(&Capability("order_medication".into())).unwrap();
        assert!(a.charge(&Capability("order_medication".into())).is_err());
    }

    #[test]
    fn unbudgeted_action_is_unlimited() {
        let mut a = acct("order_medication", 0);
        for _ in 0..100 { assert!(a.charge(&Capability("read_phi".into())).is_ok()); }
    }
}
