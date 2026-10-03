//! The harness subsystem: the attack + goal-integrity primitives that live at
//! QFIRE's prompt-assembly layer.
//!
//! - [`anchor`] — the sealed, signed goal anchor (the sanctioned MIIM).
//! - [`provenance`] — the hash-chained ledger of prompt mutations.
//! - [`mfi`] — the Moral-Filter Injection rewriting attack.

pub mod ace;
pub mod anchor;
pub mod engine_hook;
pub mod intent_eval;
pub mod intent_run;
pub mod llm_rewriter;
pub mod mfi;
pub mod provenance;
