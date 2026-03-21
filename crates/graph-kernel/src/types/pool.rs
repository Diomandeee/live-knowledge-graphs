//! Response Pool types for collaborative multi-model chaining (AAO W6).
//!
//! A "pool" allows multiple agents/models to contribute to the same task
//! collaboratively: one analyzes, another implements, a third reviews,
//! and a synthesis agent merges everything into a unified output.

use serde::{Deserialize, Serialize};

/// Strategy for how agents contribute to a pool.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PoolStrategy {
    /// A -> B -> C -> synthesis. Each sees all prior work.
    Sequential,
    /// A, B, C run simultaneously with same prompt -> synthesis merges.
    Parallel,
    /// A+B in parallel -> C reviews both -> synthesis. Default.
    Hybrid,
}

impl Default for PoolStrategy {
    fn default() -> Self {
        Self::Hybrid
    }
}

impl std::fmt::Display for PoolStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Sequential => write!(f, "sequential"),
            Self::Parallel => write!(f, "parallel"),
            Self::Hybrid => write!(f, "hybrid"),
        }
    }
}

/// Phase lifecycle for a pool session.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PoolPhase {
    /// Accepting contributions.
    Open,
    /// All expected contributions received, awaiting synthesis trigger.
    Collecting,
    /// Synthesis agent is building the merged output.
    Synthesizing,
    /// Pool closed with a final result.
    Closed,
    /// TTL expired before completion.
    Expired,
}

impl Default for PoolPhase {
    fn default() -> Self {
        Self::Open
    }
}

impl std::fmt::Display for PoolPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Open => write!(f, "open"),
            Self::Collecting => write!(f, "collecting"),
            Self::Synthesizing => write!(f, "synthesizing"),
            Self::Closed => write!(f, "closed"),
            Self::Expired => write!(f, "expired"),
        }
    }
}

/// Role of a contribution within the pool.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ContributionRole {
    Analysis,
    Implementation,
    Review,
    Extension,
    Synthesis,
}

impl Default for ContributionRole {
    fn default() -> Self {
        Self::Analysis
    }
}

impl std::fmt::Display for ContributionRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Analysis => write!(f, "analysis"),
            Self::Implementation => write!(f, "implementation"),
            Self::Review => write!(f, "review"),
            Self::Extension => write!(f, "extension"),
            Self::Synthesis => write!(f, "synthesis"),
        }
    }
}

/// A pool session tracking collaborative multi-model work on a single task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolSession {
    pub pool_id: String,
    pub task_id: String,
    pub prompt: String,
    pub strategy: PoolStrategy,
    pub phase: PoolPhase,
    pub expected_contributions: usize,
    pub ttl_seconds: u64,
    pub created_at: String,
    pub expires_at: String,
    pub invited_agents: Vec<String>,
    pub synthesis_agent: Option<String>,
    pub result: Option<PoolResult>,
}

/// A single contribution from an agent to a pool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolContribution {
    pub contribution_id: String,
    pub pool_id: String,
    pub agent_id: String,
    pub device: String,
    pub model: Option<String>,
    pub content: String,
    pub role: ContributionRole,
    pub predecessors: Vec<String>,
    pub sequence: usize,
    pub submitted_at: String,
    pub metadata: Option<serde_json::Value>,
}

/// The synthesized result of a pool session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolResult {
    pub content: String,
    pub synthesized_by: String,
    pub device: Option<String>,
    pub contributions_merged: usize,
    pub completed_at: String,
}
