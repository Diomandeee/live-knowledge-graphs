//! Incident response infrastructure for security violations.
//!
//! ## Purpose
//!
//! This module provides types for detecting, recording, and responding to
//! security violations in the Graph Kernel. It implements canaries for
//! all security invariants (INV-GK-001 through INV-GK-008).
//!
//! ## Incident Types
//!
//! | Invariant | Incident | Severity | Response |
//! |-----------|----------|----------|----------|
//! | INV-GK-001 | SliceBoundaryViolation | CRITICAL | Page, quarantine tokens |
//! | INV-GK-003 | UnverifiedEvidenceUsage | CRITICAL | Page, audit pipeline |
//! | INV-GK-004 | ContentHashMismatch | MEDIUM | Re-run backfill |
//! | INV-GK-005 | TokenVerificationFailure | CRITICAL | Rotate secret |
//! | INV-GK-008 | SQLBoundaryBypass | CRITICAL | Audit queries |
//!
//! ## Metrics Integration
//!
//! All incident types can be converted to Prometheus counter increments.
//! The `IncidentMetrics` type provides the interface for observability systems.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::turn::TurnId;

/// Severity levels for incidents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Severity {
    /// Low severity - investigate within 1 day.
    Low,
    /// Medium severity - investigate within 4 hours.
    Medium,
    /// High severity - investigate within 1 hour.
    High,
    /// Critical severity - immediate page, investigate now.
    Critical,
}

impl Severity {
    /// Get the response time SLA in seconds.
    pub fn response_time_secs(&self) -> u64 {
        match self {
            Self::Low => 86400,     // 1 day
            Self::Medium => 14400,  // 4 hours
            Self::High => 3600,     // 1 hour
            Self::Critical => 0,    // Immediate
        }
    }

    /// Check if this severity requires paging.
    pub fn requires_page(&self) -> bool {
        matches!(self, Self::Critical)
    }
}

impl std::fmt::Display for Severity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Low => write!(f, "LOW"),
            Self::Medium => write!(f, "MEDIUM"),
            Self::High => write!(f, "HIGH"),
            Self::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// Type of security incident.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum IncidentType {
    /// Attempt to access turns outside slice boundary (INV-GK-001).
    SliceBoundaryViolation {
        /// Slice fingerprint that was violated.
        slice_fingerprint: String,
        /// Number of unauthorized turn IDs.
        unauthorized_count: usize,
    },
    /// Unverified evidence used in promotion (INV-GK-003).
    UnverifiedEvidenceUsage {
        /// Where the unverified evidence was detected.
        pipeline_stage: String,
    },
    /// Content hash mismatch detected (INV-GK-004).
    ContentHashMismatch {
        /// Turn ID with mismatched hash.
        turn_id: TurnId,
        /// Expected hash.
        expected_hash: String,
        /// Computed hash.
        computed_hash: String,
    },
    /// Token verification failed (INV-GK-005).
    TokenVerificationFailure {
        /// Slice fingerprint that failed verification.
        slice_fingerprint: String,
        /// Reason for failure.
        reason: String,
    },
    /// SQL query attempted to bypass slice boundary (INV-GK-008).
    SqlBoundaryBypass {
        /// Query fingerprint or hash.
        query_fingerprint: String,
        /// Source of the query.
        source: String,
    },
    /// Policy was mutated (INV-GK-007).
    PolicyMutation {
        /// Policy ID that was mutated.
        policy_id: String,
        /// Original params hash.
        original_hash: String,
        /// New params hash.
        new_hash: String,
    },
    /// Generic security incident.
    Other {
        /// Description of the incident.
        description: String,
    },
}

impl IncidentType {
    /// Get the severity of this incident type.
    pub fn severity(&self) -> Severity {
        match self {
            Self::SliceBoundaryViolation { .. } => Severity::Critical,
            Self::UnverifiedEvidenceUsage { .. } => Severity::Critical,
            Self::ContentHashMismatch { .. } => Severity::Medium,
            Self::TokenVerificationFailure { .. } => Severity::Critical,
            Self::SqlBoundaryBypass { .. } => Severity::Critical,
            Self::PolicyMutation { .. } => Severity::High,
            Self::Other { .. } => Severity::Medium,
        }
    }

    /// Get the invariant this incident relates to.
    pub fn invariant(&self) -> &'static str {
        match self {
            Self::SliceBoundaryViolation { .. } => "INV-GK-001",
            Self::UnverifiedEvidenceUsage { .. } => "INV-GK-003",
            Self::ContentHashMismatch { .. } => "INV-GK-004",
            Self::TokenVerificationFailure { .. } => "INV-GK-005",
            Self::SqlBoundaryBypass { .. } => "INV-GK-008",
            Self::PolicyMutation { .. } => "INV-GK-007",
            Self::Other { .. } => "UNKNOWN",
        }
    }

    /// Get the Prometheus metric name for this incident type.
    pub fn metric_name(&self) -> &'static str {
        match self {
            Self::SliceBoundaryViolation { .. } => "graph_kernel_slice_boundary_violations_total",
            Self::UnverifiedEvidenceUsage { .. } => "graph_kernel_unverified_evidence_usage_total",
            Self::ContentHashMismatch { .. } => "graph_kernel_content_hash_mismatches_total",
            Self::TokenVerificationFailure { .. } => "graph_kernel_token_verification_failures_total",
            Self::SqlBoundaryBypass { .. } => "graph_kernel_sql_boundary_bypass_total",
            Self::PolicyMutation { .. } => "graph_kernel_policy_mutations_total",
            Self::Other { .. } => "graph_kernel_other_incidents_total",
        }
    }
}

/// A recorded security incident.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Incident {
    /// Unique incident ID.
    pub id: String,
    /// When the incident occurred.
    pub timestamp: DateTime<Utc>,
    /// Type of incident.
    pub incident_type: IncidentType,
    /// Severity level.
    pub severity: Severity,
    /// Source of the incident (service, component).
    pub source: String,
    /// Additional context.
    pub context: HashMap<String, String>,
    /// Whether the incident has been acknowledged.
    pub acknowledged: bool,
    /// When the incident was acknowledged.
    pub acknowledged_at: Option<DateTime<Utc>>,
    /// Who acknowledged the incident.
    pub acknowledged_by: Option<String>,
}

impl Incident {
    /// Create a new incident.
    pub fn new(incident_type: IncidentType, source: impl Into<String>) -> Self {
        let severity = incident_type.severity();
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            incident_type,
            severity,
            source: source.into(),
            context: HashMap::new(),
            acknowledged: false,
            acknowledged_at: None,
            acknowledged_by: None,
        }
    }

    /// Add context to the incident.
    pub fn with_context(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.context.insert(key.into(), value.into());
        self
    }

    /// Acknowledge the incident.
    pub fn acknowledge(&mut self, by: impl Into<String>) {
        self.acknowledged = true;
        self.acknowledged_at = Some(Utc::now());
        self.acknowledged_by = Some(by.into());
    }

    /// Log this incident as a structured event.
    pub fn log(&self) {
        tracing::error!(
            incident_id = %self.id,
            incident_type = ?self.incident_type,
            severity = %self.severity,
            invariant = %self.incident_type.invariant(),
            source = %self.source,
            context = ?self.context,
            "SECURITY_INCIDENT: {} violation detected",
            self.incident_type.invariant()
        );
    }
}

/// A quarantined token that failed verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuarantinedToken {
    /// Unique quarantine entry ID.
    pub id: String,
    /// The token value that was quarantined.
    pub token_hash: String,
    /// Slice fingerprint associated with the token.
    pub slice_fingerprint: String,
    /// When the token was quarantined.
    pub quarantined_at: DateTime<Utc>,
    /// Reason for quarantine.
    pub reason: String,
    /// Related incident ID.
    pub incident_id: Option<String>,
    /// Whether the token has been reviewed.
    pub reviewed: bool,
    /// Review decision (allow, block, delete).
    pub review_decision: Option<String>,
    /// When the token was reviewed.
    pub reviewed_at: Option<DateTime<Utc>>,
}

impl QuarantinedToken {
    /// Create a new quarantine entry.
    pub fn new(
        token_hash: impl Into<String>,
        slice_fingerprint: impl Into<String>,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            token_hash: token_hash.into(),
            slice_fingerprint: slice_fingerprint.into(),
            quarantined_at: Utc::now(),
            reason: reason.into(),
            incident_id: None,
            reviewed: false,
            review_decision: None,
            reviewed_at: None,
        }
    }

    /// Link to an incident.
    pub fn with_incident(mut self, incident_id: impl Into<String>) -> Self {
        self.incident_id = Some(incident_id.into());
        self
    }

    /// Mark as reviewed.
    pub fn review(&mut self, decision: impl Into<String>) {
        self.reviewed = true;
        self.review_decision = Some(decision.into());
        self.reviewed_at = Some(Utc::now());
    }
}

/// Metrics counter interface.
///
/// This trait defines the interface for incrementing Prometheus counters.
/// Implementations can be provided for different metrics backends.
pub trait IncidentMetrics: Send + Sync {
    /// Increment a counter by 1.
    fn increment(&self, metric_name: &str, labels: &[(&str, &str)]);

    /// Record an incident.
    fn record_incident(&self, incident: &Incident) {
        let _labels = [
            ("severity", incident.severity.to_string().as_str()),
            ("invariant", incident.incident_type.invariant()),
            ("source", incident.source.as_str()),
        ];

        // This won't compile as-is due to lifetime issues with to_string()
        // In real implementation, use static strings or owned labels
        self.increment(incident.incident_type.metric_name(), &[
            ("severity", match incident.severity {
                Severity::Low => "low",
                Severity::Medium => "medium",
                Severity::High => "high",
                Severity::Critical => "critical",
            }),
            ("invariant", incident.incident_type.invariant()),
        ]);
    }
}

/// No-op metrics implementation for testing.
#[derive(Debug, Default)]
pub struct NoOpMetrics;

impl IncidentMetrics for NoOpMetrics {
    fn increment(&self, _metric_name: &str, _labels: &[(&str, &str)]) {
        // No-op
    }
}

/// In-memory metrics for testing.
#[derive(Debug, Default)]
pub struct TestMetrics {
    /// Counter values.
    pub counters: std::sync::Mutex<HashMap<String, u64>>,
}

impl IncidentMetrics for TestMetrics {
    fn increment(&self, metric_name: &str, labels: &[(&str, &str)]) {
        let key = format!("{}:{:?}", metric_name, labels);
        let mut counters = self.counters.lock().unwrap();
        *counters.entry(key).or_insert(0) += 1;
    }
}

impl TestMetrics {
    /// Get the count for a metric.
    pub fn get_count(&self, metric_name: &str) -> u64 {
        let counters = self.counters.lock().unwrap();
        counters
            .iter()
            .filter(|(k, _)| k.starts_with(metric_name))
            .map(|(_, v)| v)
            .sum()
    }
}

/// SQL schema for the quarantine table.
pub const QUARANTINE_TABLE_SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS graph_kernel_quarantined_tokens (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    token_hash TEXT NOT NULL,
    slice_fingerprint TEXT NOT NULL,
    quarantined_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    reason TEXT NOT NULL,
    incident_id TEXT,
    reviewed BOOLEAN NOT NULL DEFAULT FALSE,
    review_decision TEXT,
    reviewed_at TIMESTAMPTZ,

    -- Indexes for common queries
    CONSTRAINT quarantine_token_hash_idx UNIQUE (token_hash)
);

CREATE INDEX IF NOT EXISTS idx_quarantine_fingerprint
    ON graph_kernel_quarantined_tokens(slice_fingerprint);
CREATE INDEX IF NOT EXISTS idx_quarantine_unreviewed
    ON graph_kernel_quarantined_tokens(reviewed) WHERE NOT reviewed;
CREATE INDEX IF NOT EXISTS idx_quarantine_incident
    ON graph_kernel_quarantined_tokens(incident_id) WHERE incident_id IS NOT NULL;
"#;

/// SQL schema for the incident log table.
pub const INCIDENT_TABLE_SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS graph_kernel_incidents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    incident_type TEXT NOT NULL,
    incident_data JSONB NOT NULL,
    severity TEXT NOT NULL,
    invariant TEXT NOT NULL,
    source TEXT NOT NULL,
    context JSONB,
    acknowledged BOOLEAN NOT NULL DEFAULT FALSE,
    acknowledged_at TIMESTAMPTZ,
    acknowledged_by TEXT,

    -- Indexes for common queries
    CONSTRAINT incident_severity_check CHECK (severity IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL'))
);

CREATE INDEX IF NOT EXISTS idx_incident_timestamp
    ON graph_kernel_incidents(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_incident_severity
    ON graph_kernel_incidents(severity);
CREATE INDEX IF NOT EXISTS idx_incident_invariant
    ON graph_kernel_incidents(invariant);
CREATE INDEX IF NOT EXISTS idx_incident_unacknowledged
    ON graph_kernel_incidents(acknowledged) WHERE NOT acknowledged;
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_severity_response_times() {
        assert_eq!(Severity::Critical.response_time_secs(), 0);
        assert_eq!(Severity::High.response_time_secs(), 3600);
        assert_eq!(Severity::Medium.response_time_secs(), 14400);
        assert_eq!(Severity::Low.response_time_secs(), 86400);
    }

    #[test]
    fn test_incident_type_severity() {
        let boundary = IncidentType::SliceBoundaryViolation {
            slice_fingerprint: "fp".to_string(),
            unauthorized_count: 1,
        };
        assert_eq!(boundary.severity(), Severity::Critical);

        let hash = IncidentType::ContentHashMismatch {
            turn_id: TurnId::new(uuid::Uuid::new_v4()),
            expected_hash: "a".to_string(),
            computed_hash: "b".to_string(),
        };
        assert_eq!(hash.severity(), Severity::Medium);
    }

    #[test]
    fn test_incident_type_invariant() {
        let boundary = IncidentType::SliceBoundaryViolation {
            slice_fingerprint: "fp".to_string(),
            unauthorized_count: 1,
        };
        assert_eq!(boundary.invariant(), "INV-GK-001");

        let token = IncidentType::TokenVerificationFailure {
            slice_fingerprint: "fp".to_string(),
            reason: "bad token".to_string(),
        };
        assert_eq!(token.invariant(), "INV-GK-005");
    }

    #[test]
    fn test_incident_creation() {
        let incident = Incident::new(
            IncidentType::SliceBoundaryViolation {
                slice_fingerprint: "test_fp".to_string(),
                unauthorized_count: 5,
            },
            "test_service",
        )
        .with_context("request_id", "req123");

        assert!(!incident.id.is_empty());
        assert_eq!(incident.severity, Severity::Critical);
        assert_eq!(incident.context.get("request_id"), Some(&"req123".to_string()));
        assert!(!incident.acknowledged);
    }

    #[test]
    fn test_incident_acknowledge() {
        let mut incident = Incident::new(
            IncidentType::ContentHashMismatch {
                turn_id: TurnId::new(uuid::Uuid::new_v4()),
                expected_hash: "a".to_string(),
                computed_hash: "b".to_string(),
            },
            "test",
        );

        assert!(!incident.acknowledged);
        incident.acknowledge("operator@example.com");
        assert!(incident.acknowledged);
        assert!(incident.acknowledged_at.is_some());
        assert_eq!(incident.acknowledged_by, Some("operator@example.com".to_string()));
    }

    #[test]
    fn test_quarantined_token() {
        let token = QuarantinedToken::new(
            "bad_token_hash",
            "slice_fp",
            "Token verification failed",
        )
        .with_incident("incident_123");

        assert!(!token.reviewed);
        assert_eq!(token.incident_id, Some("incident_123".to_string()));
    }

    #[test]
    fn test_quarantined_token_review() {
        let mut token = QuarantinedToken::new("hash", "fp", "reason");

        assert!(!token.reviewed);
        token.review("block");
        assert!(token.reviewed);
        assert_eq!(token.review_decision, Some("block".to_string()));
        assert!(token.reviewed_at.is_some());
    }

    #[test]
    fn test_test_metrics() {
        let metrics = TestMetrics::default();

        metrics.increment("test_counter", &[("label", "value")]);
        metrics.increment("test_counter", &[("label", "value")]);
        metrics.increment("other_counter", &[]);

        assert_eq!(metrics.get_count("test_counter"), 2);
        assert_eq!(metrics.get_count("other_counter"), 1);
    }

    #[test]
    fn test_metric_names() {
        let boundary = IncidentType::SliceBoundaryViolation {
            slice_fingerprint: "fp".to_string(),
            unauthorized_count: 1,
        };
        assert_eq!(boundary.metric_name(), "graph_kernel_slice_boundary_violations_total");

        let token = IncidentType::TokenVerificationFailure {
            slice_fingerprint: "fp".to_string(),
            reason: "test".to_string(),
        };
        assert_eq!(token.metric_name(), "graph_kernel_token_verification_failures_total");
    }
}
