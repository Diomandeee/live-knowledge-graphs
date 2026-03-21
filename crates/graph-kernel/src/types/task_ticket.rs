//! Task execution ticket types for Admissible Agent Orchestration.
//!
//! A TaskTicket is an HMAC-signed execution token that proves the Graph Kernel
//! authorized a specific task execution on a specific device with a specific policy.

use chrono::{DateTime, Duration, Utc};
use hmac::{Hmac, Mac};
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use uuid::Uuid;

/// An HMAC-signed task execution ticket.
///
/// Minted by `POST /api/task_ticket` and verified by `POST /api/task_ticket/verify`.
/// The signature covers all fields except itself, preventing tampering.
///
/// # Payload signed by HMAC-SHA256:
/// `task_id|device|policy_id|context_hash|graph_snapshot_hash|issued_at|ttl_seconds`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskTicket {
    /// UUID of the task being authorized.
    pub task_id: String,
    /// Device claiming execution (e.g., "mac1", "vm", "mac4").
    pub device: String,
    /// Policy governing context scope (e.g., "code", "research", "general", "team:lead").
    pub policy_id: String,
    /// xxHash64 of relevant context (placeholder "0000" until context pipeline is wired).
    pub context_hash: String,
    /// Hash of the graph state at issuance time.
    pub graph_snapshot_hash: String,
    /// ISO 8601 timestamp when the ticket was issued.
    pub issued_at: String,
    /// Time-to-live in seconds.
    pub ttl_seconds: u64,
    /// HMAC-SHA256 signature as hex string.
    pub signature: String,
}

impl TaskTicket {
    /// Mint a new task ticket with HMAC-SHA256 signature.
    ///
    /// # Arguments
    /// * `hmac_key` - Secret key for HMAC signing
    /// * `task_id` - UUID of the task
    /// * `device` - Device claiming execution
    /// * `policy_id` - Policy governing context scope
    /// * `context_hash` - Hash of relevant context
    /// * `graph_snapshot_hash` - Current graph state hash
    /// * `ttl_seconds` - Time-to-live in seconds
    pub fn mint(
        hmac_key: &[u8],
        task_id: Uuid,
        device: impl Into<String>,
        policy_id: impl Into<String>,
        context_hash: impl Into<String>,
        graph_snapshot_hash: impl Into<String>,
        ttl_seconds: u64,
    ) -> Self {
        let now = Utc::now();
        let issued_at = now.to_rfc3339();
        let device = device.into();
        let policy_id = policy_id.into();
        let context_hash = context_hash.into();
        let graph_snapshot_hash = graph_snapshot_hash.into();
        let task_id_str = task_id.to_string();

        let payload = Self::build_payload(
            &task_id_str,
            &device,
            &policy_id,
            &context_hash,
            &graph_snapshot_hash,
            &issued_at,
            ttl_seconds,
        );

        let signature = Self::compute_hmac(hmac_key, &payload);

        Self {
            task_id: task_id_str,
            device,
            policy_id,
            context_hash,
            graph_snapshot_hash,
            issued_at,
            ttl_seconds,
            signature,
        }
    }

    /// Verify the ticket's HMAC signature and check TTL hasn't expired.
    ///
    /// Returns `Ok(())` if valid, `Err(reason)` if invalid.
    pub fn verify(&self, hmac_key: &[u8]) -> Result<(), String> {
        // 1. Verify HMAC signature
        let payload = Self::build_payload(
            &self.task_id,
            &self.device,
            &self.policy_id,
            &self.context_hash,
            &self.graph_snapshot_hash,
            &self.issued_at,
            self.ttl_seconds,
        );

        let expected_sig = Self::compute_hmac(hmac_key, &payload);
        if self.signature != expected_sig {
            return Err("HMAC signature does not match".to_string());
        }

        // 2. Verify task_id is a valid UUID
        if Uuid::parse_str(&self.task_id).is_err() {
            return Err("task_id is not a valid UUID".to_string());
        }

        // 3. Verify TTL hasn't expired
        let issued_at = DateTime::parse_from_rfc3339(&self.issued_at)
            .map_err(|e| format!("Invalid issued_at timestamp: {}", e))?;

        let expires_at = issued_at + Duration::seconds(self.ttl_seconds as i64);
        let now = Utc::now();

        if now > expires_at {
            return Err(format!(
                "Ticket expired at {} (now: {})",
                expires_at, now
            ));
        }

        Ok(())
    }

    /// Check if the ticket has expired (without verifying HMAC).
    pub fn is_expired(&self) -> bool {
        let Ok(issued_at) = DateTime::parse_from_rfc3339(&self.issued_at) else {
            return true;
        };
        let expires_at = issued_at + Duration::seconds(self.ttl_seconds as i64);
        Utc::now() > expires_at
    }

    /// Get the expiration time.
    pub fn expires_at(&self) -> Option<DateTime<chrono::FixedOffset>> {
        let issued_at = DateTime::parse_from_rfc3339(&self.issued_at).ok()?;
        Some(issued_at + Duration::seconds(self.ttl_seconds as i64))
    }

    /// Encode the ticket as base64(JSON).
    pub fn encode(&self) -> String {
        use base64::Engine;
        let json = serde_json::to_string(self).expect("TaskTicket is always serializable");
        base64::engine::general_purpose::STANDARD.encode(json.as_bytes())
    }

    /// Decode a ticket from base64(JSON).
    pub fn decode(encoded: &str) -> Result<Self, String> {
        use base64::Engine;
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(encoded)
            .map_err(|e| format!("Base64 decode failed: {}", e))?;
        let json = String::from_utf8(bytes)
            .map_err(|e| format!("UTF-8 decode failed: {}", e))?;
        serde_json::from_str(&json)
            .map_err(|e| format!("JSON decode failed: {}", e))
    }

    /// Build the HMAC payload string.
    fn build_payload(
        task_id: &str,
        device: &str,
        policy_id: &str,
        context_hash: &str,
        graph_snapshot_hash: &str,
        issued_at: &str,
        ttl_seconds: u64,
    ) -> String {
        format!(
            "{}|{}|{}|{}|{}|{}|{}",
            task_id, device, policy_id, context_hash,
            graph_snapshot_hash, issued_at, ttl_seconds
        )
    }

    /// Compute HMAC-SHA256 and return hex-encoded signature.
    fn compute_hmac(key: &[u8], payload: &str) -> String {
        let mut mac = Hmac::<Sha256>::new_from_slice(key)
            .expect("HMAC-SHA256 accepts any key length");
        mac.update(payload.as_bytes());
        hex::encode(mac.finalize().into_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_KEY: &[u8] = b"test-hmac-key-for-graph-kernel";

    #[test]
    fn test_mint_and_verify_roundtrip() {
        let task_id = Uuid::new_v4();
        let ticket = TaskTicket::mint(
            TEST_KEY,
            task_id,
            "mac1",
            "code",
            "0000",
            "abc123",
            300,
        );

        assert_eq!(ticket.task_id, task_id.to_string());
        assert_eq!(ticket.device, "mac1");
        assert_eq!(ticket.policy_id, "code");
        assert_eq!(ticket.context_hash, "0000");
        assert_eq!(ticket.graph_snapshot_hash, "abc123");
        assert_eq!(ticket.ttl_seconds, 300);
        assert!(!ticket.signature.is_empty());

        // Should verify successfully
        assert!(ticket.verify(TEST_KEY).is_ok());
    }

    #[test]
    fn test_verify_rejects_wrong_key() {
        let ticket = TaskTicket::mint(
            TEST_KEY,
            Uuid::new_v4(),
            "mac1",
            "code",
            "0000",
            "abc123",
            300,
        );

        let wrong_key = b"wrong-key";
        let result = ticket.verify(wrong_key);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("HMAC signature"));
    }

    #[test]
    fn test_verify_rejects_tampered_ticket() {
        let mut ticket = TaskTicket::mint(
            TEST_KEY,
            Uuid::new_v4(),
            "mac1",
            "code",
            "0000",
            "abc123",
            300,
        );

        // Tamper with the device field
        ticket.device = "vm".to_string();

        let result = ticket.verify(TEST_KEY);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("HMAC signature"));
    }

    #[test]
    fn test_verify_rejects_expired_ticket() {
        let task_id = Uuid::new_v4();
        let now = Utc::now();
        // Create a ticket that's already expired (issued 600 seconds ago with 300s TTL)
        let issued_at = (now - Duration::seconds(600)).to_rfc3339();

        let payload = TaskTicket::build_payload(
            &task_id.to_string(),
            "mac1",
            "code",
            "0000",
            "abc123",
            &issued_at,
            300,
        );
        let signature = TaskTicket::compute_hmac(TEST_KEY, &payload);

        let ticket = TaskTicket {
            task_id: task_id.to_string(),
            device: "mac1".to_string(),
            policy_id: "code".to_string(),
            context_hash: "0000".to_string(),
            graph_snapshot_hash: "abc123".to_string(),
            issued_at,
            ttl_seconds: 300,
            signature,
        };

        let result = ticket.verify(TEST_KEY);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("expired"));
    }

    #[test]
    fn test_is_expired() {
        // Fresh ticket — not expired
        let ticket = TaskTicket::mint(
            TEST_KEY,
            Uuid::new_v4(),
            "mac1",
            "code",
            "0000",
            "abc123",
            300,
        );
        assert!(!ticket.is_expired());
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let ticket = TaskTicket::mint(
            TEST_KEY,
            Uuid::new_v4(),
            "mac1",
            "code",
            "0000",
            "abc123",
            300,
        );

        let encoded = ticket.encode();
        let decoded = TaskTicket::decode(&encoded).expect("decode should succeed");

        assert_eq!(decoded.task_id, ticket.task_id);
        assert_eq!(decoded.device, ticket.device);
        assert_eq!(decoded.policy_id, ticket.policy_id);
        assert_eq!(decoded.context_hash, ticket.context_hash);
        assert_eq!(decoded.graph_snapshot_hash, ticket.graph_snapshot_hash);
        assert_eq!(decoded.issued_at, ticket.issued_at);
        assert_eq!(decoded.ttl_seconds, ticket.ttl_seconds);
        assert_eq!(decoded.signature, ticket.signature);

        // Decoded ticket should also verify
        assert!(decoded.verify(TEST_KEY).is_ok());
    }

    #[test]
    fn test_all_policy_ids() {
        for policy in &["code", "research", "general", "team:lead", "team:subtask"] {
            let ticket = TaskTicket::mint(
                TEST_KEY,
                Uuid::new_v4(),
                "mac1",
                *policy,
                "0000",
                "abc123",
                300,
            );
            assert!(ticket.verify(TEST_KEY).is_ok());
        }
    }

    #[test]
    fn test_all_devices() {
        for device in &["mac1", "vm", "mac4"] {
            let ticket = TaskTicket::mint(
                TEST_KEY,
                Uuid::new_v4(),
                *device,
                "code",
                "0000",
                "abc123",
                300,
            );
            assert!(ticket.verify(TEST_KEY).is_ok());
        }
    }
}
