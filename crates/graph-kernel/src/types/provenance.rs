//! Replay provenance for full experiment reproduction.
//!
//! ## Purpose
//!
//! This module captures all parameters needed to exactly reproduce a slice
//! retrieval result. If provenance is complete, replaying with the same
//! parameters must yield the same slice.
//!
//! ## Provenance Components
//!
//! | Component | What It Captures | Why It Matters |
//! |-----------|-----------------|----------------|
//! | **Embedding Model** | Model ID, version, quantization | Different models → different vectors |
//! | **Normalization** | Text processing version | Different normalization → different hashes |
//! | **Retrieval Params** | k, threshold, reranking | Different params → different results |
//! | **Graph Snapshot** | Snapshot hash at retrieval time | Different graph state → different slices |
//!
//! ## Replay Contract
//!
//! Given identical provenance + identical query, retrieval MUST return
//! an identical slice (same fingerprint). Violations indicate:
//! - Non-deterministic embedding model
//! - Graph state changed between retrieval and replay
//! - Bug in the retrieval pipeline

use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

use super::slice::GraphSnapshotHash;

/// Reference to an embedding model with full version info.
///
/// This captures everything needed to reproduce embeddings:
/// - Model identifier (e.g., "text-embedding-3-small")
/// - Version or checkpoint ID
/// - Quantization settings
/// - Dimensionality
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EmbeddingModelRef {
    /// Model identifier (e.g., "openai/text-embedding-3-small").
    pub model_id: String,
    /// Model version or checkpoint.
    pub version: String,
    /// Output dimensionality.
    pub dimensions: u32,
    /// Quantization type if applicable.
    pub quantization: Option<String>,
    /// Whether the model is deterministic.
    pub deterministic: bool,
}

impl EmbeddingModelRef {
    /// Create a new embedding model reference.
    pub fn new(
        model_id: impl Into<String>,
        version: impl Into<String>,
        dimensions: u32,
    ) -> Self {
        Self {
            model_id: model_id.into(),
            version: version.into(),
            dimensions,
            quantization: None,
            deterministic: true,
        }
    }

    /// Set the quantization type.
    pub fn with_quantization(mut self, quantization: impl Into<String>) -> Self {
        self.quantization = Some(quantization.into());
        self
    }

    /// Mark the model as non-deterministic.
    pub fn non_deterministic(mut self) -> Self {
        self.deterministic = false;
        self
    }

    /// Generate a unique reference string for this model.
    pub fn to_ref_string(&self) -> String {
        let quant = self.quantization.as_deref().unwrap_or("none");
        format!(
            "{}@{}:d{}:q{}",
            self.model_id, self.version, self.dimensions, quant
        )
    }
}

/// Text normalization version.
///
/// Tracks which normalization pipeline was used to process text
/// before embedding. Changes to normalization change hashes.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NormalizationVersion {
    /// Version identifier (e.g., "v1.0.0").
    pub version: String,
    /// Hash of the normalization code/config.
    pub config_hash: String,
    /// Features enabled (e.g., ["lowercase", "strip_whitespace"]).
    pub features: Vec<String>,
}

impl NormalizationVersion {
    /// Create a new normalization version.
    pub fn new(version: impl Into<String>, config_hash: impl Into<String>) -> Self {
        Self {
            version: version.into(),
            config_hash: config_hash.into(),
            features: Vec::new(),
        }
    }

    /// Add normalization features.
    pub fn with_features(mut self, features: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.features = features.into_iter().map(|f| f.into()).collect();
        self
    }

    /// Get the current Graph Kernel normalization version.
    pub fn current() -> Self {
        Self {
            version: "1.0.0".to_string(),
            config_hash: crate::canonical_content::CANONICAL_CONTENT_VERSION.to_string(),
            features: vec![
                "crlf_to_lf".to_string(),
                "trim_whitespace".to_string(),
                "utf8_encode".to_string(),
            ],
        }
    }
}

/// Retrieval parameters that affect slice selection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RetrievalParams {
    /// Number of candidates to retrieve (k).
    pub k: u32,
    /// Similarity threshold for inclusion.
    pub similarity_threshold: f32,
    /// Whether reranking was applied.
    pub reranking_enabled: bool,
    /// Reranker model if used.
    pub reranker_model: Option<String>,
    /// Maximum context window size.
    pub max_context_tokens: Option<u32>,
    /// Policy version used for slicing.
    pub slice_policy_version: String,
    /// Policy parameters hash.
    pub policy_params_hash: String,
}

impl RetrievalParams {
    /// Create new retrieval parameters.
    pub fn new(k: u32, similarity_threshold: f32, policy_version: impl Into<String>) -> Self {
        Self {
            k,
            similarity_threshold,
            reranking_enabled: false,
            reranker_model: None,
            max_context_tokens: None,
            slice_policy_version: policy_version.into(),
            policy_params_hash: String::new(),
        }
    }

    /// Enable reranking with a specific model.
    pub fn with_reranking(mut self, model: impl Into<String>) -> Self {
        self.reranking_enabled = true;
        self.reranker_model = Some(model.into());
        self
    }

    /// Set maximum context tokens.
    pub fn with_max_tokens(mut self, tokens: u32) -> Self {
        self.max_context_tokens = Some(tokens);
        self
    }

    /// Set the policy parameters hash.
    pub fn with_policy_params_hash(mut self, hash: impl Into<String>) -> Self {
        self.policy_params_hash = hash.into();
        self
    }
}

/// Complete provenance for replay.
///
/// This struct captures everything needed to exactly reproduce
/// a slice retrieval result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayProvenance {
    /// When the retrieval was performed.
    pub timestamp: DateTime<Utc>,
    /// The embedding model used.
    pub embedding_model: EmbeddingModelRef,
    /// Text normalization version.
    pub normalization: NormalizationVersion,
    /// Retrieval parameters.
    pub retrieval_params: RetrievalParams,
    /// Graph snapshot at retrieval time.
    pub graph_snapshot: GraphSnapshotHash,
    /// The resulting slice fingerprint.
    pub slice_fingerprint: String,
    /// Query vector hash (for query reproduction).
    pub query_vector_hash: Option<String>,
    /// Additional metadata.
    pub metadata: std::collections::HashMap<String, String>,
}

impl ReplayProvenance {
    /// Check if this provenance is complete (all required fields present).
    pub fn is_complete(&self) -> bool {
        !self.embedding_model.model_id.is_empty()
            && !self.normalization.version.is_empty()
            && !self.retrieval_params.slice_policy_version.is_empty()
            && !self.slice_fingerprint.is_empty()
    }

    /// Check if replay is expected to be deterministic.
    pub fn is_deterministic(&self) -> bool {
        self.embedding_model.deterministic
    }

    /// Generate a provenance fingerprint for comparison.
    pub fn fingerprint(&self) -> String {
        use xxhash_rust::xxh64::xxh64;

        let data = format!(
            "{}|{}|{}|{}|{}",
            self.embedding_model.to_ref_string(),
            self.normalization.version,
            self.retrieval_params.slice_policy_version,
            self.retrieval_params.policy_params_hash,
            self.graph_snapshot.as_str()
        );

        format!("{:016x}", xxh64(data.as_bytes(), 0))
    }

    /// Compare two provenances for replay compatibility.
    pub fn is_replay_compatible(&self, other: &Self) -> bool {
        self.embedding_model == other.embedding_model
            && self.normalization == other.normalization
            && self.retrieval_params == other.retrieval_params
            && self.graph_snapshot == other.graph_snapshot
    }
}

/// Error when building provenance.
#[derive(Debug, thiserror::Error)]
pub enum ProvenanceError {
    /// Missing required field.
    #[error("Missing required provenance field: {0}")]
    MissingField(String),
    /// Invalid field value.
    #[error("Invalid provenance value for {field}: {reason}")]
    InvalidValue {
        /// Field name.
        field: String,
        /// Reason for invalidity.
        reason: String,
    },
}

/// Builder for constructing provenance.
#[derive(Debug, Default)]
pub struct ProvenanceBuilder {
    embedding_model: Option<EmbeddingModelRef>,
    normalization: Option<NormalizationVersion>,
    retrieval_params: Option<RetrievalParams>,
    graph_snapshot: Option<GraphSnapshotHash>,
    slice_fingerprint: Option<String>,
    query_vector_hash: Option<String>,
    metadata: std::collections::HashMap<String, String>,
}

impl ProvenanceBuilder {
    /// Create a new provenance builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the embedding model.
    pub fn embedding_model(mut self, model: EmbeddingModelRef) -> Self {
        self.embedding_model = Some(model);
        self
    }

    /// Set the normalization version.
    pub fn normalization(mut self, norm: NormalizationVersion) -> Self {
        self.normalization = Some(norm);
        self
    }

    /// Set the retrieval parameters.
    pub fn retrieval_params(mut self, params: RetrievalParams) -> Self {
        self.retrieval_params = Some(params);
        self
    }

    /// Set the graph snapshot.
    pub fn graph_snapshot(mut self, snapshot: GraphSnapshotHash) -> Self {
        self.graph_snapshot = Some(snapshot);
        self
    }

    /// Set the slice fingerprint.
    pub fn slice_fingerprint(mut self, fingerprint: impl Into<String>) -> Self {
        self.slice_fingerprint = Some(fingerprint.into());
        self
    }

    /// Set the query vector hash.
    pub fn query_vector_hash(mut self, hash: impl Into<String>) -> Self {
        self.query_vector_hash = Some(hash.into());
        self
    }

    /// Add metadata.
    pub fn metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Build the provenance.
    pub fn build(self) -> Result<ReplayProvenance, ProvenanceError> {
        let embedding_model = self.embedding_model
            .ok_or_else(|| ProvenanceError::MissingField("embedding_model".to_string()))?;
        let normalization = self.normalization
            .ok_or_else(|| ProvenanceError::MissingField("normalization".to_string()))?;
        let retrieval_params = self.retrieval_params
            .ok_or_else(|| ProvenanceError::MissingField("retrieval_params".to_string()))?;
        let graph_snapshot = self.graph_snapshot
            .ok_or_else(|| ProvenanceError::MissingField("graph_snapshot".to_string()))?;
        let slice_fingerprint = self.slice_fingerprint
            .ok_or_else(|| ProvenanceError::MissingField("slice_fingerprint".to_string()))?;

        Ok(ReplayProvenance {
            timestamp: Utc::now(),
            embedding_model,
            normalization,
            retrieval_params,
            graph_snapshot,
            slice_fingerprint,
            query_vector_hash: self.query_vector_hash,
            metadata: self.metadata,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_model_ref() {
        let model = EmbeddingModelRef::new("openai/text-embedding-3-small", "v1", 1536)
            .with_quantization("fp16");

        assert_eq!(model.model_id, "openai/text-embedding-3-small");
        assert_eq!(model.dimensions, 1536);
        assert_eq!(model.quantization, Some("fp16".to_string()));
        assert!(model.deterministic);

        let ref_str = model.to_ref_string();
        assert!(ref_str.contains("text-embedding-3-small"));
        assert!(ref_str.contains("1536"));
    }

    #[test]
    fn test_normalization_version_current() {
        let norm = NormalizationVersion::current();

        assert_eq!(norm.version, "1.0.0");
        assert!(norm.features.contains(&"crlf_to_lf".to_string()));
        assert!(norm.features.contains(&"trim_whitespace".to_string()));
    }

    #[test]
    fn test_retrieval_params() {
        let params = RetrievalParams::new(10, 0.7, "slice_policy_v1")
            .with_reranking("cohere-rerank-v3")
            .with_max_tokens(4096)
            .with_policy_params_hash("abc123");

        assert_eq!(params.k, 10);
        assert!((params.similarity_threshold - 0.7).abs() < 0.001);
        assert!(params.reranking_enabled);
        assert_eq!(params.reranker_model, Some("cohere-rerank-v3".to_string()));
        assert_eq!(params.max_context_tokens, Some(4096));
    }

    #[test]
    fn test_provenance_builder_success() {
        let provenance = ProvenanceBuilder::new()
            .embedding_model(EmbeddingModelRef::new("model", "v1", 1536))
            .normalization(NormalizationVersion::current())
            .retrieval_params(RetrievalParams::new(10, 0.7, "v1"))
            .graph_snapshot(GraphSnapshotHash::new("snapshot_hash".to_string()))
            .slice_fingerprint("slice_fp")
            .metadata("env", "test")
            .build();

        assert!(provenance.is_ok());
        let prov = provenance.unwrap();
        assert!(prov.is_complete());
        assert!(prov.is_deterministic());
        assert_eq!(prov.metadata.get("env"), Some(&"test".to_string()));
    }

    #[test]
    fn test_provenance_builder_missing_field() {
        let result = ProvenanceBuilder::new()
            .embedding_model(EmbeddingModelRef::new("model", "v1", 1536))
            // Missing other required fields
            .build();

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("Missing"));
    }

    #[test]
    fn test_provenance_fingerprint_determinism() {
        let make_prov = || {
            ProvenanceBuilder::new()
                .embedding_model(EmbeddingModelRef::new("model", "v1", 1536))
                .normalization(NormalizationVersion::current())
                .retrieval_params(RetrievalParams::new(10, 0.7, "v1").with_policy_params_hash("hash"))
                .graph_snapshot(GraphSnapshotHash::new("snapshot".to_string()))
                .slice_fingerprint("fp")
                .build()
                .unwrap()
        };

        let p1 = make_prov();
        let p2 = make_prov();

        assert_eq!(p1.fingerprint(), p2.fingerprint());
    }

    #[test]
    fn test_replay_compatibility() {
        let base = ProvenanceBuilder::new()
            .embedding_model(EmbeddingModelRef::new("model", "v1", 1536))
            .normalization(NormalizationVersion::current())
            .retrieval_params(RetrievalParams::new(10, 0.7, "v1"))
            .graph_snapshot(GraphSnapshotHash::new("snapshot".to_string()))
            .slice_fingerprint("fp")
            .build()
            .unwrap();

        let same = ProvenanceBuilder::new()
            .embedding_model(EmbeddingModelRef::new("model", "v1", 1536))
            .normalization(NormalizationVersion::current())
            .retrieval_params(RetrievalParams::new(10, 0.7, "v1"))
            .graph_snapshot(GraphSnapshotHash::new("snapshot".to_string()))
            .slice_fingerprint("different_fp") // This is the result, not input
            .build()
            .unwrap();

        let different = ProvenanceBuilder::new()
            .embedding_model(EmbeddingModelRef::new("different_model", "v1", 1536))
            .normalization(NormalizationVersion::current())
            .retrieval_params(RetrievalParams::new(10, 0.7, "v1"))
            .graph_snapshot(GraphSnapshotHash::new("snapshot".to_string()))
            .slice_fingerprint("fp")
            .build()
            .unwrap();

        assert!(base.is_replay_compatible(&same));
        assert!(!base.is_replay_compatible(&different));
    }

    #[test]
    fn test_non_deterministic_model() {
        let model = EmbeddingModelRef::new("model", "v1", 1536).non_deterministic();

        let prov = ProvenanceBuilder::new()
            .embedding_model(model)
            .normalization(NormalizationVersion::current())
            .retrieval_params(RetrievalParams::new(10, 0.7, "v1"))
            .graph_snapshot(GraphSnapshotHash::new("snapshot".to_string()))
            .slice_fingerprint("fp")
            .build()
            .unwrap();

        assert!(!prov.is_deterministic());
    }
}
