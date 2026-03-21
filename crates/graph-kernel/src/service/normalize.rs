//! Entity normalization for the Graph Kernel.
//!
//! Canonicalizes entity names to prevent knowledge fragmentation.
//! Port of the Python `entity-normalizer.py` middleware into the native Rust service layer.
//!
//! ## Problem
//!
//! Without normalization, "Dream Weaver" ≠ "dream-weaver-engine" creates duplicate entities.
//! 169 raw subjects collapse to 132 canonical entities (37 duplicates, 0.94 → 1.00 relevance).
//!
//! ## Approach
//!
//! 1. Lowercase + trim
//! 2. Replace spaces/underscores with hyphens
//! 3. Strip common punctuation (apostrophes, colons, §)
//! 4. Collapse whitespace
//! 5. Check alias map for known variants
//! 6. Return canonical form or normalized input

use std::collections::HashMap;
use std::sync::LazyLock;

/// Reverse lookup map: normalized_key → canonical_name.
/// Built from the ALIAS_TABLE at startup (zero runtime cost after first access).
static REVERSE_MAP: LazyLock<HashMap<String, &'static str>> = LazyLock::new(|| {
    let mut map = HashMap::with_capacity(512);
    for &(canonical, variants) in ALIAS_TABLE.iter() {
        // Map canonical → itself
        map.insert(normalize_key(canonical), canonical);
        for &variant in variants {
            map.insert(normalize_key(variant), canonical);
        }
    }
    map
});

/// Normalize a name to a lookup key: lowercase, strip punctuation, collapse whitespace.
fn normalize_key(name: &str) -> String {
    let s = name.to_lowercase();
    let s = s.trim();
    // Replace hyphens, underscores, apostrophes, colons, § with space
    let mut result = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '-' | '_' | '\'' | ':' | '§' => result.push(' '),
            _ => result.push(ch),
        }
    }
    // Collapse whitespace
    let parts: Vec<&str> = result.split_whitespace().collect();
    parts.join(" ")
}

/// Canonicalize an entity name.
///
/// Returns the canonical form if known, or the normalized input if not in the alias table.
///
/// # Examples
///
/// ```
/// use cc_graph_kernel::service::normalize::canonicalize_entity;
///
/// assert_eq!(canonicalize_entity("Dream Weaver"), "dream-weaver-engine");
/// assert_eq!(canonicalize_entity("BWB"), "brews-with-beats");
/// assert_eq!(canonicalize_entity("unknown thing"), "unknown-thing");
/// ```
pub fn canonicalize_entity(name: &str) -> String {
    if name.is_empty() {
        return name.to_string();
    }
    let key = normalize_key(name);
    if let Some(&canonical) = REVERSE_MAP.get(&key) {
        canonical.to_string()
    } else {
        // Return a basic normalization: lowercase, trim, spaces→hyphens
        name.trim()
            .to_lowercase()
            .replace(' ', "-")
            .replace('_', "-")
    }
}

/// Check if an entity name has a known canonical form.
pub fn is_known_entity(name: &str) -> bool {
    let key = normalize_key(name);
    REVERSE_MAP.contains_key(&key)
}

/// Get all known aliases for a canonical name.
pub fn get_aliases(canonical: &str) -> &'static [&'static str] {
    for &(canon, variants) in ALIAS_TABLE.iter() {
        if canon == canonical {
            return variants;
        }
    }
    &[]
}

// ============================================================================
// Alias Table — ported from scripts/entity-normalizer.py
// ============================================================================
// Format: (canonical_name, &[known_variants])

static ALIAS_TABLE: &[(&str, &[&str])] = &[
    // ── Core Infrastructure ──
    ("clawdbot", &[
        "Clawdbot", "ClawdBot", "clawdbot-gateway", "Clawdbot Gateway",
        "clawdbot gateway", "CLAWDBOT", "clawdbot daemon",
    ]),
    ("graph-kernel", &[
        "Graph Kernel", "Graph Kernel (Cloud)", "graph kernel", "GraphKernel",
        "graph_kernel", "graph_kernel_service", "GK",
    ]),
    ("rag-plusplus", &[
        "RAG++", "RAG++ (Cloud)", "rag-plusplus", "rag-plusplus-core",
        "RAG plus plus", "rag plusplus", "ragplusplus",
    ]),
    ("orbit", &[
        "Orbit", "Orbit (Cloud)", "trajectory-orbit", "orbit-core",
        "orbit-server",
    ]),
    ("supabase", &[
        "Supabase", "supabase", "Supabase PostgreSQL", "Supabase DB",
    ]),

    // ── People ──
    ("mohamed-diomande", &[
        "Mohamed Diomande", "Mohameddiomande", "mohameddiomande",
        "Mohamed", "mohamed", "mohamed diomande",
    ]),

    // ── Products ──
    ("dream-weaver-engine", &[
        "Dream Weaver", "dream weaver", "DreamWeaver", "Dream-weaver-engine",
        "dream-weaver", "dream weaver engine", "Dream Weaver Engine",
        "bot:dream-weaver",
    ]),
    ("comp-core", &[
        "Comp-Core", "CompCore", "comp core", "CompCoreMotion",
        "comp-core suite", "Comp-Core Suite",
    ]),
    ("koji-crm", &[
        "Koji", "koji-crm", "Koji CRM", "koji", "koji-assistant",
        "ask-koji",
    ]),
    ("milk-men", &[
        "Milk Men", "MilkMen", "milkmendelivery", "milkmen",
        "Milk Men Suite", "milkmen-expo-build",
    ]),
    ("meaning-full-power", &[
        "Meaning Full Power", "MFP", "mfp", "Meaningful Power",
        "mfp-cards", "mfp-dynamic-qr", "meaningful power",
    ]),
    ("serenity-soother", &[
        "Serenity Soother", "SerenitySoother", "serenity soother",
        "serenity", "Eternal_Serenity", "eternal-serenity",
        "Eternal Serenity",
    ]),
    ("brews-with-beats", &[
        "BWB", "BrewsWithBeats", "BufBarista", "Buf Barista",
        "brews with beats", "BWB_Customer", "bwb",
        "BufBarista-iOS-Template", "buffbarista-dance",
    ]),
    ("cali-lights", &[
        "Cali Lights", "CaliLights", "CaliLightsIOS", "cali-lights",
        "cali-lights-expo", "cali lights",
    ]),
    ("spore", &[
        "Spore", "Spore iOS app", "spore",
    ]),
    ("lifeos", &[
        "LifeOS", "life-os", "bot:life-os", "Life OS",
    ]),
    ("nko-suite", &[
        "N'Ko Suite", "nko", "N'Ko", "nko-keyboard-ai",
        "nko-code-comments", "learnnko", "sound-sigils",
        "cross-script-bridge", "lin:nko",
    ]),
    ("eternal-odyssey", &[
        "Eternal Odyssey", "LitRPG", "litrpg", "gam:odyssey",
    ]),
    ("link-it", &[
        "LinkIt", "link-it", "Link It",
    ]),

    // ── Pipelines ──
    ("kimi-k2", &[
        "KimiK2", "Kimi", "kimi", "Kimi-K2", "kimi-k2",
        "kimi_memory.db", "Kimi Knowledge Pipeline",
        "kimi-k2-extraction",
    ]),
    ("chronicle", &[
        "Chronicle", "chronicle", "chronicles-cli", "Chronicle Capture",
        "Chronicle Synthesis",
    ]),
    ("content-distributor", &[
        "Content Distributor", "content-pipeline", "content distributor",
    ]),
    ("pulse", &[
        "Pulse", "pulse", "pulse-v3", "Pulse v3", "pulse-control",
    ]),

    // ── Agent Infrastructure ──
    ("dual-max", &[
        "Dual-Max", "dual-max", "Dual Max", "Dual Max Dispatch",
        "dual dispatch", "enriched-dual-dispatch",
    ]),
    ("enriched-spawn", &[
        "enriched-spawn", "Enriched Spawn", "enriched spawn",
        "enriched dual dispatch",
    ]),

    // ── Protocols ──
    ("tpp", &[
        "Task Persistence Protocol", "TPP", "tpp",
    ]),
    ("rfiip", &[
        "Rapid-Fire Idea Intake", "RFIIP", "rfiip",
    ]),
    ("rtd", &[
        "Recursive Task Decomposition", "RTD", "rtd",
    ]),

    // ── Research / Experiments ──
    ("cognitive-twin", &[
        "cognitive-twin", "Cognitive Twin", "cognitive twin",
    ]),
    ("thought-mesh", &[
        "thought-mesh", "Thought Mesh", "thought mesh",
    ]),
    ("swarm-consensus", &[
        "swarm-consensus", "Swarm Consensus", "swarm consensus",
    ]),
    ("evoflow", &[
        "evoflow", "EvoFlow", "evo flow",
    ]),

    // ── Technologies ──
    ("rust", &["Rust", "rust"]),
    ("python", &["Python", "python"]),
    ("typescript", &["TypeScript", "typescript", "TS"]),
    ("swift", &["Swift", "Swift 6", "swift"]),
    ("react-native", &["React Native", "react-native", "Expo"]),
    ("docker", &["Docker", "Docker Compose", "docker", "docker-compose"]),
    ("gcp", &[
        "GCP", "Google Cloud Platform", "Google Cloud", "Cloud Run",
        "gcp", "gcp-deploy-prep",
    ]),
    ("github-actions", &[
        "GitHub Actions", "github-actions", "GitHub CI", "CI/CD",
    ]),

    // ── Misc subjects from existing data ──
    ("pruning-pipeline", &[
        "Pruning Pipeline", "pruning pipeline", "pruning-pipeline",
    ]),
    ("corpus-surgery", &[
        "Corpus Surgery classifier", "corpus surgery", "Corpus Surgery",
        "corpus-surgery",
    ]),
    ("prompt-synthesizer", &[
        "Prompt Synthesizer", "prompt synthesizer", "prompt-synthesizer",
    ]),
    ("dw-ai", &[
        "DW AI", "dw-ai", "DW ai",
    ]),
    ("context-search-skill", &[
        "context-search-skill", "Context Search Skill",
    ]),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_canonicalize_known_entities() {
        assert_eq!(canonicalize_entity("Dream Weaver"), "dream-weaver-engine");
        assert_eq!(canonicalize_entity("dream-weaver-engine"), "dream-weaver-engine");
        assert_eq!(canonicalize_entity("DreamWeaver"), "dream-weaver-engine");
        assert_eq!(canonicalize_entity("BWB"), "brews-with-beats");
        assert_eq!(canonicalize_entity("Buf Barista"), "brews-with-beats");
        assert_eq!(canonicalize_entity("Mohamed Diomande"), "mohamed-diomande");
        assert_eq!(canonicalize_entity("RAG++"), "rag-plusplus");
        assert_eq!(canonicalize_entity("Graph Kernel"), "graph-kernel");
        assert_eq!(canonicalize_entity("Graph Kernel (Cloud)"), "graph-kernel");
        assert_eq!(canonicalize_entity("KimiK2"), "kimi-k2");
        assert_eq!(canonicalize_entity("GCP"), "gcp");
        assert_eq!(canonicalize_entity("Google Cloud Platform"), "gcp");
        assert_eq!(canonicalize_entity("Clawdbot"), "clawdbot");
        assert_eq!(canonicalize_entity("Milk Men"), "milk-men");
        assert_eq!(canonicalize_entity("MilkMen"), "milk-men");
        assert_eq!(canonicalize_entity("Corpus Surgery classifier"), "corpus-surgery");
    }

    #[test]
    fn test_canonicalize_unknown_entity() {
        // Unknown entities get basic normalization
        assert_eq!(canonicalize_entity("unknown thing"), "unknown-thing");
        assert_eq!(canonicalize_entity("Some Random Name"), "some-random-name");
    }

    #[test]
    fn test_canonicalize_empty() {
        assert_eq!(canonicalize_entity(""), "");
    }

    #[test]
    fn test_is_known_entity() {
        assert!(is_known_entity("Dream Weaver"));
        assert!(is_known_entity("BWB"));
        assert!(!is_known_entity("totally-unknown-xyz"));
    }

    #[test]
    fn test_get_aliases() {
        let aliases = get_aliases("clawdbot");
        assert!(aliases.contains(&"Clawdbot"));
        assert!(aliases.contains(&"ClawdBot"));
    }
}
