# Knowledge Graph Extraction Strategy for Industrial Manuals
## Analysis of Buehler's GraphReasoning & Adaptation Plan

**Date:** December 3, 2025  
**Context:** Extracting symptom→cause→fix knowledge graphs from industrial manuals for edge-deployable fault diagnosis

---

## 1. Buehler's Approach: Key Insights

### 1.1 Core Method Summary
Buehler's work transforms **1,000 scientific papers** into ontological knowledge graphs using:

**Graph Construction:**
- LLM-based extraction of entities and relationships from text chunks
- Node embeddings using transformer models (e.g., `gte-large-en-v1.5`)
- Scale-free, highly connected graph structure
- Community detection via Louvain algorithm
- Graph simplification by merging similar nodes (similarity threshold ~0.95)

**Graph Reasoning:**
- Path finding between semantically distant concepts using embedding similarity
- Multi-path analysis combining diverse reasoning chains
- Isomorphic mapping revealing structural parallels across domains
- Deep node embeddings for combinatorial similarity ranking

**Key Technical Components:**
1. Text chunking (default: 2500 tokens, configurable overlap)
2. LLM prompting for triple extraction (entity-relation-entity)
3. Contextual proximity edges (optional)
4. Iterative refinement loops
5. Graph merging/composition for multi-document graphs

### 1.2 Critical Differences: Papers vs. Manuals

| Aspect | Scientific Papers (Buehler) | Industrial Manuals (Our Case) |
|--------|----------------------------|-------------------------------|
| **Document count** | 1,000 papers | 1 manual → N manuals (same line) |
| **Content structure** | Abstract concepts, theories | Concrete: symptoms, causes, fixes |
| **Graph density** | High (cross-domain connections) | Sparse (domain-specific chains) |
| **Reasoning goal** | Novel hypothesis generation | Fault diagnosis & repair |
| **Deployment** | Research exploration | Edge device (resource-constrained) |
| **Multimodal content** | Minimal | **Critical**: diagrams, tables, schematics |
| **Graph topology** | Scale-free, community-rich | **Hierarchical DAG**: symptom→cause→fix |

---

## 2. Proposed Graph Structure for Manuals

### 2.1 Node Types & Schema

```
ManualsGraph Schema:
├── SYMPTOM (observable manifestation)
│   ├── properties: description, sensor_data_pattern, frequency, severity
│   ├── embedding: semantic vector
│   └── source: page_number, section, image_ref
├── ROOT_CAUSE (underlying failure mode)
│   ├── properties: description, component, failure_mechanism
│   ├── embedding: semantic vector
│   └── source: page_number, diagram_ref
├── FIX (repair procedure)
│   ├── properties: steps[], tools_required[], time_estimate, safety_notes
│   ├── embedding: semantic vector
│   └── source: page_number, table_ref
├── COMPONENT (physical part)
│   ├── properties: part_number, specs, location
│   └── relationships: affects→, upstream_of→, downstream_of→
└── CONDITION (environmental/operational context)
    ├── properties: description, threshold_values
    └── relationships: modulates→
```

### 2.2 Edge Types & Semantics

**Primary Causal Chains:**
- `(SYMPTOM)-[INDICATES]->(ROOT_CAUSE)` [weight: confidence score]
- `(ROOT_CAUSE)-[RESOLVED_BY]->(FIX)` [weight: effectiveness score]
- `(SYMPTOM)-[DIRECT_FIX]->(FIX)` [for symptom-only troubleshooting]

**Contextual Relations:**
- `(COMPONENT)-[EXHIBITS]->(SYMPTOM)`
- `(COMPONENT)-[FAILS_AS]->(ROOT_CAUSE)`
- `(FIX)-[TARGETS]->(COMPONENT)`
- `(CONDITION)-[TRIGGERS]->(SYMPTOM)` [conditional edges]
- `(ROOT_CAUSE)-[PROPAGATES_TO]->(ROOT_CAUSE)` [cascade failures]

**Cross-Manual Relations (for multi-manual graphs):**
- `(COMPONENT_upstream)-[CAUSES]->(SYMPTOM_downstream)` [system-level]
- `(FIX_prerequisite)-[ENABLES]->(FIX_main)` [procedural dependencies]

### 2.3 Graph Topology Design

**For Single Manual:**
```
Directed Acyclic Graph (DAG) with hierarchical layers:
Layer 1: SYMPTOMS (entry points for diagnosis)
Layer 2: ROOT_CAUSES (intermediate reasoning)
Layer 3: FIXES (terminal actions)
```

**For Multi-Manual (Same Line):**
```
Composite Graph with cross-document edges:
- Intra-manual: Dense DAGs (as above)
- Inter-manual: Sparse bridges connecting shared COMPONENTS
- Hierarchical clustering: Manual_ID as graph attribute
```

**Why DAG over Scale-Free?**
- Manuals encode **deterministic troubleshooting logic**, not exploratory networks
- Prevents cycles that would confuse diagnostic reasoning
- Enables topological sort for sequential troubleshooting steps
- Compact for edge deployment (fewer edges than Buehler's dense graphs)

---

## 3. Extraction Pipeline Architecture

### 3.1 Phase 1: Multimodal Document Parsing

**Text Extraction:**
```python
1. PDF → structured text (pypdf, pdfplumber)
2. Section detection (regex + heuristics):
   - "Troubleshooting" / "Fault Diagnosis" / "Error Codes"
   - "Maintenance Procedures"
   - "Parts List" / "Component Specifications"
3. Chunk creation:
   - Semantic chunking (preserve symptom-fix pairs)
   - Overlap: 200 tokens (to capture cross-boundary relationships)
```

**Visual Element Extraction (CRITICAL):**
```python
1. Table extraction:
   - Detect fault code tables (code→description→action)
   - Component specification tables
   - Tools: camelot, tabula-py, pdfplumber
   
2. Diagram/Image analysis:
   - Extract wiring diagrams, flowcharts, component layouts
   - OCR for labels (tesseract, EasyOCR)
   - Vision-Language Model (VLM) for interpretation:
     * GPT-4V / Claude 3 Opus / LLaVA
     * Prompt: "Extract all labeled components, connections, 
                and any indicated fault points"
   
3. Flowchart parsing:
   - Decision trees for troubleshooting
   - Convert to graph structure directly
   - Tools: CV-based edge detection + LLM validation
```

**Why This Matters:**
- ~60-70% of troubleshooting info in industrial manuals is in **tables and diagrams**
- Fault code tables are pre-structured symptom→cause→fix triples
- Wiring diagrams encode component dependencies
- Flowcharts are **ready-made subgraphs**

### 3.2 Phase 2: Entity & Relation Extraction

**Approach: Hybrid (Rule-Based + LLM)**

**Rule-Based Extraction (High Precision for Structured Content):**
```python
# For fault code tables
pattern_fault_table = {
    "columns": ["Code", "Description", "Cause", "Action"],
    "extraction": regex + table parser
    "output": [(SYMPTOM, ROOT_CAUSE, FIX), ...]
}

# For component lists
pattern_parts = {
    "format": "Part Number | Description | Location",
    "output": [COMPONENT, ...]
}
```

**LLM Extraction (for Unstructured Text):**
```python
prompt_template = """
You are extracting troubleshooting knowledge from an industrial manual.

Text chunk:
{chunk}

Extract all instances of:
1. SYMPTOM: Observable machine behavior/error
2. ROOT_CAUSE: Underlying problem causing the symptom
3. FIX: Corrective action/procedure

Format as JSON:
[
  {
    "symptom": {"text": "...", "severity": "low|medium|high"},
    "root_cause": {"text": "...", "component": "..."},
    "fix": {
      "steps": ["...", "..."], 
      "tools": ["..."],
      "safety": "..."
    }
  }
]

Important: 
- Only extract explicit troubleshooting info
- Preserve exact terminology from manual
- Include page/section references
"""
```

**Buehler's Method Adaptation:**
- Use his chunk size (2500 tokens) as baseline
- **But**: Preserve table/section boundaries (don't chunk mid-table)
- His `repeat_refine=0` → we use `repeat_refine=1` for validation pass
- His `include_contextual_proximity=False` → we use `True` for component co-occurrence

### 3.3 Phase 3: Graph Construction

**Single Manual Graph:**
```python
def build_manual_graph(manual_path):
    # 1. Multimodal extraction
    text_chunks = extract_text_semantic_chunks(manual_path)
    tables = extract_tables(manual_path)
    diagrams = extract_diagrams_with_vlm(manual_path)
    
    # 2. Entity extraction
    entities = []
    for chunk in text_chunks:
        entities += llm_extract_entities(chunk, prompt_template)
    for table in tables:
        entities += rule_extract_from_table(table)
    for diagram in diagrams:
        entities += vlm_extract_from_diagram(diagram)
    
    # 3. Graph initialization
    G = nx.DiGraph()  # Directed for causal flow
    
    # 4. Node creation
    for entity in deduplicate_entities(entities):
        G.add_node(entity.id, 
                   type=entity.type,
                   text=entity.text,
                   embedding=None,  # filled later
                   source=entity.source_ref)
    
    # 5. Edge creation
    for entity in entities:
        if entity.type == "SYMPTOM":
            for cause in entity.causes:
                G.add_edge(entity.id, cause.id, 
                          relation="INDICATES", 
                          confidence=entity.confidence)
        if entity.type == "ROOT_CAUSE":
            for fix in entity.fixes:
                G.add_edge(entity.id, fix.id,
                          relation="RESOLVED_BY",
                          confidence=entity.confidence)
    
    # 6. Component dependency edges (from diagrams)
    for component_link in diagram_component_links:
        G.add_edge(component_link.upstream, 
                   component_link.downstream,
                   relation="UPSTREAM_OF")
    
    # 7. Generate embeddings
    node_embeddings = generate_node_embeddings(G, tokenizer, model)
    
    # 8. Graph simplification (Buehler's method)
    G_simplified = simplify_graph(G, node_embeddings, 
                                  similarity_threshold=0.92,
                                  use_llm=True)  # LLM renames merged nodes
    
    return G_simplified, node_embeddings
```

**Multi-Manual Graph (Same Production Line):**
```python
def merge_manual_graphs(manual_graphs: List[nx.DiGraph]):
    # 1. Load individual graphs
    G_combined = nx.DiGraph()
    
    # 2. Union of all nodes (with manual_id attribute)
    for manual_id, G in enumerate(manual_graphs):
        for node, attrs in G.nodes(data=True):
            G_combined.add_node(f"{manual_id}_{node}",
                               manual_id=manual_id,
                               **attrs)
    
    # 3. Intra-manual edges (preserve all)
    for manual_id, G in enumerate(manual_graphs):
        for u, v, attrs in G.edges(data=True):
            G_combined.add_edge(f"{manual_id}_{u}",
                               f"{manual_id}_{v}",
                               **attrs)
    
    # 4. Inter-manual edges (via component matching)
    component_index = build_component_index(G_combined)
    for comp_name, nodes in component_index.items():
        if len(nodes) > 1:  # Component appears in multiple manuals
            # Link symptoms/fixes across manuals for same component
            for node_a in nodes:
                for node_b in nodes:
                    if node_a != node_b and can_link(node_a, node_b):
                        G_combined.add_edge(node_a, node_b,
                                           relation="CROSS_MANUAL_LINK",
                                           confidence=embedding_similarity(node_a, node_b))
    
    # 5. Prune weak inter-manual edges (threshold: 0.85)
    edges_to_remove = [
        (u, v) for u, v, d in G_combined.edges(data=True)
        if d.get('relation') == 'CROSS_MANUAL_LINK' 
        and d.get('confidence', 0) < 0.85
    ]
    G_combined.remove_edges_from(edges_to_remove)
    
    # 6. Community detection (Buehler's Louvain)
    communities = nx.community.louvain_communities(G_combined.to_undirected())
    # Annotate nodes with community_id
    
    return G_combined
```

### 3.4 Phase 4: Graph Compression for Edge Deployment

**Problem:** Buehler's graphs are large (1000s of nodes). Edge devices need <10MB models.

**Solution: Hierarchical Abstraction + Pruning**

```python
def compress_for_edge_deployment(G, max_nodes=500, max_edges=1500):
    """
    Compress graph while preserving diagnostic utility
    """
    # 1. Identify critical paths (high betweenness centrality)
    centrality = nx.betweenness_centrality(G)
    critical_nodes = [n for n, c in centrality.items() 
                     if c > np.percentile(list(centrality.values()), 80)]
    
    # 2. Abstract low-frequency symptoms
    symptom_freq = get_symptom_frequencies(G)  # from manual emphasis
    rare_symptoms = [s for s, f in symptom_freq.items() if f < threshold]
    
    # Group rare symptoms into "OTHER_<category>" nodes
    G_abstract = abstract_rare_nodes(G, rare_symptoms)
    
    # 3. Prune redundant paths
    # If multiple paths SYMPTOM→CAUSE→FIX exist, keep shortest + highest confidence
    G_pruned = prune_redundant_paths(G_abstract)
    
    # 4. Quantize embeddings (float32 → int8)
    embeddings_quantized = quantize_embeddings(node_embeddings, bits=8)
    
    # 5. Validate: Check diagnostic coverage
    # Ensure 95%+ of original symptom→fix paths still reachable
    coverage = validate_diagnostic_coverage(G, G_pruned)
    assert coverage > 0.95, "Too much information loss"
    
    return G_pruned, embeddings_quantized
```

**Resulting Graph Size:**
- Nodes: 300-500 (vs. Buehler's 1000s)
- Edges: 800-1500 (DAG is sparser than scale-free)
- Embeddings: 384-dim int8 → ~150KB per 500 nodes
- Total: **<5MB for graph + embeddings**

---

## 4. Extraction Success Factors

### 4.1 What Will Work (From Buehler)

✅ **Node embeddings for similarity search**
- Find nearest symptom match when sensor data doesn't match exactly
- Enable "fuzzy" fault diagnosis

✅ **Path finding algorithms**
- Shortest path = fastest diagnosis
- Multi-path analysis = alternative solutions

✅ **Graph simplification**
- Merge synonymous terms ("motor failure" ≈ "motor malfunction")
- Critical for consistent troubleshooting vocabulary

✅ **Community detection**
- Group related faults (e.g., all bearing failures)
- Helps identify systemic issues

### 4.2 What Needs Adaptation

⚠️ **Buehler's scale-free assumption**
- Papers have broad cross-domain links
- Manuals are **domain-specific** → use DAG not scale-free

⚠️ **Chunk size**
- Papers: uniform paragraphs
- Manuals: respect table/section boundaries

⚠️ **Isomorphic reasoning**
- Buehler maps Beethoven→materials (creative)
- We need **deterministic** reasoning (no creativity in fault diagnosis)

### 4.3 Novel Extensions (Beyond Buehler)

🚀 **Multimodal extraction**
- Buehler: text-only
- Us: tables (60% of info) + diagrams (30%) + text (10%)

🚀 **Hierarchical graph compression**
- Buehler: research-scale graphs
- Us: edge-deployable graphs (<5MB)

🚀 **Conditional edges**
- Encode "IF temperature > 80°C AND vibration detected THEN..."
- Critical for context-dependent diagnostics

🚀 **Temporal reasoning**
- Track symptom progression: "First A, then B" → cause C
- Requires temporal edge attributes

---

## 5. Implementation Roadmap

### Phase 1: Single Manual Proof-of-Concept (2 weeks)
**Deliverables:**
- [ ] Multimodal extraction pipeline (text + tables + diagrams)
- [ ] LLM prompt engineering for entity extraction
- [ ] Graph construction with DAG topology
- [ ] Node embedding generation
- [ ] Visualization & validation (manual review)

**Test Case:** Choose 1 representative manual (~200 pages)
- Must have: fault code table, wiring diagrams, troubleshooting flowchart
- Success metric: Extract 80%+ of fault codes as complete symptom→cause→fix triples

### Phase 2: Graph Reasoning & Compression (2 weeks)
**Deliverables:**
- [ ] Path finding for symptom→fix queries
- [ ] Embedding-based similarity search
- [ ] Graph simplification (merge synonyms)
- [ ] Compression to <5MB
- [ ] Export to edge-friendly format (GraphML → binary)

**Test Case:** Deploy on simulated edge device
- Query: "Motor overheating + unusual noise"
- Expected: Return top-3 diagnostic paths with fixes
- Latency: <100ms

### Phase 3: Multi-Manual Integration (3 weeks)
**Deliverables:**
- [ ] Cross-manual component matching
- [ ] Inter-manual edge creation
- [ ] Community detection (fault clusters)
- [ ] Transfer learning evaluation (manual A helps diagnose manual B)

**Test Case:** 5 manuals from same production line
- Success metric: 20%+ improvement in diagnostic accuracy vs. single-manual baseline

### Phase 4: Evaluation at Scale (4 weeks)
**Deliverables:**
- [ ] Process 1000+ manuals from Internet Archive
- [ ] Benchmark graph quality metrics:
  * Node coverage (% of symptoms extracted)
  * Edge precision (human validation sample)
  * Diagnostic accuracy (simulated fault injection)
- [ ] Ablation studies (with/without diagrams, tables, etc.)
- [ ] Paper write-up

---

## 6. Key Research Questions

### RQ1: Multimodal Fusion
**How to weight text vs. table vs. diagram extractions when conflicts arise?**
- Hypothesis: Tables > Diagrams > Text (for fault codes)
- Experiment: Compare extraction sources against ground truth

### RQ2: Graph Topology
**DAG vs. scale-free: Which is better for fault diagnosis?**
- Metrics: Diagnostic accuracy, query latency, explainability
- Ablation: Test both topologies on same manual set

### RQ3: Cross-Manual Transfer
**Does a multi-manual graph improve single-fault diagnosis?**
- Baseline: Per-manual graphs
- Experiment: Add cross-manual edges, measure accuracy delta

### RQ4: Embedding Models
**Domain-specific vs. general embeddings?**
- Test: `gte-large-en-v1.5` (Buehler's) vs. `all-MiniLM-L6-v2` vs. fine-tuned industrial BERT
- Metric: Symptom similarity ranking precision

### RQ5: Compression Trade-offs
**How much can we compress before diagnostic accuracy degrades?**
- Sweep: 100, 300, 500, 1000 nodes
- Target: Pareto frontier (size vs. accuracy)

---

## 7. Technical Stack

### Core Libraries
```python
# Graph processing
networkx>=3.0
pyvis  # Interactive visualization (Buehler's approach)

# NLP & embeddings
transformers>=4.30
sentence-transformers
torch>=2.0

# Multimodal extraction
pdfplumber  # Tables
camelot-py[cv]  # Advanced table extraction
pytesseract  # OCR
openai  # GPT-4V for diagram analysis
anthropic  # Claude for complex extractions

# Graph reasoning (Buehler's library)
git+https://github.com/lamm-mit/GraphReasoning

# Utilities
numpy, pandas, matplotlib
```

### LLM Strategy
**Extraction:** GPT-4 Turbo (best accuracy for structured output)
**Reasoning:** Claude 3 Opus (longer context for multi-path analysis)
**Diagram analysis:** GPT-4V or LLaVA (open-source alternative)

---

## 8. Success Criteria

### Technical Metrics
- **Coverage:** Extract 85%+ of manually annotated fault codes
- **Precision:** 90%+ of extracted triples are valid (human eval)
- **Compression:** Graphs fit in <5MB while retaining 95%+ diagnostic paths
- **Latency:** <100ms query time on edge device (Raspberry Pi 4)

### Research Metrics
- **Novelty:** Demonstrate multimodal extraction beats text-only by 30%+
- **Scalability:** Process 1000+ manuals, validate quality on 10% sample
- **Generalization:** Multi-manual graphs improve accuracy by 15%+ vs. single-manual

### Publication Target
**Venue:** NeurIPS 2025 (Datasets & Benchmarks) or EMNLP 2025
**Contribution:**
1. Novel multimodal KG extraction for industrial manuals
2. First large-scale troubleshooting graph dataset (1M+ manuals)
3. Edge-deployable graph compression methodology
4. Benchmark for fault diagnosis reasoning

---

## 9. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Low-quality PDFs** (scanned, poor OCR) | High | Pre-filter dataset, use GPT-4V as fallback |
| **Ambiguous symptom descriptions** | Medium | Multi-expert validation, confidence scores |
| **Diagram extraction failures** | High | Manual annotation for 100-manual subset, fine-tune VLM |
| **Graph too sparse** (single manual) | Medium | Cross-reference with manufacturer knowledge bases |
| **Edge device memory limits** | Medium | Implement streaming graph queries, lazy loading |
| **LLM hallucination** (false triples) | High | Rule-based validation, dual-model consensus |

---

## 10. Next Steps (This Week)

### Immediate Actions
1. **Select pilot manual:**
   - Criteria: 100-300 pages, has fault table + diagrams, publicly available
   - Source: Internet Archive automotive/industrial category
   
2. **Build extraction pipeline MVP:**
   - Text: `pdfplumber` → semantic chunking
   - Tables: `camelot` → fault code extraction
   - Diagrams: GPT-4V API → component/connection extraction
   
3. **Prompt engineering:**
   - Design 5 extraction prompts for different manual sections
   - Test on 10-page sample, iterate until 80%+ precision
   
4. **Initial graph:**
   - Build first ManualsGraph from pilot manual
   - Visualize with `pyvis`
   - Manual validation session (mark errors)

### Weekly Goal
**Deliverable:** Working single-manual extraction pipeline + 1 validated graph
**Metric:** Extract 50+ symptom→cause→fix triples from pilot manual

---

## Conclusion

Buehler's GraphReasoning provides a **strong foundation** for scientific knowledge extraction, but industrial manuals require **significant adaptations**:

1. **Multimodal focus:** Tables and diagrams are primary data sources
2. **DAG topology:** Deterministic troubleshooting logic, not exploratory scale-free
3. **Compression:** Edge deployment demands <5MB graphs
4. **Cross-manual reasoning:** Production line context spans multiple documents

By combining Buehler's embedding + path finding techniques with novel multimodal extraction and hierarchical compression, we can create **deployable, high-fidelity troubleshooting graphs** that outperform traditional rule-based systems while remaining explainable.

**The key innovation:** Treating manuals as **multimodal structured documents** rather than unstructured text, leveraging their inherent organization (fault tables, flowcharts) to build cleaner graphs than pure LLM extraction.
