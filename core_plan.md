# POLYMIND v2: Core Development Plan

**Guiding Principle:** This plan outlines the development of POLYMIND v2, a modular Python project. Its architecture and functionality will be deeply informed by the Theoretical Foundations (Category Theory, Semiotics, Cognitive Linguistics, Embodied Cognition) and the core Architectural Pillars (Ingestion Layer, Knowledge Graph, Synthesis Engine, Interfaces) as defined in the POLYMIND vision.

---

## Phase 1: Foundational Core & Initial Text Ingestion (Architecture Focus)

**Objective:** Establish the basic project structure, core data handling modules, and a minimal pipeline for ingesting textual information into a structured knowledge graph.

**Architectural Components Targeted:**
*   Partial **Ingestion Layer** (Text-only)
*   Initial **Knowledge Graph** (Neo4j setup, basic schema)
*   Basic **Interfaces** (CLI for testing ingestion)

**Key Modules & Tasks:**

1.  **Project Setup (`polymind_v2/`)**
    *   Create directory structure (as previously outlined: `polymind_core`, `data_ingestion`, `app`, `config.py`, `requirements.txt`, `README.md`, `.gitignore`).
    *   Initialize `requirements.txt` (Python, python-dotenv, py2neo, google-generativeai, sentence-transformers, faiss-cpu, wikipedia).

2.  **Configuration (`config.py`)**
    *   Centralize API keys (Gemini, Neo4j credentials).
    *   Define constants: `ALLOWED_RELATIONSHIP_TYPES`, `_REL_MAP`, default domains, embedding model names.

3.  **Core Module: Knowledge Graph (`polymind_core/knowledge_graph/`)**
    *   `graph_handler.py`:
        *   Neo4j connection management (reusable function).
        *   Function: `add_concept(name: str, domain: str, properties: Optional[dict] = None)` - Uses `MERGE` on `Concept` nodes.
        *   Function: `add_relationship(source_name: str, rel_type: str, target_name: str, properties: Optional[dict] = None)` - `MERGE`s concepts if they don't exist, then `MERGE`s the relationship.
        *   Function: `get_all_concept_names()`
        *   Function: `get_concept_features(concept_name: str, max_features: int = 10)` - Basic 1-hop retrieval (similar to existing).

4.  **Core Module: Meaning Extraction (Text) (`polymind_core/meaning_extraction/`)**
    *   `text_processor.py`:
        *   Initialize Gemini model.
        *   Function: `extract_meaning_atoms_from_text(text_content: str)` -> `(concepts: List[str], triples: List[Tuple[str, str, str]], raw_output: str)` (refactor from `polymind_app.py`, including concept normalization and relationship canonicalization using `config.py` constants).
        *   Function: `suggest_domains_for_concepts(concepts: List[str])` -> `Dict[str, str]` (refactor from `polymind_app.py`).

5.  **Core Module: Vector Store (`polymind_core/vector_store/`)**
    *   `embedding_manager.py`:
        *   Function: `initialize_embedding_model(model_name: str)` (from `config.py`).
        *   Function: `generate_embeddings(texts: List[str], model)` -> `np.ndarray`.
    *   `faiss_manager.py`:
        *   Function: `create_faiss_index(embeddings: np.ndarray)` -> `faiss.Index`.
        *   Function: `add_to_faiss_index(index, new_embeddings: np.ndarray)`.
        *   Function: `search_faiss_index(index, query_embedding: np.ndarray, k: int)` -> `(distances, indices)`.

6.  **Initial Data Ingestion Script (`data_ingestion/wikipedia_ingestor.py`)**
    *   **Objective:** Ingest content from a *few* predefined Wikipedia articles.
    *   Use `wikipedia` library to fetch article title, summary.
    *   Use `text_processor` to extract atoms and suggest domains.
    *   Use `graph_handler` to load concepts and triples into Neo4j.
    *   **No FAISS integration in this initial script yet.**
    *   Provide a simple `if __name__ == "__main__":` to run this script.

**Theoretical Foundation Considerations (Initial Touchpoints):**
*   **Semiotics:** Concept normalization in `text_processor` is a first step towards distinguishing signifier (raw text) from signified (normalized concept string).
*   **Cognitive Linguistics:** The idea of (Subject, Relationship, Object) triples is a basic schema, a precursor to more complex frames.

---

## Phase 2: Vector Search, Basic Synthesis & Wikipedia Pipeline Expansion

**Objective:** Integrate vector search capabilities, implement initial synthesis algorithms, and expand the Wikipedia ingestion pipeline for broader data collection.

**Architectural Components Targeted:**
*   Enhanced **Knowledge Graph** (integrated with vector store)
*   Initial **Synthesis Engine** (Analogy, Blending)
*   Enhanced **Ingestion Layer** (more robust Wikipedia pipeline)

**Key Modules & Tasks:**

1.  **Vector Store Integration:**
    *   Modify `data_ingestion/wikipedia_ingestor.py`:
        *   After ingesting a batch of articles, fetch all unique concepts from Neo4j (`graph_handler`).
        *   Generate embeddings for these concepts (`embedding_manager`).
        *   Build/update a FAISS index and save it (along with the concept list for mapping indices to names) (`faiss_manager`).
    *   `polymind_core/vector_store/faiss_manager.py`: Add functions to save/load FAISS index and concept list.

2.  **Core Module: Synthesis Engine (`polymind_core/synthesis_engine/`)**
    *   `analogy_finder.py`:
        *   Refactor `find_analogous_concept_vector` (using `faiss_manager` and `embedding_manager`).
        *   Refactor `find_analogous_concept_structural_graph` (using `graph_handler`).
        *   Refactor `generate_analogy_explanation_gemini`.
    *   `conceptual_blender.py`:
        *   Refactor `blend_concepts_gemini` (using `graph_handler` for feature retrieval).

3.  **Enhance `data_ingestion/wikipedia_ingestor.py`:**
    *   Implement fetching articles by iterating through specified Wikipedia Categories.
    *   Add batching for processing and FAISS updates (e.g., process 50 articles, then update FAISS).
    *   Improve logging and error handling.

**Theoretical Foundation Considerations:**
*   **Cognitive Linguistics (Conceptual Blending):** `conceptual_blender.py` directly implements this.
*   **Semiotics (Analogy):** `analogy_finder.py` looks for similarities, which is a form of relating signs/concepts.

---

## Phase 3: Streamlit UI & Multimodal Foundations

**Objective:** Develop the Streamlit user interface for interacting with the core system and lay the groundwork for future multimodal ingestion.

**Architectural Components Targeted:**
*   Full **Interfaces** (Streamlit UI)
*   Planning for advanced **Ingestion Layer** (Multimodal)

**Key Modules & Tasks:**

1.  **Streamlit Application (`app/`)**
    *   `main_app.py`: Main Streamlit application file. Handles page configuration, session state initialization (connections to Neo4j, Gemini, loading embedding model, FAISS index).
    *   `pages/`:
        *   `01_Knowledge_Ingestion.py`: UI for manual text input, calls `text_processor` and `graph_handler`. Option to trigger Wikipedia ingestion (subset for demo).
        *   `02_Analogical_Search.py`: UI for analogy queries, calls `analogy_finder`. Displays results and graph visualizations (porting from `polymind_app.py`).
        *   `03_Conceptual_Blending.py`: UI for blending, calls `conceptual_blender`. Displays results.
        *   `04_Graph_Explorer.py`: Basic UI to browse some nodes/relationships from Neo4j.
    *   Ensure UI components appropriately call functions from `polymind_core`.

2.  **Multimodal Ingestion - Scaffolding (`polymind_core/meaning_extraction/`)**
    *   `image_processor.py` (Placeholder):
        *   Define function signatures for `extract_features_from_image(image_path_or_bytes)` (e.g., using CLIP).
        *   Think about how image features/embeddings would be stored in Neo4j (e.g., as properties on existing nodes, or linking to new `ImageAtom` nodes).
    *   Update `requirements.txt`: Add `clip-openai`, `Pillow`.

**Theoretical Foundation Considerations:**
*   **Embodied Cognition (Multimodal):** Starting to think about how image (visual sensorimotor) data will be integrated. Even just storing an image embedding alongside a concept node is a first step.

---

## Phase 4: Richer Meaning Atoms & Advanced Synthesis (Ongoing Research & Development)

**Objective:** Progress towards the deeper theoretical goals by enriching the "meaning atom" representation and exploring more advanced synthesis techniques. This phase is more research-oriented and iterative.

**Architectural Components Targeted:**
*   Advanced **Ingestion Layer** (Schema Mappers, True Multimodal)
*   Richer **Knowledge Graph** (Schema reflecting theoretical foundations)
*   Advanced **Synthesis Engine** (Category Theory applications, Frame-based reasoning)

**Key Research & Development Areas:**

1.  **Richer Meaning Atom Representation:**
    *   **Knowledge Graph Schema Evolution (`polymind_core/knowledge_graph/`):**
        *   How to represent Semiotic components (signifier, signified, Peirce's icon/index/symbol) in Neo4j? E.g., A `Concept` node (signified) can have multiple `Signifier` nodes (text in different languages, image representations).
        *   How to represent Cognitive Linguistic Frames? E.g., A "RestaurantVisit" `FrameNode` linked to `RoleNodes` (Customer, Waiter) and `ObjectNodes` (Food, Menu).
        *   How to link to Embodied Cognition prototypes? E.g., A `Concept` node having a property `visual_embedding_clip` or a relationship to a `VisualPrototype` node.
    *   **Ingestion Layer Enhancement (`polymind_core/meaning_extraction/`):**
        *   Develop/integrate schema mappers to translate incoming data into these richer structures.
        *   Implement `image_processor.py` to extract and store CLIP embeddings.

2.  **Advanced Synthesis Engine (`polymind_core/synthesis_engine/`)**
    *   Explore rule-based or LLM-driven methods for identifying structural isomorphisms between subgraphs for analogy.
    *   Investigate how Category Theory concepts (e.g., functors for mapping between ologs/schemas) could be practically implemented for synthesis.
    *   Develop methods for "subgraph merging" beyond simple conceptual blending.

3.  **Interface Enhancements (`app/`)**
    *   Develop more dynamic graph visualization and exploration tools.
    *   Consider a LangChain-powered conversational interface component.

**Theoretical Foundation Considerations (Deep Dive):**
*   **All four foundations** will heavily guide the design choices in this phase. For instance, how do we represent the "restaurant frame" (Cognitive Linguistics) in Neo4j, link its components to visual prototypes (Embodied Cognition), ensure its conceptual representation is language-neutral (Semiotics), and define its structure formally (Category Theory)?

---

## Cross-Cutting Concerns (Throughout all Phases)

*   **Testing (`tests/`):** Implement unit tests for core logic and integration tests for pipelines.
*   **Documentation (`README.md`, docstrings):** Maintain clear documentation.
*   **Ethical Considerations (Ongoing Thought):** As the system becomes more capable, particularly in synthesis, revisit ethical implications (bias, misuse) as outlined in the research.
*   **Modularity:** Continuously ensure that new code is placed in appropriate modules and that interfaces between modules are clean.

This plan provides a roadmap. Each phase, especially Phase 4, will involve significant iteration and refinement based on research and experimentation. 