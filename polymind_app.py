import spacy
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from py2neo import Graph, Node, Relationship, Subgraph
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import re
import os
import google.generativeai as genai
import logging
from textwrap import dedent
import time
import json
from enum import Enum
from dotenv import load_dotenv
from typing import Iterable, List, Dict, Set, Tuple, Optional

# --- Constants for Schema (Restoring from user patch / previous state) ---
ALLOWED_RELATIONSHIP_TYPES: set[str] = {
    "IS_A",               # taxonomic
    "PART_OF",            # mereology
    "HAS_PROPERTY",       # attributes / qualities
    "CAUSES",             # causal
    "ENABLES",            # pre-condition / affordance
    "USED_FOR",           # functional purpose
    "LOCATED_IN",         # spatial containment
    "EXHIBITS",           # displays behaviour / phenotype
    "HAS_COMPONENT",      # direct compositional link
    "OCCURS_AROUND",      # contextual co-occurrence
    "OBSERVED_IN",        # empirical observation site
    "CONCEPTUALLY_RELATED_TO",  # broad conceptual link
    "RELATED_TO"          # catch-all (use sparingly)
}

_REL_MAP = {
    # Causal variants
    "LEADS_TO": "CAUSES",
    "RESULTS_IN": "CAUSES",
    "CAUSE": "CAUSES",
    # Part/whole synonyms
    "COMPONENT_OF": "PART_OF",
    "BELONGS_TO": "PART_OF",
    # Attribute variants
    "HAS_ATTRIBUTE": "HAS_PROPERTY",
    "IS_CHARACTERIZED_BY": "HAS_PROPERTY",
    # Spatial
    "IN": "LOCATED_IN",
    "INSIDE": "LOCATED_IN",
    # Fallback for common variations that Gemini might produce if it slips
    "IS_RELATED_TO": "RELATED_TO",
    "RELATED": "RELATED_TO",
    "IS_A_TYPE_OF": "IS_A",
    "TYPE_OF": "IS_A",
    "PART": "PART_OF"
}
# --- END Constants for Schema ---

# --- Streamlit UI Setup (MUST BE FIRST STREAMLIT COMMANDS) ---
st.set_page_config(layout="wide")
st.sidebar.title("POLYMIND Options")

# --- Configuration & Setup ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    GEMINI_API_KEY = "AIzaSyDuSjazaZTKq4TJJGnpz8P8IvDOFqOA3cc"

# gemini_model = None # Initialize gemini_model to None - Replaced by session state
if GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_API_KEY_HERE":
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model_instance = genai.GenerativeModel("gemini-1.5-flash-latest")
        st.session_state.gemini_model = model_instance # Store in session state
        print("Gemini API configured successfully.")
        st.session_state.gemini_api_configured_successfully = True 
    except Exception as e:
        print(f"Error configuring Gemini API: {e}")
        st.session_state.gemini_api_configured_successfully = False
        st.session_state.gemini_model = None # Ensure it's None in session state
else:
    print("Gemini API Key not set or is a placeholder.")
    st.session_state.gemini_api_configured_successfully = False
    st.session_state.gemini_model = None # Ensure it's None in session state

# Neo4j Credentials (replace with your actual credentials or environment variables)
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "URDJ632F8AnqZYkJCku500_8eYNOR1iy-QGXMNyuGsI" # Updated with provided password

# Domain specific concept lists (to be populated based on Section 5)
biology_list = [ # Example, will be expanded later
    "DNA", "Cell", "Evolution", "Gene", "Protein", "Neuron", "RNA", "Immune system", "Metabolism", "Species"
]
philosophy_list = [ # Example, will be expanded later
    "Meme", "Dualism", "Metaphysics", "Epistemology", "Dialectic", "Consciousness", "Idea", "Logic", "Mind–body problem", "Ethics"
]

# Load a lightweight English model for spaCy
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy en_core_web_sm model...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Global variable for Neo4j graph connection
# graph_db_connection = None # Replaced by session state

# Global placeholder for SentenceTransformer model
# embed_model = None # Replaced by session state

# Global placeholders for FAISS index and related data
# faiss_index = None # Replaced by session state
# These lists/arrays will store data aligned with the FAISS index
# concept_list_for_faiss should store the actual concept strings
# embeddings_for_faiss should store the embeddings
# domains_for_faiss should store the domain for each concept at the same index
# concept_list_for_faiss = [] # Replaced by session state
# embeddings_for_faiss = None # Replaced by session state
# domains_for_faiss = [] # Replaced by session state

# --- Sample Data Definition (can be expanded or moved) ---
# Defined globally to be accessible by init and potentially __main__ if needed elsewhere.
SAMPLE_CONCEPTS_TO_LOAD = list(set([
    c.capitalize() for c in [
        "DNA", "Cell", "Evolution", "Gene", "Protein", "Neuron", "RNA", "Immune system", "Metabolism", "Species",
        "Meme", "Dualism", "Metaphysics", "Epistemology", "Dialectic", "Consciousness", "Idea", "Logic", "Mind–body problem", "Ethics",
        "genetic instructions", "living organisms", "unit of cultural information", 
        "Monism", "basic unit of life", "elementary individual substance"
    ]
]))

SAMPLE_TRIPLES_TO_LOAD = [
    (s.capitalize(), r, o.capitalize()) for s, r, o in [
        ("DNA", "CARRIES", "genetic instructions"), ("DNA", "IN", "living organisms"),
        ("Meme", "IS_A", "unit of cultural information"), ("DNA", "CODES_FOR", "Protein"),
        ("Meme", "SUBCONCEPT_OF", "Idea"), ("Evolution", "APPLIES_TO", "Species"),
        ("Dualism", "CONTRASTS_WITH", "Monism"),
        ("Cell", "IS_A", "basic unit of life"), 
        ("Monad", "IS_A", "elementary individual substance")
    ]
]

biology_list = [c.capitalize() for c in biology_list]
philosophy_list = [c.capitalize() for c in philosophy_list]

# Ensure all subjects/objects from triples are in concepts for completeness
for s,_,o in SAMPLE_TRIPLES_TO_LOAD:
    if s not in SAMPLE_CONCEPTS_TO_LOAD: SAMPLE_CONCEPTS_TO_LOAD.append(s)
    if o not in SAMPLE_CONCEPTS_TO_LOAD: SAMPLE_CONCEPTS_TO_LOAD.append(o)
for concept in biology_list:
    if concept not in SAMPLE_CONCEPTS_TO_LOAD: SAMPLE_CONCEPTS_TO_LOAD.append(concept)
for concept in philosophy_list:
    if concept not in SAMPLE_CONCEPTS_TO_LOAD: SAMPLE_CONCEPTS_TO_LOAD.append(concept)
SAMPLE_CONCEPTS_TO_LOAD = sorted(list(set(SAMPLE_CONCEPTS_TO_LOAD))) # Remove duplicates and sort

def connect_to_neo4j():
    """Establishes a connection to the Neo4j database and stores it in session_state."""
    # global graph_db_connection # No longer using global for this
    print(f"Attempting to connect to Neo4j with URI: {NEO4J_URI}, User: {NEO4J_USER}")
    # st.info(f"Attempting to connect to Neo4j with URI: {NEO4J_URI}, User: {NEO4J_USER}. Check console for password if debugging.") # Reduced verbosity
    try:
        conn = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        conn.run("MATCH (n) RETURN count(n)").data()
        print("Successfully connected to Neo4j.")
        st.session_state.graph_db_connection = conn
        return conn
    except Exception as e:
        print(f"Failed to connect to Neo4j: {e}")
        st.error(f"Failed to connect to Neo4j: {e}. Please ensure Neo4j is running and credentials are correct.")
        st.session_state.graph_db_connection = None
        return None

# Placeholder for FAISS index and related data
# faiss_index = None # Managed by session_state
# concept_list_for_faiss = [] # Managed by session_state
# embeddings_for_faiss = None # Managed by session_state
# domains_for_faiss = [] # Managed by session_state

# Placeholder for SentenceTransformer model
# embed_model = None # Managed by session_state

# --- Section 3.1: Ingest Text and Extract Key Concepts ---
def extract_concepts_spacy(text):
    """Extracts concepts using spaCy."""
    doc = nlp(text)
    concepts = set()
    for np in doc.noun_chunks:
        phrase = np.text.strip()
        if len(phrase) > 2 and not np.root.is_stop:
            concepts.add(phrase)
    for ent in doc.ents:
        concepts.add(ent.text.strip())
    print(f"Extracted concepts (spaCy): {concepts}")
    return list(concepts)

def extract_meaning_atoms_gemini(text_content, gemini_model_to_use): # Renamed and updated signature
    """
    Extracts concepts and relationships from text using the Gemini API.
    Returns a tuple: (raw_gemini_output_text, list_of_unique_concepts, list_of_triples)
    Each triple is (subject, relationship, object).
    """
    if not gemini_model_to_use: # Use passed model
        st.error("Gemini model not provided or not initialized. Cannot extract from text.")
        print("Gemini model not provided. Skipping extraction.")
        return "", [], [] # Return three values

    prompt = f"""Extract all significant concepts and their relationships from the following text.
Present the relationships strictly in the format:
(Subject; Relationship; Object)
Each triple should be on a new line. Do not use any other formatting.

Example:
(DNA; CARRIES; genetic instructions)
(Cell; IS_A; basic unit of life)

Text:
{text_content}

Extracted Triples:
"""
    try:
        print(f"Sending text to Gemini for extraction (length: {len(text_content)} chars)")
        response = gemini_model_to_use.generate_content(prompt) # Use passed model
        
        extracted_text = response.text
        print(f"Gemini response: {extracted_text}")
        # st.text_area("Gemini Raw Output:", extracted_text, height=150)

        concepts = set()
        triples = []
        
        # Basic parsing of the (Subject; Relationship; Object) format
        for line in extracted_text.split('\n'):
            line = line.strip()
            if line.startswith("(") and line.endswith(")") and ";" in line:
                try:
                    # Remove parentheses and split
                    content = line[1:-1]
                    parts = [p.strip() for p in content.split(';')]
                    if len(parts) == 3:
                        subject, relationship, object_ = parts[0], parts[1], parts[2]
                        if subject and relationship and object_: # Ensure no empty parts
                            # Normalize concept names (e.g., capitalize first word, or title case for multi-word)
                            # Using .capitalize() as a simple approach for now.
                            # More sophisticated title casing might be needed if concepts are multi-word phrases
                            # where only the first word should be capitalized, or if acronyms are present.
                            # For simplicity, let's try a robust title() and then strip.
                            norm_subject = subject.title().strip()
                            norm_object_ = object_.title().strip()
                            # Relationship types are often better in UPPER_SNAKE_CASE
                            norm_relationship = relationship.upper().replace(" ", "_").replace("-", "_").strip()

                            if norm_subject and norm_relationship and norm_object_:
                                triples.append((norm_subject, norm_relationship, norm_object_))
                                concepts.add(norm_subject)
                                concepts.add(norm_object_)
                            else:
                                print(f"Skipping triple due to empty parts after normalization: Original ({subject}, {relationship}, {object_})")
                    else:
                        print(f"Skipping malformed triple (not 3 parts): {line}")
                except Exception as e:
                    print(f"Error parsing triple line '{line}': {e}")
            elif line: # Non-empty line that doesn't match format
                print(f"Skipping non-triple line from Gemini output: {line}")
        
        print(f"Gemini extracted concepts: {concepts}")
        print(f"Gemini extracted triples: {triples}")
        return extracted_text, list(concepts), triples # Return three values

    except Exception as e:
        print(f"Error calling Gemini API or parsing response: {e}")
        st.error(f"Error during Gemini knowledge extraction: {e}")
        return "", [], [] # Return three values

def extract_relationships_llm(text_chunk, llm_pipeline):
    # This function is now effectively replaced by extract_meaning_atoms_gemini
    # It can be removed or kept for other LLM experiments if needed.
    print("extract_relationships_llm is deprecated. Use extract_meaning_atoms_gemini.")

# --- Section 3.2: Loading Concepts into a Graph Database (Neo4j) ---
def get_concept_domain(concept_name):
    """Determines the domain of a concept, prioritizing Neo4j, then lists, then default."""
    # Priority 1: Check Neo4j if a connection is available
    graph_conn = st.session_state.get("graph_db_connection")
    if graph_conn:
        try:
            result = graph_conn.run("MATCH (c:Concept {name: $name}) RETURN c.domain AS domain", name=concept_name).data()
            if result and result[0]['domain']:
                # print(f"Domain for '{concept_name}' from Neo4j: {result[0]['domain']}")
                return result[0]['domain']
        except Exception as e:
            print(f"Error querying domain for '{concept_name}' from Neo4j: {e}. Will fallback.")

    # Priority 2: Check predefined lists (fallback or for concepts not yet in DB)
    if concept_name in biology_list:
        return "Biology"
    elif concept_name in philosophy_list:
        return "Philosophy"
    
    # Add more domains or a default strategy if needed
    # print(f"Domain for '{concept_name}' not found in Neo4j or lists. Defaulting to General.")
    return "General" # Default domain

def load_concepts_to_neo4j(graph, concept_domain_map): # Modified signature
    """Loads concepts and their domains into Neo4j as Concept nodes."""
    if not graph:
        print("Neo4j connection not available. Skipping concept loading.")
        return 0 # Return count of loaded concepts
    
    loaded_count = 0
    tx = graph.begin()
    for concept_name, domain in concept_domain_map.items(): # Iterate through dict
        # Using MERGE to avoid duplicates if run multiple times
        # Ensure domain is a string, default to "General" if None/empty for safety
        effective_domain = domain if domain else "General" 
        tx.run("MERGE (c:Concept {name: $name}) SET c.domain = $domain", 
               name=concept_name, 
               domain=effective_domain)
        loaded_count += 1
    try:
        graph.commit(tx)
        print(f"Successfully loaded/merged {loaded_count} concepts into Neo4j with domains.")
        return loaded_count
    except Exception as e:
        print(f"Error committing concepts to Neo4j: {e}")
        graph.rollback(tx)
        return 0

def load_triples_to_neo4j(
    graph_conn, 
    triples: Iterable[list[str]], # Changed from tuple to list for consistency
    node_label: str = "Concept"
) -> int:
    """
    Loads a list of triples into Neo4j, enforcing the schema for relationships.
    Uses the _canonical_rel helper to map relationships to allowed types.
    Args:
        graph_conn: Active Neo4j graph connection.
        triples (Iterable[list[str]]): An iterable of [Subject, Relationship, Object] lists.
        node_label (str): The Neo4j label to use for concept nodes.
    Returns:
        int: The number of triples successfully loaded.
    """
    loaded_count = 0
    if not graph_conn:
        logging.error("Neo4j graph connection not available in load_triples_to_neo4j.")
        st.error("Neo4j connection not available. Cannot load triples.")
        return 0

    tx = None
    try:
        tx = graph_conn.begin()
        for subj, rel_raw, obj in triples: # rel_raw to signify it's before canonicalization
            # Concept names (subj, obj) are assumed to be normalized by extract_meaning_atoms_gemini
            
            # Get canonical relationship type
            rel_type = _canonical_rel(rel_raw) 
            
            # Ensure rel_type for Neo4j is a valid identifier (no spaces unless backticked)
            # Our ALLOWED_RELATIONSHIP_TYPES are underscore_separated or single words, so this is fine.
            
            query = f"""
            MERGE (s:{node_label} {{name: $subj}})
            MERGE (o:{node_label} {{name: $obj}})
            MERGE (s)-[r:{rel_type}]->(o)
            """
            tx.run(query, subj=subj, obj=obj)
            loaded_count += 1
        graph_conn.commit(tx)
        logging.info(f"Successfully loaded {loaded_count} triples into Neo4j.")
        if loaded_count > 0:
            st.success(f"Loaded {loaded_count} new relationships into the graph.")
    except Exception as e:
        error_msg = f"Error loading triples to Neo4j: {e}"
        logging.error(error_msg)
        st.error(error_msg)
        if tx:
            try:
                graph_conn.rollback(tx)
                logging.info("Rolled back transaction due to error.")
            except Exception as rb_e:
                logging.error(f"Error during transaction rollback: {rb_e}")
        return 0 # Return 0 on error
        
    return loaded_count

# --- Section 3.3: Generating Embeddings for Concepts ---
def initialize_embedding_model(model_name='sentence-transformers/all-MiniLM-L6-v2'):
    """Initializes and returns a SentenceTransformer model, storing it in session_state."""
    # global embed_model # No longer using global for this
    try:
        model = SentenceTransformer(model_name)
        print(f"Successfully initialized embedding model: {model_name}")
        st.session_state.embed_model = model
        return model
    except Exception as e:
        print(f"Error initializing embedding model: {e}")
        st.error(f"Error initializing embedding model: {e}")
        st.session_state.embed_model = None
        return None

def generate_embeddings(concepts_to_embed, model):
    """Generates embeddings for a list of concept strings using the provided model."""
    if not model:
        print("Embedding model not available. Skipping embedding generation.")
        return None
    if not concepts_to_embed:
        print("No concepts provided for embedding.")
        return None
    
    print(f"Generating embeddings for {len(concepts_to_embed)} concepts...")
    try:
        embeddings = model.encode(concepts_to_embed, convert_to_numpy=True)
        print(f"Generated embeddings of shape: {embeddings.shape}")
        return embeddings
    except Exception as e:
        print(f"Error during embedding generation: {e}")
        return None

def build_faiss_index(concept_names, concept_embeddings, concept_domains_list):
    """Builds a FAISS index and stores it and associated data in session_state."""
    # global faiss_index, concept_list_for_faiss, embeddings_for_faiss, domains_for_faiss # No longer using globals

    if concept_embeddings is None or len(concept_embeddings) == 0:
        print("No embeddings provided to build FAISS index.")
        st.session_state.faiss_index = None
        st.session_state.concept_list_for_faiss = []
        st.session_state.embeddings_for_faiss = None
        st.session_state.domains_for_faiss = []
        return None

    dim = concept_embeddings.shape[1]
    try:
        current_faiss_index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(concept_embeddings) 
        current_faiss_index.add(concept_embeddings)
        
        st.session_state.faiss_index = current_faiss_index
        st.session_state.concept_list_for_faiss = list(concept_names)
        st.session_state.embeddings_for_faiss = concept_embeddings
        st.session_state.domains_for_faiss = list(concept_domains_list)

        print(f"Successfully built FAISS index with {current_faiss_index.ntotal} concept vectors.")
        return current_faiss_index
    except Exception as e:
        print(f"Error building FAISS index: {e}")
        st.session_state.faiss_index = None
        st.session_state.concept_list_for_faiss = []
        st.session_state.embeddings_for_faiss = None
        st.session_state.domains_for_faiss = []
        return None

# --- Section 3.4: Analogical Search (Finding Similar Concepts) ---
def find_analogous_concept_vector(query_concept_name, k=5):
    """Finds an analogous concept using FAISS vector similarity.
    Retrieves FAISS data and embed_model from st.session_state.
    """
    # Retrieve from session_state
    current_faiss_index = st.session_state.get('faiss_index')
    current_concept_list = st.session_state.get('concept_list_for_faiss', [])
    current_embeddings = st.session_state.get('embeddings_for_faiss')
    current_domains_list = st.session_state.get('domains_for_faiss', [])
    current_embed_model = st.session_state.get('embed_model')

    print(f"[find_analogous_concept_vector] Called for query: {query_concept_name}")
    print(f"[find_analogous_concept_vector] embed_model is None: {current_embed_model is None}")
    print(f"[find_analogous_concept_vector] faiss_index is None: {current_faiss_index is None}")
    print(f"[find_analogous_concept_vector] concept_list_for_faiss is empty: {not current_concept_list}")

    if not current_faiss_index or not current_concept_list or current_embed_model is None:
        print("FAISS index, concept list, or embedding model not initialized (retrieved from session_state).")
        st.error("(Debug from func) FAISS/model not initialized from session_state.")
        return None

    if query_concept_name not in current_concept_list:
        print(f"Query concept '{query_concept_name}' not in FAISS list. Attempting to embed on-the-fly.")
        q_vec = current_embed_model.encode([query_concept_name], convert_to_numpy=True)
        query_domain = get_concept_domain(query_concept_name)
    else:
        q_idx = current_concept_list.index(query_concept_name)
        q_vec = current_embeddings[q_idx:q_idx+1]
        query_domain = current_domains_list[q_idx]

    faiss.normalize_L2(q_vec)
    distances, indices = current_faiss_index.search(q_vec, k + 1) 

    analogous_candidates = []
    for i in range(len(indices[0])):
        neighbor_idx = indices[0][i]
        neighbor_concept_name = current_concept_list[neighbor_idx]
        neighbor_domain = current_domains_list[neighbor_idx]
        similarity_score = distances[0][i]

        if neighbor_concept_name != query_concept_name and neighbor_domain != query_domain:
            analogous_candidates.append({
                "concept": neighbor_concept_name, 
                "domain": neighbor_domain, 
                "similarity": similarity_score
            })
            if len(analogous_candidates) >= k:
                break
    
    if analogous_candidates:
        return analogous_candidates[0] 
    return None

def find_analogous_concept_graph(graph_conn, query_concept_name, target_domain, max_hops=3): # graph_conn from session_state
    """Finds an analogous concept using Neo4j graph traversal (shortest path)."""
    if not graph_conn: # Check graph_conn passed from session_state
        print("Neo4j connection not available (from session_state).")
        return None

    query_concept_domain = get_concept_domain(query_concept_name)
    if not query_concept_domain or query_concept_domain == target_domain:
        print(f"Query concept '{query_concept_name}' domain invalid or same as target domain '{target_domain}'.")
        return None

    # Cypher query to find the shortest path to a concept in the target domain
    cypher_query = f"""
    MATCH (start:Concept {{name: $name, domain: $start_domain}}),
          (target:Concept {{domain: $target_dom}}),
          p = shortestPath((start)-[*..{max_hops}]-(target))
    WHERE start <> target  // Ensure start and target are not the same node
    RETURN target.name AS analogous_name, target.domain AS analogous_domain, length(p) AS hops
    ORDER BY hops
    LIMIT 1
    """
    try:
        result = graph_conn.run(cypher_query, 
                             name=query_concept_name, 
                             start_domain=query_concept_domain, 
                             target_dom=target_domain).data()
        if result:
            return {"concept": result[0]['analogous_name'], "domain": result[0]['analogous_domain'], "hops": result[0]['hops']}
        else:
            print(f"No graph path found from '{query_concept_name}' ({query_concept_domain}) to domain '{target_domain}' within {max_hops} hops.")
            return None
    except Exception as e:
        print(f"Error during graph-based analogy search for '{query_concept_name}': {e}")
        return None

# --- NEW Section 3.4.1: Advanced Graph-Based Analogical Search (Structural) ---
def find_analogous_concept_structural_graph(graph_conn, source_name, target_domain_name=None, k=5):
    """
    Finds analogous concepts by identifying shared 2-hop neighbors in the graph.
    A shared 2-hop neighbor is a concept reachable from both the source and a candidate concept in exactly two steps.

    Args:
        graph_conn: Active Neo4j graph connection.
        source_name (str): The name of the source concept.
        target_domain_name (str, optional): The specific domain to search for analogies in.
                                         If None, searches in domains different from the source.
        k (int, optional): The maximum number of analogous concepts to return.

    Returns:
        list: A list of dictionaries, where each dictionary represents an analogous concept
              and includes: 'analogous_concept', 'analogous_domain',
              'num_shared_neighbors', and 'shared_neighbor_details'.
              Returns an empty list if no analogies are found or an error occurs.
    """
    if not graph_conn:
        print("Neo4j connection not available for structural analogy search.")
        st.error("Neo4j connection not available for structural analogy search.")
        return []

    print(f"Starting structural analogy search for \'{source_name}\' (shared 2-hop neighbors), target domain: {target_domain_name}")

    # This Cypher query finds concepts that share common 2-hop neighbors with the source concept.
    # It counts how many such common neighbors exist and collects details about the paths.
    cypher_query = """
    // Part 1: Find all 2-hop neighbors of the source concept
    MATCH (source:Concept {name: $source_name})-[r1_src:RELATIONSHIP]->(mid_src:Concept)-[r2_src:RELATIONSHIP]->(shared_end_node:Concept)
    WHERE source <> shared_end_node AND source <> mid_src AND mid_src <> shared_end_node // Avoid trivial paths and self-loops in 2-hop
    WITH source, COLLECT(DISTINCT {
        neighbor: shared_end_node, // 'neighbor' IS the node 'shared_end_node'
        path_src: [source.name, r1_src.type, mid_src.name, r2_src.type, shared_end_node.name] // Changed type(r1_src) to r1_src.type and type(r2_src) to r2_src.type
    }) AS source_2hop_destinations_with_paths

    UNWIND source_2hop_destinations_with_paths AS s_2hop // s_2hop is the map {neighbor: Node, path_src: List}
    // Explicitly define the node and its path from source before the next MATCH
    WITH source, s_2hop.neighbor AS common_dest_node, s_2hop.path_src AS path_from_source_to_dest

    // Part 2: Find candidate concepts that also reach common_dest_node
    MATCH (candidate:Concept)-[r1_cand:RELATIONSHIP]->(mid_cand:Concept)-[r2_cand:RELATIONSHIP]->(common_dest_node) // Use the simple variable common_dest_node
    WHERE candidate.name <> source.name // Candidate is not the source
      AND candidate <> common_dest_node AND candidate <> mid_cand AND mid_cand <> common_dest_node // Avoid trivial paths for candidate
      // Domain filtering for the candidate:
      AND (
            (source.domain IS NULL AND candidate.domain IS NOT NULL) OR
            (candidate.domain <> source.domain AND $target_domain_name IS NULL) OR
            (candidate.domain = $target_domain_name AND candidate.domain <> source.domain)
          )

    // Group by candidate and the common_dest_node (which is the shared 2-hop neighbor)
    WITH candidate, source, common_dest_node, path_from_source_to_dest,
         COLLECT(DISTINCT { // Collect distinct paths from candidate to this common_dest_node
             path_cand_parts: [candidate.name, r1_cand.type, mid_cand.name, r2_cand.type, common_dest_node.name] // Changed type(r1_cand) to r1_cand.type and type(r2_cand) to r2_cand.type
         }) AS candidate_paths_to_common_node_list

    // Aggregate per candidate: count shared neighbors and collect their details
    WITH candidate,
         COUNT(DISTINCT common_dest_node) AS num_shared_neighbors,
         COLLECT({
             shared_node: common_dest_node.name,
             domain_shared_node: common_dest_node.domain,
             path_from_source: path_from_source_to_dest, // Path from source to this common_dest_node
             paths_from_candidate: [item IN candidate_paths_to_common_node_list | item.path_cand_parts] // Extract the list of path parts
         }) AS shared_neighbor_details_for_candidate
    ORDER BY num_shared_neighbors DESC, candidate.name ASC
    LIMIT $k_limit

    RETURN candidate.name AS analogous_concept,
           candidate.domain AS analogous_domain,
           num_shared_neighbors,
           shared_neighbor_details_for_candidate AS shared_neighbor_details
    """

    try:
        results = graph_conn.run(cypher_query,
                                 source_name=source_name,
                                 target_domain_name=target_domain_name,
                                 k_limit=k).data()

        if results:
            print(f"Found {len(results)} structural analogy candidates for \'{source_name}\' based on shared 2-hop neighbors.")
            # Example of what results might look like now:
            # [{
            # 'analogous_concept': 'Solar System', 'analogous_domain': 'Astronomy',
            # 'num_shared_neighbors': 1,
            # 'shared_neighbor_details': [{
            #         'shared_node': 'Orbits Central Body',
            #         'path_from_source': ['Atom', 'HAS_PART', 'Electron', 'TRAVELS_IN', 'Orbits Central Body'],
            #         'paths_from_candidate': [['Solar System', 'HAS_PART', 'Planet', 'TRAVELS_IN', 'Orbits Central Body']]
            #     }]
            # }]
            return results
        else:
            print(f"No structural analogies found for \'{source_name}\' (shared 2-hop neighbors) with the given criteria.")
            return []
    except Exception as e:
        print(f"Error during structural analogy search (shared 2-hop neighbors) for \'{source_name}\': {e}")
        st.error(f"Error during structural analogy search: {e}")
        return []

# --- Section 3.5: Generate Human-like Explanation for Analogy (NEW) ---
def generate_analogy_explanation_gemini(source_concept, source_domain, analogous_concept, analogous_domain, gemini_model_to_use, shared_patterns_info=None):
    """
    Generates a human-like explanation of an analogy using the Gemini API.
    Optionally includes shared structural patterns (common 2-hop neighbors and paths) if provided.
    """
    if not gemini_model_to_use:
        st.error("Gemini model not provided or not initialized. Cannot generate explanation.")
        print("Gemini model not provided for explanation generation.")
        return "Could not generate an explanation because the Gemini model is not available."

    base_prompt = f"""Please explain the analogy between the concept '{source_concept}' (from domain: {source_domain}) and the concept '{analogous_concept}' (from domain: {analogous_domain}).

Describe how these two concepts might be considered analogous. What are the key similarities in their roles, functions, or characteristics within their respective domains?
What insights or new perspectives might this analogy offer?"""

    # shared_patterns_info is now expected to be the direct list of results
    # from find_analogous_concept_structural_graph (or a single element of it if we decide to pass one)
    # For now, assume shared_patterns_info is the 'shared_neighbor_details' list for a *single* chosen analogy.

    if shared_patterns_info and isinstance(shared_patterns_info, list) and shared_patterns_info:
        # If shared_patterns_info is the full result list, let's pick the first one's details.
        # Or, if it's already the details for one analogy.
        # This part might need refinement depending on how it's called.
        # Assuming shared_patterns_info = result_item['shared_neighbor_details']

        patterns_string = "The analogy is strengthened by shared structural connections in our knowledge graph. Specifically, both concepts connect to common 'landmark' concepts via 2-hop paths:\n"
        
        # Limiting to a few examples to keep the prompt concise
        max_shared_neighbors_to_show = 3 
        
        for i, detail in enumerate(shared_patterns_info[:max_shared_neighbors_to_show]):
            shared_node = detail.get('shared_node', 'Unknown Shared Node')
            path_src_list = detail.get('path_from_source', [])
            paths_cand_list = detail.get('paths_from_candidate', []) # This is a list of paths

            path_src_str = " -> ".join(path_src_list) if path_src_list else "N/A"
            
            # Show one example path from candidate
            path_cand_str = "N/A"
            if paths_cand_list and isinstance(paths_cand_list, list) and paths_cand_list[0].get('path_cand'):
                 path_cand_str = " -> ".join(paths_cand_list[0]['path_cand'])


            patterns_string += f"{i+1}. Both '{source_concept}' and '{analogous_concept}' connect to '{shared_node}'.\n"
            patterns_string += f"     Path from '{source_concept}': {path_src_str}\n"
            patterns_string += f"     Example path from '{analogous_concept}': {path_cand_str}\n"
        
        if len(shared_patterns_info) > max_shared_neighbors_to_show:
            patterns_string += f"... and {len(shared_patterns_info) - max_shared_neighbors_to_show} other shared landmark(s).\n"

        prompt = f"""{base_prompt}

Additionally, consider the following structural similarities found in our knowledge graph when explaining the analogy:
{patterns_string}
Based on these shared structural connections and the nature of the concepts involved, please delve deeper into the conceptual blend this analogy suggests.
1. What is the core abstract principle or underlying mechanism that these shared connections to common 'landmark' concepts highlight in both domains?
2. What new insights, novel perspectives, or even hypothetical "blended concepts" or "hybrid ideas" could emerge from mapping these two structurally similar situations?
3. How does understanding this structural correspondence allow for a richer, more nuanced, or more actionable understanding of one or both concepts?
Focus on the emergent properties and the creative potential that arises from this analogical mapping, going beyond a simple statement of similarity.

Present the explanation in a clear, deeply insightful, and thought-provoking manner, weaving together the semantic and structural aspects to illuminate the power of this analogy.
"""
    else:
        prompt = f"""{base_prompt}

Present the explanation in a clear, concise, and insightful paragraph or two.
"""

    try:
        is_structural = shared_patterns_info is not None and bool(shared_patterns_info)
        print(f"Sending analogy to Gemini for explanation: {source_concept} ({source_domain}) <-> {analogous_concept} ({analogous_domain}). Structural info provided: {is_structural}")
        # if is_structural:
            # print(f"DEBUG: Structural patterns string for Gemini: {patterns_string}") # For debugging
            # print(f"DEBUG: Full prompt for Gemini: {prompt}") # For debugging
        response = gemini_model_to_use.generate_content(prompt)
        explanation_text = response.text
        print(f"Gemini explanation response: {explanation_text}")
        return explanation_text
    except Exception as e:
        print(f"Error calling Gemini API for analogy explanation: {e}")
        st.error(f"Error during Gemini explanation generation: {e}")
        return f"An error occurred while trying to generate an explanation for the analogy: {e}"

# --- Helper function to get all concepts from Neo4j ---
def get_all_concepts_from_neo4j(graph_conn):
    """Retrieves all distinct concept names from Neo4j."""
    if not graph_conn:
        print("Neo4j connection not available for fetching all concepts.")
        st.error("Neo4j connection unavailable to fetch concepts.")
        return []
    try:
        query_result = graph_conn.run("MATCH (c:Concept) RETURN DISTINCT c.name AS name").data()
        concepts = [record['name'] for record in query_result if record['name'] is not None]
        print(f"Retrieved {len(concepts)} distinct concepts from Neo4j.")
        return concepts
    except Exception as e:
        print(f"Error fetching all concepts from Neo4j: {e}")
        st.error(f"Error fetching concepts from Neo4j: {e}")
        return []

# --- NEW FUNCTION: Get Domains for Concepts using Gemini ---
def get_domains_for_concepts_gemini(concepts_list, gemini_model_to_use):
    """
    Suggests a primary domain for each concept in a list using the Gemini API.
    Returns a dictionary mapping concept_name to domain_name.
    """
    if not gemini_model_to_use:
        st.error("Gemini model not provided or not initialized. Cannot suggest domains.")
        print("Gemini model not provided for domain suggestion.")
        return {concept: "General" for concept in concepts_list} # Fallback

    if not concepts_list:
        return {}

    # Prepare the list of concepts for the prompt
    concept_bullet_list = "\n".join([f"- {concept}" for concept in concepts_list])

    prompt = f"""For each concept in the following list, suggest a single, primary academic or general domain it belongs to.
Output each concept and its domain on a new line, strictly in the format:
Concept: Domain

Example:
DNA: Biology
Metaphysics: Philosophy
Algorithm: Computer Science
Photosynthesis: Biology
Market Economy: Economics

Concepts to categorize:
{concept_bullet_list}

Categorized Concepts:
"""
    try:
        print(f"Sending {len(concepts_list)} concepts to Gemini for domain suggestion.")
        response = gemini_model_to_use.generate_content(prompt)
        
        response_text = response.text
        # print(f"Gemini domain suggestion response: {response_text}")

        domain_map = {}
        # Parse the "Concept: Domain" format
        for line in response_text.split('\n'):
            line = line.strip()
            if ": " in line: # Check for the separator
                parts = line.split(": ", 1) # Split only on the first occurrence
                if len(parts) == 2:
                    concept_name = parts[0].strip()
                    domain_name = parts[1].strip()
                    # Ensure the concept was in the original list to avoid hallucinated concepts
                    if concept_name in concepts_list:
                        domain_map[concept_name] = domain_name
                    else:
                        print(f"Gemini suggested domain for concept '{concept_name}' not in original list. Skipping.")
                else:
                    print(f"Skipping malformed domain line (not 2 parts after split): {line}")
            elif line: # Non-empty line that doesn't match format
                print(f"Skipping non-domain line from Gemini output: {line}")
        
        # For any concepts Gemini didn't provide a domain for, assign "General"
        for concept in concepts_list:
            if concept not in domain_map:
                print(f"Gemini did not provide domain for '{concept}'. Defaulting to 'General'.")
                domain_map[concept] = "General"
        
        print(f"Gemini suggested domains: {domain_map}")
        return domain_map

    except Exception as e:
        print(f"Error calling Gemini API or parsing domain response: {e}")
        st.error(f"Error during Gemini domain suggestion: {e}")
        # Fallback: assign "General" to all concepts
        return {concept: "General" for concept in concepts_list}

# --- NEW Function: Get Concept Features from Neo4j (for Blending) ---
def get_concept_features_neo4j(graph_conn, concept_name: str, max_features: int = 10) -> List[str]:
    """
    Retrieves 1-hop features (relationships and connected nodes) for a concept from Neo4j.
    Formats them as strings for use in LLM prompts.
    Args:
        graph_conn: Active Neo4j graph connection.
        concept_name (str): The name of the concept to get features for.
        max_features (int): Maximum number of features (triples) to retrieve.
    Returns:
        List[str]: A list of strings, where each string represents a feature triple.
                   e.g., ["ConceptA IS_A Entity", "ConceptA HAS_PART PartX"]
    """
    if not graph_conn:
        logging.warning(f"Neo4j connection not available for get_concept_features_neo4j('{concept_name}').")
        return []
    
    features = []
    try:
        # Query for outgoing and incoming relationships up to max_features/2 each to try and get a balance
        # Ensure relationship types are from our schema for consistency in feature description.
        query = f"""
        MATCH (c:Concept {{name: $concept_name}})-[r]->(neighbor:Concept)
        WHERE type(r) IN $allowed_types_param
        RETURN c.name AS source, type(r) AS relationship, neighbor.name AS target
        LIMIT $limit_param
        UNION ALL
        MATCH (neighbor:Concept)-[r]->(c:Concept {{name: $concept_name}})
        WHERE type(r) IN $allowed_types_param
        RETURN neighbor.name AS source, type(r) AS relationship, c.name AS target
        LIMIT $limit_param
        """
        
        # Convert set to list for Neo4j parameter
        allowed_types_list = list(ALLOWED_RELATIONSHIP_TYPES)

        results = graph_conn.run(query, 
                                 concept_name=concept_name, 
                                 limit_param=max_features, # Total limit, but applied to each part of UNION
                                 allowed_types_param=allowed_types_list
                                 ).data()
        
        for record in results:
            # Ensure all parts are strings and not None before joining
            source = str(record['source']) if record['source'] else 'Unknown'
            relationship = str(record['relationship']) if record['relationship'] else 'RELATED_TO'
            target = str(record['target']) if record['target'] else 'Unknown'
            features.append(f"{source}; {relationship}; {target}")
        
        # Deduplicate and limit again in case UNION ALL limit behavior is not exact
        unique_features = sorted(list(set(features)))
        logging.info(f"Retrieved {len(unique_features)} unique features for concept '{concept_name}'.")
        return unique_features[:max_features]

    except Exception as e:
        logging.error(f"Error retrieving features for '{concept_name}' from Neo4j: {e}")
        st.error(f"Error getting features for {concept_name}: {e}")
        return []
# --- END NEW Function ---

# --- NEW Function: Blend Concepts with Gemini ---
def blend_concepts_gemini(concept_a_name: str, concept_a_features: List[str], 
                        concept_b_name: str, concept_b_features: List[str],
                        gemini_model_to_use) -> Optional[Dict[str, any]]:
    """
    Uses Gemini to generate a conceptual blend of two concepts based on their features.

    Args:
        concept_a_name (str): Name of the first concept.
        concept_a_features (List[str]): List of feature strings for concept A.
        concept_b_name (str): Name of the second concept.
        concept_b_features (List[str]): List of feature strings for concept B.
        gemini_model_to_use: The initialized Gemini model instance.

    Returns:
        Optional[Dict[str, any]]: A dictionary containing the blended concept's details
                                  (name, description, features, emergent_properties)
                                  or None if an error occurs.
    """
    if not gemini_model_to_use:
        st.error("Gemini model not available for blending.")
        logging.error("blend_concepts_gemini: Gemini model not provided.")
        return None

    # Prepare feature strings for the prompt
    features_a_str = "\n".join([f"- {f}" for f in concept_a_features]) if concept_a_features else "No specific features provided."
    features_b_str = "\n".join([f"- {f}" for f in concept_b_features]) if concept_b_features else "No specific features provided."

    prompt = dedent(f"""
    You are a highly creative and insightful AI assistant specializing in conceptual blending and innovation.
    Your task is to blend two concepts, Concept A and Concept B, into a novel, coherent, and meaningful new concept.

    **Concept A: {concept_a_name}**
    Known features and relationships:
    {features_a_str}

    **Concept B: {concept_b_name}**
    Known features and relationships:
    {features_b_str}

    **Your Task:**
    1.  **Invent a Name:** Create a concise and imaginative name for the new blended concept that reflects both Concept A and Concept B.
    2.  **Describe the Blend:** Provide a rich description of this new blended concept. Explain how it combines aspects of Concept A and Concept B.
    3.  **Identify Key Features:** List 3-5 key features or characteristics of this blended concept. These should be specific and clearly derived from the blend.
    4.  **Highlight Emergent Properties:** Most importantly, describe any **emergent properties** or novel capabilities this blend possesses. What can the blended concept do, or what unique quality does it have, that neither Concept A nor Concept B can achieve on their own? What new niche does it fill, or what problem does it solve?
    5.  **Structure your output EXACTLY as follows, using these specific headers:**

        **Blended Concept Name:**
        [Your inventive name here]

        **Blended Concept Description:**
        [Your detailed description of the blended concept here]

        **Key Features of Blend:**
        - [Feature 1]
        - [Feature 2]
        - [Feature 3]
        (List 3-5 features)

        **Emergent Properties:**
        [Your description of emergent properties and novel capabilities here]

    Focus on creativity, coherence, and the insightful generation of emergent properties. The goal is to produce a truly novel concept.
    """)

    try:
        logging.info(f"Sending blend request to Gemini for '{concept_a_name}' and '{concept_b_name}'.")
        response = gemini_model_to_use.generate_content(prompt)
        
        if not response.candidates or not response.text:
            raw_gemini_output_text = "Gemini response was empty or blocked for blending. Check safety settings or prompt."
            logging.warning(raw_gemini_output_text)
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                 raw_gemini_output_text += f" Prompt Feedback: {response.prompt_feedback}"
            st.error(raw_gemini_output_text)
            return None

        raw_response_text = response.text.strip()
        logging.info(f"Gemini blend response received: {raw_response_text[:500]}...") 

        blended_data = {}
        current_section = None
        full_parsed_content = [] # For debugging

        for line_num, raw_line in enumerate(raw_response_text.splitlines()):
            line = raw_line.strip()
            logging.debug(f"BLEND_PARSE Line {line_num}: '{line}' | Current Section: {current_section}")

            if line.startswith("**Blended Concept Name:**"):
                current_section = "name"
                blended_data['name'] = line.replace("**Blended Concept Name:**", "").strip()
                logging.debug(f"  -> Parsed Name: {blended_data['name']}")
            elif line.startswith("**Blended Concept Description:**"):
                current_section = "description"
                # Initialize with the first part, subsequent lines will append
                blended_data['description'] = line.replace("**Blended Concept Description:**", "").strip()
                logging.debug(f"  -> Parsed Description (start): {blended_data['description']}")
            elif line.startswith("**Key Features of Blend:**"):
                current_section = "features"
                features_list = [] # Initialize here for each blend attempt
                logging.debug(f"  -> Switched to Features section")
            elif line.startswith("**Emergent Properties:**"):
                current_section = "emergent_properties"
                # Initialize with the first part, subsequent lines will append
                blended_data['emergent_properties'] = line.replace("**Emergent Properties:**", "").strip()
                logging.debug(f"  -> Parsed Emergent Properties (start): {blended_data['emergent_properties']}")
            elif current_section:
                if current_section == "name": # Should ideally be captured by the startswith
                    if not blended_data.get('name'): # If somehow missed by startswith but section is name
                        blended_data['name'] = line
                        logging.debug(f"  -> Appended to Name (fallback): {line}")
                    # else: already captured, or this line is something else.
                elif current_section == "description":
                    blended_data['description'] += " " + line 
                    logging.debug(f"  -> Appended to Description: {line}")
                elif current_section == "features" and line.startswith("-"):
                    feature_to_add = line[1:].strip()
                    if feature_to_add:
                        features_list.append(feature_to_add)
                        logging.debug(f"  -> Added Feature: {feature_to_add}")
                elif current_section == "emergent_properties":
                    blended_data['emergent_properties'] += " " + line
                    logging.debug(f"  -> Appended to Emergent Properties: {line}")
            full_parsed_content.append(line) # For debugging
        
        blended_data['features'] = features_list # Assign collected features
        blended_data['_debug_full_parsed_content'] = "\n".join(full_parsed_content) # Add for debugging output

        if not blended_data.get('name') or not blended_data.get('description'):
            logging.warning(f"Could not parse essential fields (name/description) from Gemini blend response: {raw_response_text}")
            st.warning("Could not fully parse the blend response from Gemini. The structure might be unexpected.")
            blended_data['raw_response'] = raw_response_text 
            if not blended_data.get('name'): blended_data['name'] = "Unnamed Blend"
            if not blended_data.get('description'): blended_data['description'] = raw_response_text 
        return blended_data

    except Exception as e:
        logging.error(f"Error during Gemini conceptual blending for '{concept_a_name}' and '{concept_b_name}': {e}")
        st.error(f"Error generating conceptual blend: {e}")
        return None
# --- END NEW Function ---

# --- Prompt Helper for Extractor (from user patch) ---
# ... existing code ...

# --- Tab 2: Analogical Search & Query ---
# ... (UI setup for blending already added) ...
    if st.button("Blend Concepts with Gemini", key="blend_concepts_button"):
        if concept_a_blend and concept_b_blend:
            st.info(f"Attempting to blend '{concept_a_blend}' and '{concept_b_blend}'...")
            # ... (get graph_conn_blend, gemini_model_blend) ...
            if not graph_conn_blend:
                st.error("Neo4j connection not available for blending.")
            elif not gemini_model_blend:
                st.error("Gemini model not available for blending.")
            else:
                # ... (get features_a, features_b) ...
                # Now call the blend_concepts_gemini function
                blended_result = blend_concepts_gemini(concept_a_blend, features_a, 
                                                       concept_b_blend, features_b, 
                                                       gemini_model_blend)
                
                if blended_result:
                    st.session_state.current_blend_result = blended_result # Store in session state
                    st.subheader(f"Blended Concept: {blended_result.get('name', 'Unnamed Blend')}")
                    st.markdown("**Description:**")
                    st.write(blended_result.get('description', 'No description provided.'))
                    
                    if blended_result.get('features'):
                        st.markdown("**Key Features of Blend:**")
                        for feature_item in blended_result.get('features', []):
                            st.markdown(f"- {feature_item}")
                    
                    if blended_result.get('emergent_properties'):
                        st.markdown("**Emergent Properties:**")
                        st.write(blended_result.get('emergent_properties', 'None specified.'))
                    
                    # Placeholder for "Add to Graph" button - to be implemented next
                    if st.button("Add Blended Concept to Knowledge Graph", key="add_blend_to_graph_button"):
                        st.info("Functionality to add blend to graph to be implemented.")
                        # Call a new function: add_blended_concept_to_graph(graph_conn_blend, blended_result, concept_a_blend, concept_b_blend)

                else:
                    st.error("Failed to generate a blend from Gemini.")
                    st.session_state.current_blend_result = None
        else:
            st.warning("Please enter both Concept A and Concept B for blending.")
    # --- END Conceptual Blending Section ---
# ... existing code ...

# --- Initialization code for Streamlit app (run once at the start) ---
if 'app_initialized' not in st.session_state:
    print("--- POLYMIND Application Initializing --- ")
    # Initialize Neo4j Connection - stores in st.session_state.graph_db_connection
    connect_to_neo4j() 
    
    # Retrieve for immediate use if needed, primarily for checks here
    current_graph_db_connection = st.session_state.get('graph_db_connection')

    # Initialize Gemini Model (already done globally, but check if successful)
    if st.session_state.get('gemini_model') is None:
        st.warning("Gemini API client could not be initialized. Text ingestion features will be limited.")

    if current_graph_db_connection is None:
        st.error("CRITICAL: Failed to connect to Neo4j. Some functionalities will be unavailable.")
    else:
        try:
            concept_count_result = current_graph_db_connection.run("MATCH (c:Concept) RETURN count(c) as count").data()
            if concept_count_result and concept_count_result[0]['count'] == 0:
                print("Neo4j is empty. Loading sample data for initialization...")
                st.info("Neo4j database appears empty. Loading sample concepts and relationships for demonstration.")
                
                # Create a domain map for sample concepts
                sample_concept_domain_map = {}
                for concept_name in SAMPLE_CONCEPTS_TO_LOAD:
                    if concept_name in biology_list:
                        sample_concept_domain_map[concept_name] = "Biology"
                    elif concept_name in philosophy_list:
                        sample_concept_domain_map[concept_name] = "Philosophy"
                    else:
                        sample_concept_domain_map[concept_name] = "General"
                
                # Load concepts with their domains
                load_concepts_to_neo4j(current_graph_db_connection, sample_concept_domain_map)
                # Load triples (relationships will be canonicalized by load_triples_to_neo4j)
                load_triples_to_neo4j(current_graph_db_connection, SAMPLE_TRIPLES_TO_LOAD)
                print("Sample data loaded into Neo4j.")
                st.success("Sample data loaded into Neo4j for first-time setup.")
            else:
                print(f"Neo4j contains {concept_count_result[0]['count']} concepts. Skipping sample data load.")
        except Exception as e:
            st.warning(f"Could not check or load sample data into Neo4j: {e}")
            print(f"Neo4j check/sample data load error: {e}")

    # Initialize Embedding Model - stores in st.session_state.embed_model
    initialize_embedding_model()
    current_embed_model = st.session_state.get('embed_model')

    if current_embed_model is None:
        st.error("CRITICAL: Failed to initialize embedding model. Vector search will be unavailable.")

    if current_graph_db_connection and current_embed_model:
        try:
            query_result = current_graph_db_connection.run("MATCH (c:Concept) RETURN DISTINCT c.name AS name").data()
            concepts_from_db = [record['name'] for record in query_result if record['name'] is not None]
            if concepts_from_db:
                print(f"Streamlit Init: Building FAISS index with {len(concepts_from_db)} concepts from DB...")
                concept_embeddings_generated = generate_embeddings(concepts_from_db, current_embed_model) # Pass model
                if concept_embeddings_generated is not None:
                    current_concept_domains = [get_concept_domain(c) for c in concepts_from_db]
                    # build_faiss_index stores results in st.session_state
                    build_faiss_index(concepts_from_db, concept_embeddings_generated, current_concept_domains)
                    
                    # Check st.session_state for faiss_index after build attempt
                    if st.session_state.get('faiss_index'):
                        st.success(f"FAISS index built/loaded with {st.session_state.get('faiss_index').ntotal} concepts.")
                        print(f"Streamlit Init: FAISS index built with {st.session_state.get('faiss_index').ntotal} concepts.")
                    else:
                        st.warning("FAISS index could not be built during Streamlit initialization (check logs).")
                        print("Streamlit Init: FAISS index could not be built (from build_faiss_index).")
            else:
                st.warning("No concepts found in Neo4j to build FAISS index for Streamlit app.")
                print("Streamlit Init: No concepts found in Neo4j to build FAISS index.")
        except Exception as e:
            st.error(f"Error during FAISS index creation from DB for Streamlit app: {e}")
            print(f"Streamlit Init: Error during FAISS index creation: {e}")
            # Ensure FAISS session state vars are cleared on error
            st.session_state.faiss_index = None
            st.session_state.concept_list_for_faiss = []
            st.session_state.embeddings_for_faiss = None
            st.session_state.domains_for_faiss = []
            
    st.session_state.app_initialized = True
    print("--- POLYMIND Application Initialization Complete --- ")
else:
    # On subsequent runs (e.g. widget interaction), ensure global vars are accessible if needed
    # This part might need refinement based on how Streamlit handles global variables across reruns
    # For now, relying on the initial setup within session_state check.
    pass 


# --- Section 4: Streamlit UI for Querying and Visualization ---
# st.title("POLYMIND Chat Interface") # Removed, tabs will have titles

# Old UI Element for Text Ingestion (Basic) - REMOVED (Functionality moved to Tab 1)
# st.sidebar.header("Knowledge Ingestion")
# new_text_input = st.sidebar.text_area("Add new knowledge (paste text here):", height=150, key="new_text_area")
# if st.sidebar.button("Process and Add Text", key="process_text_button"):
#     if new_text_input and st.session_state.get('gemini_model') and st.session_state.get('graph_db_connection') and st.session_state.get('embed_model'):
#         st.sidebar.info("Processing text with Gemini... this may take a moment.")
#         # Use the correct function name and pass the model from session state
#         gemini_model_instance = st.session_state.get('gemini_model')
#         new_concepts, new_triples = extract_meaning_atoms_gemini(new_text_input, gemini_model_instance)
        
#         if new_concepts or new_triples:
#             st.sidebar.success(f"Extracted {len(new_concepts)} concepts and {len(new_triples)} relationships.")
            
#             graph_conn = st.session_state.get('graph_db_connection')
#             load_concepts_to_neo4j(graph_conn, new_concepts)
#             load_triples_to_neo4j(graph_conn, new_triples)
#             st.sidebar.success("Loaded new knowledge into Neo4j.")

#             st.sidebar.info("Rebuilding FAISS index with new data...")
#             all_concepts_from_db = get_all_concepts_from_neo4j(graph_conn) # Use helper
#             if all_concepts_from_db:
#                 embed_model_instance = st.session_state.get('embed_model')
#                 new_embeddings = generate_embeddings(all_concepts_from_db, embed_model_instance)
#                 if new_embeddings is not None:
#                     new_domains = [get_concept_domain(c) for c in all_concepts_from_db]
#                     build_faiss_index(all_concepts_from_db, new_embeddings, new_domains) # Call with correct args
#                     if st.session_state.get('faiss_index'):
#                         st.sidebar.success(f"FAISS index rebuilt with {st.session_state.get('faiss_index').ntotal} total concepts.")
#                     else:
#                         st.sidebar.error("Failed to rebuild FAISS index after adding new text.")
#                 else:
#                     st.sidebar.error("Failed to generate embeddings for FAISS rebuild.")
#             else:
#                 st.sidebar.warning("No concepts found in DB after loading; FAISS index not rebuilt.")
            
#             st.session_state.new_text_area = "" 
#             st.experimental_rerun()

#         else:
#             st.sidebar.warning("Gemini did not return any concepts or triples, or there was an error.")
#     elif not new_text_input:
#         st.sidebar.warning("Please paste some text to process.")
#     else:
#         st.sidebar.error("A required component (Gemini, Neo4j, or Embed Model) is not ready for text processing.")

# st.markdown("Ask a question or seek an analogy between concepts (e.g., 'Analogy for DNA in philosophy?')") # Removed, moved to Tab 2

# user_query = st.text_input("Enter your query:") # Removed, moved to Tab 2

# if user_query: # All this logic moved to Tab 2
#     ui_embed_model = st.session_state.get('embed_model')
#     ui_faiss_index = st.session_state.get('faiss_index')
#     ui_concept_list_for_faiss = st.session_state.get('concept_list_for_faiss', [])
#     ui_graph_db_connection = st.session_state.get('graph_db_connection')

#     st.write(f"Processing query: {user_query}")
    
#     is_analogy_query = False
#     extracted_concept = None
#     target_domain_query = None

#     pattern_str = (
#         r"analog(?:y|ous)?\s*(?:for|of|to|between)?\s*" + 
#         r"\'?\"?([A-Z][A-Za-z\s\-\(\)]*(?:[a-z\s\-\(\)]|[A-Z])|[A-Z]{2,})" + 
#         r"\'?\"?" + 
#         r"(?:\s*in\s*\'?\"?([A-Za-z\s]+)\'?\"?)?" 
#     )
#     analogy_query_match = re.search(pattern_str, user_query, re.IGNORECASE)

#     if analogy_query_match:
#         is_analogy_query = True
#         extracted_concept = analogy_query_match.group(1).strip()
#         if analogy_query_match.group(2):
#             target_domain_query = analogy_query_match.group(2).strip().capitalize()
        
#         st.write(f"Detected concept for analogy: **{extracted_concept}**" + (f" (aiming for domain: {target_domain_query})" if target_domain_query else ""))
    
#     elif "analog" in user_query.lower() or "analogy" in user_query.lower():
#         is_analogy_query = True
#         # Fallback regex, ensure it's robust or provide clear instructions
#         concept_match_fallback = re.search(r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)*|DNA|RNA|Mind-body problem)\b", user_query)
#         if concept_match_fallback:
#             extracted_concept = concept_match_fallback.group(1).strip()
#             st.write(f"Detected concept (fallback): **{extracted_concept}**")
#         else:
#             st.warning("Could not clearly identify a concept for analogy. Please phrase as 'Analogy for [Concept Name] in [Domain]?' or use clearer terms.")

#     if is_analogy_query and extracted_concept:
#         st.write(f"UI Check: embed_model is None: {ui_embed_model is None}")
#         st.write(f"UI Check: faiss_index is None: {ui_faiss_index is None}")
#         if not ui_concept_list_for_faiss:
#              st.write(f"UI Check: concept_list_for_faiss is empty.")
#         else:
#              st.write(f"UI Check: concept_list_for_faiss has {len(ui_concept_list_for_faiss)} items.")

#         if not ui_embed_model or not ui_faiss_index:
#             st.error("Embedding model or FAISS index not ready (from session_state). Cannot perform vector analogy search.")
#         else:
#             analogous_result = find_analogous_concept_vector(extracted_concept) 

#             if analogous_result:
#                 analog_concept = analogous_result['concept']
#                 analog_domain = analogous_result['domain']
#                 similarity = analogous_result.get('similarity', 0.0)

#                 answer = f"Based on vector similarity, an analogous concept to **{extracted_concept}** (Domain: {get_concept_domain(extracted_concept)}) in **{analog_domain}** might be **{analog_concept}** (Similarity: {similarity:.4f})."
#                 st.success(answer)

#                 if ui_graph_db_connection: 
#                     st.subheader(f"Knowledge Graph Context for '{extracted_concept}' and '{analog_concept}'")
#                     try:
#                         subG = nx.Graph()
#                         nodes_to_fetch = [(extracted_concept, get_concept_domain(extracted_concept)), 
#                                           (analog_concept, analog_domain)]
                        
#                         for node_name, node_domain in nodes_to_fetch:
#                             subG.add_node(node_name, domain=node_domain, name=node_name, size=20) 
                            
#                             query_neighbors = """
#                             MATCH (n:Concept {name:$name})-[r]-(m:Concept)
#                             RETURN m.name AS neighbor_name, m.domain AS neighbor_domain, type(r) as rel_type
#                             LIMIT 5 """ 
#                             results = ui_graph_db_connection.run(query_neighbors, name=node_name).data()
#                             for record in results:
#                                 neigh_name = record['neighbor_name']
#                                 neigh_domain = record['neighbor_domain']
#                                 rel_type = record['rel_type']
#                                 if not subG.has_node(neigh_name):
#                                     subG.add_node(neigh_name, domain=neigh_domain, name=neigh_name, size=10)
#                                 if not subG.has_edge(node_name, neigh_name):
#                                     subG.add_edge(node_name, neigh_name, label=rel_type)
                        
#                         if subG.number_of_nodes() > 0:
#                             plt.figure(figsize=(10, 7))
#                             pos = nx.spring_layout(subG, k=0.5, iterations=50)
                            
#                             domain_colors = {"Biology": "skyblue", "Philosophy": "salmon", "General": "lightgray"}
#                             node_colors_list = [domain_colors.get(subG.nodes[n]['domain'], "lightgray") for n in subG.nodes()]
                            
#                             node_sizes_list = [subG.nodes[n]['size'] for n in subG.nodes]

#                             labels = {n: subG.nodes[n]['name'] for n in subG.nodes}
#                             nx.draw(subG, pos, with_labels=True, labels=labels, node_color=node_colors_list, 
#                                     node_size=node_sizes_list, font_size=9, width=0.5, edge_color='gray')
                            
#                             plt.title("Concept Neighborhood Graph")
#                             st.pyplot(plt.gcf())
#                             plt.clf() 
#                         else:
#                             st.info("Could not retrieve sufficient graph context for visualization.")

#                     except Exception as e:
#                         st.error(f"Error generating graph visualization: {e}")
#                         print(f"Error generating graph visualization: {e}")
#                 else:
#                     st.warning("Neo4j not connected. Cannot display graph context.")
#             else:
#                 st.write(f"Sorry, I couldn't find a strong vector-based analogy for **{extracted_concept}** in a different domain.")
#     elif is_analogy_query and not extracted_concept: 
#          st.warning("Could not identify the main concept for your analogy query. Please try rephrasing.")
#     else: # Not an analogy query (or not parsed as one)
#         st.info("This prototype primarily supports analogy queries like 'Analogy for DNA in philosophy?'.")

# --- __main__ block for CLI testing (optional, Streamlit runs the app directly) ---
# if __name__ == "__main__": # REMOVED
    # print("Starting POLYMIND CLI test mode...")
    # pass # Keep the block, but ensure it does nothing when run by streamlit

# --- Section 4 (Streamlit UI) will be added below --- # Removed comment

# --- Streamlit UI Setup --- # Removed comment
# st.set_page_config(layout="wide") # REMOVED - THIS WAS THE CAUSE OF THE ERROR
# st.sidebar.title("POLYMIND Options") # REMOVED - Already set at the top

# Initialize session state variables if they don't exist # Removed comment
# if 'app_initialized' not in st.session_state: # REMOVED - Initialization happens earlier
#    initialize_app_state() # Call the main initialization function # REMOVED - function not defined, logic is in the earlier block

# Create tabs
tab1, tab2, tab3 = st.tabs(["📝 Gemini Knowledge Extraction", "🔎 Analogical Search", "📊 Graph Overview"])

with tab1:
    st.header("📝 Gemini Knowledge Extraction")
    st.markdown("Paste text below and click 'Process with Gemini' to extract concepts and relationships. These will be added to the knowledge graph.")

    # Use a controller variable for the text_area's displayed value
    if 'gemini_input_value_controller' not in st.session_state:
        st.session_state.gemini_input_value_controller = ""

    new_text_area_content = st.text_area(
        "Enter text for Gemini to process:",
        value=st.session_state.gemini_input_value_controller,
        height=250,
        key="gemini_text_input_main" # Unique key for this text_area
    )

    if st.button("Process Text with Gemini", key="gemini_process_button"):
        if new_text_area_content:
            st.info("Processing text with Gemini... this may take a moment.")
            
            gemini_model_instance = st.session_state.get("gemini_model")
            graph_conn = st.session_state.get("graph_db_connection")
            embed_model_instance = st.session_state.get("embed_model")

            if not gemini_model_instance:
                st.error("Gemini model not available. Please check configuration.")
            elif not graph_conn:
                st.error("Neo4j connection not available. Please check connection.")
            elif not embed_model_instance:
                st.error("Embedding model not available. Please check initialization.")
            else:
                # Step 1: Extract meaning atoms (concepts and triples)
                gemini_output_text, new_concepts, new_triples = extract_meaning_atoms_gemini(new_text_area_content, gemini_model_instance)
                
                st.session_state.gemini_raw_output = gemini_output_text # Store actual raw output
                st.session_state.gemini_extracted_concepts = new_concepts
                st.session_state.gemini_extracted_triples = new_triples

                if not new_concepts and not new_triples:
                    st.warning("Gemini did not extract any concepts or triples. The input might have been too short or unclear.")
                else:
                    st.success(f"Gemini extracted {len(new_concepts)} unique concepts and {len(new_triples)} triples.")

                    concept_domain_map = {}
                    # Step 2: Get domains for the new concepts
                    if new_concepts:
                        st.info(f"Getting domains for {len(new_concepts)} new concepts from Gemini...")
                        concept_domain_map = get_domains_for_concepts_gemini(new_concepts, gemini_model_instance)
                        st.session_state.gemini_concept_domain_map = concept_domain_map
                        
                        # Step 3: Load concepts with their new domains
                        loaded_concept_count = load_concepts_to_neo4j(graph_conn, concept_domain_map)
                        if loaded_concept_count > 0:
                            st.success(f"Loaded/merged {loaded_concept_count} concepts with domains into Neo4j.")
                        else:
                            st.warning("No new concepts were loaded/updated in Neo4j.")
                    else:
                        st.info("No new concepts extracted to assign domains for.")
                        st.session_state.gemini_concept_domain_map = {}

                    # Step 4: Load triples (relationships)
                    if new_triples:
                        loaded_triple_count = load_triples_to_neo4j(graph_conn, new_triples)
                        if loaded_triple_count > 0:
                            st.success(f"Loaded/merged {loaded_triple_count} relationships into Neo4j.")
                        else:
                            st.warning("No new relationships were loaded/updated in Neo4j.")
                    else:
                        st.info("No new relationships extracted to load.")

                    # Step 5: Rebuild FAISS index if new data was added
                    if new_concepts: # Rebuild if there were any new concepts processed
                        st.info("Rebuilding FAISS index with potentially new data...")
                        all_concepts_from_db = get_all_concepts_from_neo4j(graph_conn)
                        if all_concepts_from_db:
                            concept_embeddings_for_rebuild = generate_embeddings(all_concepts_from_db, embed_model_instance)
                            if concept_embeddings_for_rebuild is not None:
                                # Get domains from Neo4j for all concepts for consistency
                                concept_domains_for_rebuild = []
                                for concept_name in all_concepts_from_db:
                                    # Try fetching from our fresh map first, then Neo4j, then default
                                    domain = concept_domain_map.get(concept_name)
                                    if not domain: # If not in recently fetched map (e.g. pre-existing concept)
                                        # This is where an updated get_concept_domain that queries Neo4j would be good
                                        # For now, use the old get_concept_domain for simplicity
                                        domain = get_concept_domain(concept_name) 
                                    concept_domains_for_rebuild.append(domain)

                                build_faiss_index(all_concepts_from_db, concept_embeddings_for_rebuild, concept_domains_for_rebuild)
                                if st.session_state.get('faiss_index'):
                                    st.success(f"FAISS index rebuilt with {st.session_state.get('faiss_index').ntotal} total concepts.")
                                else:
                                    st.error("Failed to rebuild FAISS index after processing new data.")
                            else:
                                st.error("Failed to generate embeddings for FAISS rebuild.")
                        else:
                            st.warning("No concepts found in Neo4j to rebuild FAISS index.")
                    
                # Clear the input text area and rerun
                st.session_state.gemini_input_value_controller = "" 
                st.rerun()
        else:
            st.warning("Please enter some text to process.")

    # Display area for Gemini results from session state
    if 'gemini_raw_output' in st.session_state:
        st.subheader("Gemini Processing Results")
        st.text_area("Raw Output from Gemini (Triples Extraction):", st.session_state.gemini_raw_output, height=100, key="gemini_raw_out_display_tab1")
        if st.session_state.get('gemini_extracted_concepts'):
            st.write("Extracted Concepts:", st.session_state.gemini_extracted_concepts)
        if st.session_state.get('gemini_concept_domain_map'):
            st.write("Suggested Domains for Concepts (by Gemini):", st.session_state.gemini_concept_domain_map)
        if st.session_state.get('gemini_extracted_triples'):
            st.write("Extracted Triples:", st.session_state.gemini_extracted_triples)

with tab2:
    st.header("🔎 Analogical Search & Query")
    st.markdown("Ask a question or seek an analogy between concepts (e.g., 'Analogy for DNA in philosophy?')")
    user_query_tab2 = st.text_input("Enter your query for analogy:", key="analogy_query_input")

    if user_query_tab2:
        ui_embed_model = st.session_state.get('embed_model')
        ui_faiss_index = st.session_state.get('faiss_index')
        ui_graph_db_connection = st.session_state.get('graph_db_connection')
        gemini_model_instance = st.session_state.get("gemini_model")

        st.write(f"Processing query: {user_query_tab2}")
        
        is_analogy_query = False
        extracted_concept_tab2 = None
        target_domain_query_tab2 = None

        # Simplified regex, removing optional quotes to isolate linter issue, and allowing optional trailing question mark
        pattern_str_tab2 = r"^(?:Analogy for\s+)?(?P<concept>[A-Za-z0-9\s\-]+?)(?:\s+in\s+(?P<domain>[A-Za-z\s]+))?\??$"

        analogy_query_match_tab2 = re.search(pattern_str_tab2, user_query_tab2, re.IGNORECASE)

        if analogy_query_match_tab2:
            is_analogy_query = True
            extracted_concept_tab2 = analogy_query_match_tab2.group("concept").strip().capitalize() # Normalize here
            if analogy_query_match_tab2.group("domain"):
                target_domain_query_tab2 = analogy_query_match_tab2.group("domain").strip().capitalize()
            st.write(f"Detected concept for analogy: **{extracted_concept_tab2}**" + (f" (aiming for domain: {target_domain_query_tab2})" if target_domain_query_tab2 else ""))
        elif "analog" in user_query_tab2.lower() or "analogy" in user_query_tab2.lower():
            is_analogy_query = True
            concept_match_fallback_tab2 = re.search(r"\\b([A-Z][a-z]+(?:\\s[A-Z][a-z]+)*|DNA|RNA|Mind-body problem)\\b", user_query_tab2)
            if concept_match_fallback_tab2:
                extracted_concept_tab2 = concept_match_fallback_tab2.group(1).strip()
                st.write(f"Detected concept (fallback): **{extracted_concept_tab2}**")
            else:
                st.warning("Could not clearly identify a concept for analogy. Please phrase as 'Analogy for [Concept Name] in [Domain]?' or use clearer terms.")

        if is_analogy_query and extracted_concept_tab2:
            analogy_found = False
            analogy_type = "" 

            if ui_graph_db_connection:
                with st.spinner(f"Searching for concepts structurally similar to '{extracted_concept_tab2}'..."):
                    structural_analogs = find_analogous_concept_structural_graph(
                        ui_graph_db_connection, 
                        extracted_concept_tab2, 
                        target_domain_name=target_domain_query_tab2,
                        k=3 
                    )
                
                if structural_analogs:
                    analogy_found = True
                    analogy_type = "Structurally Derived"
                    top_structural_analog = structural_analogs[0]
                    analog_concept = top_structural_analog['analogous_concept']
                    analog_domain = top_structural_analog['analogous_domain']
                    
                    st.subheader(f"Analogy Found (Structurally Derived)")
                    st.write(f"Source Concept: **{extracted_concept_tab2}** (Domain: {get_concept_domain(extracted_concept_tab2)})" +
                             f"\\nAnalogous Concept: **{analog_concept}** (Domain: {analog_domain})")
                    st.write(f"Shared Pattern Types: {top_structural_analog['num_shared_neighbors']}")

                    with st.expander("Show All Structural Analogs & Details"):
                        for i, analog_info_detail in enumerate(structural_analogs):
                            st.markdown(f"**{i+1}. {analog_info_detail['analogous_concept']}** (Domain: {analog_info_detail['analogous_domain']}) - Shared Neighbors: {analog_info_detail['num_shared_neighbors']}")
                            for shared_node_detail in analog_info_detail['shared_neighbor_details']:
                                shared_node = shared_node_detail.get('shared_node', 'Unknown Shared Node')
                                path_src_list = shared_node_detail.get('path_from_source', [])
                                paths_cand_list = shared_node_detail.get('paths_from_candidate', []) # This is a list of paths

                                path_src_str = " -> ".join(path_src_list) if path_src_list else "N/A"
                                
                                # Show one example path from candidate
                                path_cand_str = "N/A"
                                if paths_cand_list and isinstance(paths_cand_list, list) and paths_cand_list[0].get('path_cand'):
                                     path_cand_str = " -> ".join(paths_cand_list[0]['path_cand'])

                                st.markdown(f"  - Shared Node: '{shared_node}'")
                                st.write(f"     Path from '{extracted_concept_tab2}': {path_src_str}")
                                st.write(f"     Example path from '{analog_concept}': {path_cand_str}")
                            st.markdown("---")
                    
                    if gemini_model_instance:
                        with st.spinner("Gemini is crafting an explanation for the structural analogy..."):
                            explanation = generate_analogy_explanation_gemini(
                                extracted_concept_tab2, 
                                get_concept_domain(extracted_concept_tab2), 
                                analog_concept, 
                                analog_domain, 
                                gemini_model_instance,
                                shared_patterns_info=top_structural_analog 
                            )
                        st.markdown("### Gemini's Explanation (Structural)")
                        st.markdown(explanation)

            if not analogy_found:
                if not ui_embed_model or not ui_faiss_index:
                    st.error("Embedding model or FAISS index not ready. Cannot perform vector analogy search.")
                else:
                    vector_analog_result = find_analogous_concept_vector(extracted_concept_tab2) 
                    if vector_analog_result:
                        analogy_found = True
                        analogy_type = "Semantically Similar (Vector-based)"
                        analog_concept = vector_analog_result['concept']
                        analog_domain = vector_analog_result['domain']
                        similarity = vector_analog_result.get('similarity', 0.0)
                        st.subheader(f"Analogy Found (Semantically Similar (Vector-based))")
                        st.write(f"Source Concept: **{extracted_concept_tab2}** (Domain: {get_concept_domain(extracted_concept_tab2)})" +
                                 f"\\nAnalogous Concept: **{analog_concept}** (Domain: {analog_domain}, Similarity: {similarity:.4f})")
                        if gemini_model_instance:
                            with st.spinner("Gemini is crafting an explanation for the vector-based analogy..."):
                                explanation = generate_analogy_explanation_gemini(
                                    extracted_concept_tab2, 
                                    get_concept_domain(extracted_concept_tab2), 
                                    analog_concept, 
                                    analog_domain, 
                                    gemini_model_instance
                                )
                            st.markdown("### Gemini's Explanation (Vector-based)")
                            st.markdown(explanation)
            
            if analogy_found and ui_graph_db_connection:
                st.subheader(f"Knowledge Graph Context for '{extracted_concept_tab2}' and '{analog_concept}'")
                try:
                    subG = nx.Graph()
                    nodes_to_fetch = [
                        (extracted_concept_tab2, get_concept_domain(extracted_concept_tab2)), 
                        (analog_concept, analog_domain) # analog_concept and analog_domain are set by either structural or vector search
                    ]
                    
                    for node_name, node_domain_val in nodes_to_fetch: # Renamed node_domain to node_domain_val
                        subG.add_node(node_name, domain=node_domain_val, name=node_name, size=20)
                        query_neighbors = """
                        MATCH (n:Concept {name:$name})-[r]-(m:Concept)
                        RETURN m.name AS neighbor_name, m.domain AS neighbor_domain, type(r) as rel_type
                        LIMIT 7 """ 
                        results_struct = ui_graph_db_connection.run(query_neighbors, name=node_name).data()
                        for record in results_struct:
                            neigh_name = record['neighbor_name']
                            neigh_domain_val = record['neighbor_domain'] # Renamed
                            rel_type = record['rel_type']
                            if not subG.has_node(neigh_name):
                                subG.add_node(neigh_name, domain=neigh_domain_val, name=neigh_name, size=10)
                            if not subG.has_edge(node_name, neigh_name):
                                subG.add_edge(node_name, neigh_name, label=rel_type)
                    
                    if subG.number_of_nodes() > 0:
                        plt.figure(figsize=(10, 7))
                        pos_struct = nx.spring_layout(subG, k=0.5, iterations=50)
                        domain_colors = {"Biology": "skyblue", "Philosophy": "salmon", "General": "lightgray"}
                        node_colors_list_struct = [domain_colors.get(subG.nodes[n]['domain'], "lightgray") for n in subG.nodes()]
                        node_sizes_list_struct = [subG.nodes[n]['size'] for n in subG.nodes()]
                        labels_struct = {n: subG.nodes[n]['name'] for n in subG.nodes()}

                        nx.draw(subG, pos_struct, with_labels=True, labels=labels_struct, 
                                node_color=node_colors_list_struct, node_size=node_sizes_list_struct, 
                                font_size=9, width=0.5, edge_color='gray')
                        # Use the correctly parsed/found concept and analog for the title
                        title_concept_name = extracted_concept_tab2 if extracted_concept_tab2 else "Source"
                        title_analog_name = analog_concept if analog_concept else "Analog"
                        plt.title(f"Analogy Graph: {title_concept_name} & {title_analog_name}")
                        st.pyplot(plt.gcf())
                        plt.clf()
                    else:
                        st.info("Could not retrieve sufficient graph context for visualization.")
                except Exception as e:
                    st.error(f"Error generating graph visualization: {e}")
                    print(f"Error generating graph visualization: {e}")
            elif analogy_found and not ui_graph_db_connection:
                 st.warning("Neo4j not connected. Cannot display graph context for the found analogy.")
            elif not analogy_found:
                st.info(f"No analogy (neither structural nor vector-based) could be established for '{extracted_concept_tab2}'.")

        elif is_analogy_query and not extracted_concept_tab2: 
             st.warning("Could not identify the main concept for your analogy query. Please try rephrasing.")
        else: 
            st.info("This tab primarily supports analogy queries like 'Analogy for DNA in philosophy?'.")

        # --- NEW Conceptual Blending Section in Tab 2 (Ensuring it's outside any function) ---
        st.markdown("---") 
        st.subheader("🧪 Conceptual Blending")
        st.markdown("Enter two concepts below to attempt a conceptual blend using Gemini.")

        col1_blend, col2_blend = st.columns(2)
        with col1_blend:
            concept_a_blend_input = st.text_input("Concept A for Blending:", key="concept_a_blend_input")
        with col2_blend:
            concept_b_blend_input = st.text_input("Concept B for Blending:", key="concept_b_blend_input")

        if st.button("Blend Concepts with Gemini", key="blend_concepts_button"):
            if concept_a_blend_input and concept_b_blend_input:
                concept_a_blend = concept_a_blend_input.strip().capitalize() # Normalize here
                concept_b_blend = concept_b_blend_input.strip().capitalize() # Normalize here
                st.info(f"Attempting to blend '{concept_a_blend}' and '{concept_b_blend}'...")
                graph_conn_blend = st.session_state.get("graph_db_connection")
                gemini_model_blend = st.session_state.get("gemini_model")

                if not graph_conn_blend:
                    st.error("Neo4j connection not available for blending.")
                elif not gemini_model_blend:
                    st.error("Gemini model not available for blending.")
                else:
                    st.write(f"Retrieving features for Concept A: {concept_a_blend}...")
                    features_a = get_concept_features_neo4j(graph_conn_blend, concept_a_blend)
                    if not features_a:
                        st.warning(f"Could not retrieve features for '{concept_a_blend}' or it has no connections.")
                    else:
                        st.write(f"Found {len(features_a)} features for {concept_a_blend}:")
                        with st.expander(f"Show features for {concept_a_blend}"):
                            st.json(features_a)
                    
                    st.write(f"Retrieving features for Concept B: {concept_b_blend}...")
                    features_b = get_concept_features_neo4j(graph_conn_blend, concept_b_blend)
                    if not features_b:
                        st.warning(f"Could not retrieve features for '{concept_b_blend}' or it has no connections.")
                    else:
                        st.write(f"Found {len(features_b)} features for {concept_b_blend}:")
                        with st.expander(f"Show features for {concept_b_blend}"):
                            st.json(features_b)

                    blended_result = blend_concepts_gemini(concept_a_blend, features_a, 
                                                           concept_b_blend, features_b, 
                                                           gemini_model_blend)
                    
                    if blended_result:
                        st.session_state.current_blend_result = blended_result 
                        st.subheader(f"Blended Concept: {blended_result.get('name', 'Unnamed Blend')}")
                        st.markdown("**Description:**")
                        st.write(blended_result.get('description', 'No description provided.'))
                        
                        if blended_result.get('features'):
                            st.markdown("**Key Features of Blend:**")
                            for feature_item in blended_result.get('features', []):
                                st.markdown(f"- {feature_item}")
                        
                        if blended_result.get('emergent_properties'):
                            st.markdown("**Emergent Properties:**")
                            st.write(blended_result.get('emergent_properties', 'None specified.'))
                        
                        if st.button("Add Blended Concept to Knowledge Graph", key="add_blend_to_graph_button"):
                            st.info("Functionality to add blend to graph to be implemented.")
                    else:
                        st.error("Failed to generate a blend from Gemini.")
                        st.session_state.current_blend_result = None
            else:
                st.warning("Please enter both Concept A and Concept B for blending.")
        # --- END Conceptual Blending Section ---

    if user_query_tab2: # Moved the analogy search processing below the blending UI setup
        ui_embed_model = st.session_state.get('embed_model')
        ui_faiss_index = st.session_state.get('faiss_index')
        ui_graph_db_connection = st.session_state.get('graph_db_connection')
        gemini_model_instance = st.session_state.get("gemini_model")

        st.write(f"Processing query: {user_query_tab2}")
        
        is_analogy_query = False
        extracted_concept_tab2 = None
        target_domain_query_tab2 = None

        # Simplified regex, removing optional quotes to isolate linter issue, and allowing optional trailing question mark
        pattern_str_tab2 = r"^(?:Analogy for\s+)?(?P<concept>[A-Za-z0-9\s\-]+?)(?:\s+in\s+(?P<domain>[A-Za-z\s]+))?\??$"

        analogy_query_match_tab2 = re.search(pattern_str_tab2, user_query_tab2, re.IGNORECASE)

        if analogy_query_match_tab2:
            is_analogy_query = True
            extracted_concept_tab2 = analogy_query_match_tab2.group("concept").strip().capitalize() # Normalize here
            if analogy_query_match_tab2.group("domain"):
                target_domain_query_tab2 = analogy_query_match_tab2.group("domain").strip().capitalize()
            st.write(f"Detected concept for analogy: **{extracted_concept_tab2}**" + (f" (aiming for domain: {target_domain_query_tab2})" if target_domain_query_tab2 else ""))
        elif "analog" in user_query_tab2.lower() or "analogy" in user_query_tab2.lower():
            is_analogy_query = True
            concept_match_fallback_tab2 = re.search(r"\\b([A-Z][a-z]+(?:\\s[A-Z][a-z]+)*|DNA|RNA|Mind-body problem)\\b", user_query_tab2)
            if concept_match_fallback_tab2:
                extracted_concept_tab2 = concept_match_fallback_tab2.group(1).strip()
                st.write(f"Detected concept (fallback): **{extracted_concept_tab2}**")
            else:
                st.warning("Could not clearly identify a concept for analogy. Please phrase as 'Analogy for [Concept Name] in [Domain]?' or use clearer terms.")

        if is_analogy_query and extracted_concept_tab2:
            analogy_found = False
            analogy_type = "" 

            if ui_graph_db_connection:
                with st.spinner(f"Searching for concepts structurally similar to '{extracted_concept_tab2}'..."):
                    structural_analogs = find_analogous_concept_structural_graph(
                        ui_graph_db_connection, 
                        extracted_concept_tab2, 
                        target_domain_name=target_domain_query_tab2,
                        k=3 
                    )
                
                if structural_analogs:
                    analogy_found = True
                    analogy_type = "Structurally Derived"
                    top_structural_analog = structural_analogs[0]
                    analog_concept = top_structural_analog['analogous_concept']
                    analog_domain = top_structural_analog['analogous_domain']
                    
                    st.subheader(f"Analogy Found (Structurally Derived)")
                    st.write(f"Source Concept: **{extracted_concept_tab2}** (Domain: {get_concept_domain(extracted_concept_tab2)})" +
                             f"\\nAnalogous Concept: **{analog_concept}** (Domain: {analog_domain})")
                    st.write(f"Shared Pattern Types: {top_structural_analog['num_shared_neighbors']}")

                    with st.expander("Show All Structural Analogs & Details"):
                        for i, analog_info_detail in enumerate(structural_analogs):
                            st.markdown(f"**{i+1}. {analog_info_detail['analogous_concept']}** (Domain: {analog_info_detail['analogous_domain']}) - Shared Neighbors: {analog_info_detail['num_shared_neighbors']}")
                            for shared_node_detail in analog_info_detail['shared_neighbor_details']:
                                shared_node = shared_node_detail.get('shared_node', 'Unknown Shared Node')
                                path_src_list = shared_node_detail.get('path_from_source', [])
                                paths_cand_list = shared_node_detail.get('paths_from_candidate', []) # This is a list of paths

                                path_src_str = " -> ".join(path_src_list) if path_src_list else "N/A"
                                
                                # Show one example path from candidate
                                path_cand_str = "N/A"
                                if paths_cand_list and isinstance(paths_cand_list, list) and paths_cand_list[0].get('path_cand'):
                                     path_cand_str = " -> ".join(paths_cand_list[0]['path_cand'])

                                st.markdown(f"  - Shared Node: '{shared_node}'")
                                st.write(f"     Path from '{extracted_concept_tab2}': {path_src_str}")
                                st.write(f"     Example path from '{analog_concept}': {path_cand_str}")
                            st.markdown("---")
                    
                    if gemini_model_instance:
                        with st.spinner("Gemini is crafting an explanation for the structural analogy..."):
                            explanation = generate_analogy_explanation_gemini(
                                extracted_concept_tab2, 
                                get_concept_domain(extracted_concept_tab2), 
                                analog_concept, 
                                analog_domain, 
                                gemini_model_instance,
                                shared_patterns_info=top_structural_analog 
                            )
                        st.markdown("### Gemini's Explanation (Structural)")
                        st.markdown(explanation)

            if not analogy_found:
                if not ui_embed_model or not ui_faiss_index:
                    st.error("Embedding model or FAISS index not ready. Cannot perform vector analogy search.")
                else:
                    vector_analog_result = find_analogous_concept_vector(extracted_concept_tab2) 
                    if vector_analog_result:
                        analogy_found = True
                        analogy_type = "Semantically Similar (Vector-based)"
                        analog_concept = vector_analog_result['concept']
                        analog_domain = vector_analog_result['domain']
                        similarity = vector_analog_result.get('similarity', 0.0)
                        st.subheader(f"Analogy Found (Semantically Similar (Vector-based))")
                        st.write(f"Source Concept: **{extracted_concept_tab2}** (Domain: {get_concept_domain(extracted_concept_tab2)})" +
                                 f"\\nAnalogous Concept: **{analog_concept}** (Domain: {analog_domain}, Similarity: {similarity:.4f})")
                        if gemini_model_instance:
                            with st.spinner("Gemini is crafting an explanation for the vector-based analogy..."):
                                explanation = generate_analogy_explanation_gemini(
                                    extracted_concept_tab2, 
                                    get_concept_domain(extracted_concept_tab2), 
                                    analog_concept, 
                                    analog_domain, 
                                    gemini_model_instance
                                )
                            st.markdown("### Gemini's Explanation (Vector-based)")
                            st.markdown(explanation)
            
            if analogy_found and ui_graph_db_connection:
                st.subheader(f"Knowledge Graph Context for '{extracted_concept_tab2}' and '{analog_concept}'")
                try:
                    subG = nx.Graph()
                    nodes_to_fetch = [
                        (extracted_concept_tab2, get_concept_domain(extracted_concept_tab2)), 
                        (analog_concept, analog_domain) # analog_concept and analog_domain are set by either structural or vector search
                    ]
                    
                    for node_name, node_domain_val in nodes_to_fetch: # Renamed node_domain to node_domain_val
                        subG.add_node(node_name, domain=node_domain_val, name=node_name, size=20)
                        query_neighbors = """
                        MATCH (n:Concept {name:$name})-[r]-(m:Concept)
                        RETURN m.name AS neighbor_name, m.domain AS neighbor_domain, type(r) as rel_type
                        LIMIT 7 """ 
                        results_struct = ui_graph_db_connection.run(query_neighbors, name=node_name).data()
                        for record in results_struct:
                            neigh_name = record['neighbor_name']
                            neigh_domain_val = record['neighbor_domain'] # Renamed
                            rel_type = record['rel_type']
                            if not subG.has_node(neigh_name):
                                subG.add_node(neigh_name, domain=neigh_domain_val, name=neigh_name, size=10)
                            if not subG.has_edge(node_name, neigh_name):
                                subG.add_edge(node_name, neigh_name, label=rel_type)
                    
                    if subG.number_of_nodes() > 0:
                        plt.figure(figsize=(10, 7))
                        pos_struct = nx.spring_layout(subG, k=0.5, iterations=50)
                        domain_colors = {"Biology": "skyblue", "Philosophy": "salmon", "General": "lightgray"}
                        node_colors_list_struct = [domain_colors.get(subG.nodes[n]['domain'], "lightgray") for n in subG.nodes()]
                        node_sizes_list_struct = [subG.nodes[n]['size'] for n in subG.nodes()]
                        labels_struct = {n: subG.nodes[n]['name'] for n in subG.nodes()}

                        nx.draw(subG, pos_struct, with_labels=True, labels=labels_struct, 
                                node_color=node_colors_list_struct, node_size=node_sizes_list_struct, 
                                font_size=9, width=0.5, edge_color='gray')
                        # Use the correctly parsed/found concept and analog for the title
                        title_concept_name = extracted_concept_tab2 if extracted_concept_tab2 else "Source"
                        title_analog_name = analog_concept if analog_concept else "Analog"
                        plt.title(f"Analogy Graph: {title_concept_name} & {title_analog_name}")
                        st.pyplot(plt.gcf())
                        plt.clf()
                    else:
                        st.info("Could not retrieve sufficient graph context for visualization.")
                except Exception as e:
                    st.error(f"Error generating graph visualization: {e}")
                    print(f"Error generating graph visualization: {e}")
            elif analogy_found and not ui_graph_db_connection:
                 st.warning("Neo4j not connected. Cannot display graph context for the found analogy.")
            elif not analogy_found:
                st.info(f"No analogy (neither structural nor vector-based) could be established for '{extracted_concept_tab2}'.")

        elif is_analogy_query and not extracted_concept_tab2: 
             st.warning("Could not identify the main concept for your analogy query. Please try rephrasing.")
        else: # Not an analogy query (or not parsed as one)
            st.info("This tab primarily supports analogy queries like 'Analogy for DNA in philosophy?'.")

        # --- REMOVE the separate button and logic for Structural Analogy Search ---
        # The block starting with:
        # if is_analogy_query and extracted_concept_tab2 and ui_graph_db_connection: 
        #    st.markdown("---")
        #    st.subheader(f"Advanced Structural Analogy for: {extracted_concept_tab2}")
        #    if st.button(f"Find Structural Analogy (Graph - Max 2 hops)", key="structural_analogy_button"):
        # ... and all its nested content down to its corresponding `else` or end of block,
        # should be removed as its functionality is now integrated above.
        # The edit tool will handle removing this section based on the context if this comment is clear enough.

with tab3:
    st.header("📊 Knowledge Graph Overview")
    # ... existing code for tab3 ...
    st.write("This tab will display an overview of the knowledge graph. (Functionality to be implemented)")
    if st.button("Show Full Graph (Sample - Top 25 nodes)", key="show_full_graph_button"):
        graph_conn_tab3 = st.session_state.get('graph_db_connection')
        if graph_conn_tab3:
            try:
                query = """
                MATCH (n)-[r]->(m)
                RETURN n.name AS source, n.domain AS source_domain, 
                       m.name AS target, m.domain AS target_domain,
                       type(r) AS relationship
                LIMIT 25 
                """ # Query a limited number of nodes/relationships for overview
                results = graph_conn_tab3.run(query).data()
                
                if results:
                    overview_G = nx.DiGraph() # Use DiGraph if relationships are directed
                    domain_colors_overview = {"Biology": "skyblue", "Philosophy": "salmon", "General": "lightgray", None: "lightgray"}
                    
                    for record in results:
                        source_name = record['source']
                        target_name = record['target']
                        
                        if source_name not in overview_G:
                            overview_G.add_node(source_name, name=source_name, domain=record['source_domain'])
                        if target_name not in overview_G:
                             overview_G.add_node(target_name, name=target_name, domain=record['target_domain'])
                        overview_G.add_edge(source_name, target_name, label=record['relationship'])

                    if overview_G.number_of_nodes() > 0:
                        plt.figure(figsize=(12, 10))
                        pos_overview = nx.spring_layout(overview_G, k=0.6, iterations=50)
                        
                        node_colors_list_overview = [domain_colors_overview.get(overview_G.nodes[n]['domain'], "lightgray") for n in overview_G.nodes()]
                        labels_overview = {n: overview_G.nodes[n]['name'] for n in overview_G.nodes()}

                        nx.draw(overview_G, pos_overview, with_labels=True, labels=labels_overview, 
                                node_color=node_colors_list_overview, node_size=1500, font_size=8, 
                                width=0.5, edge_color='gray', arrows=True)
                        
                        edge_labels_overview = nx.get_edge_attributes(overview_G, 'label')
                        nx.draw_networkx_edge_labels(overview_G, pos_overview, edge_labels=edge_labels_overview, font_size=7)
                        
                        plt.title("Knowledge Graph Overview (Sample)")
                        st.pyplot(plt.gcf())
                        plt.clf()
                    else:
                        st.info("No data to display for graph overview.")
                else:
                    st.info("No relationships found in the graph to display an overview.")
            except Exception as e:
                st.error(f"Error generating graph overview: {e}")
                print(f"Error generating graph overview (Tab3): {e}")
        else:
            st.warning("Neo4j connection not available for graph overview.") 

# --- NEW FUNCTION: Get Domains for Concepts using Gemini ---
def get_domains_for_concepts_gemini(concepts_list, gemini_model_to_use):
    """
    Suggests a primary domain for each concept in a list using the Gemini API.
    Returns a dictionary mapping concept_name to domain_name.
    """
    if not gemini_model_to_use:
        st.error("Gemini model not provided or not initialized. Cannot suggest domains.")
        print("Gemini model not provided for domain suggestion.")
        return {concept: "General" for concept in concepts_list} # Fallback

    if not concepts_list:
        return {}

    # Prepare the list of concepts for the prompt
    concept_bullet_list = "\n".join([f"- {concept}" for concept in concepts_list])

    prompt = f"""For each concept in the following list, suggest a single, primary academic or general domain it belongs to.
Output each concept and its domain on a new line, strictly in the format:
Concept: Domain

Example:
DNA: Biology
Metaphysics: Philosophy
Algorithm: Computer Science
Photosynthesis: Biology
Market Economy: Economics

Concepts to categorize:
{concept_bullet_list}

Categorized Concepts:
"""
    try:
        print(f"Sending {len(concepts_list)} concepts to Gemini for domain suggestion.")
        response = gemini_model_to_use.generate_content(prompt)
        
        response_text = response.text
        # print(f"Gemini domain suggestion response: {response_text}")

        domain_map = {}
        # Parse the "Concept: Domain" format
        for line in response_text.split('\n'):
            line = line.strip()
            if ": " in line: # Check for the separator
                parts = line.split(": ", 1) # Split only on the first occurrence
                if len(parts) == 2:
                    concept_name = parts[0].strip()
                    domain_name = parts[1].strip()
                    # Ensure the concept was in the original list to avoid hallucinated concepts
                    if concept_name in concepts_list:
                        domain_map[concept_name] = domain_name
                    else:
                        print(f"Gemini suggested domain for concept '{concept_name}' not in original list. Skipping.")
                else:
                    print(f"Skipping malformed domain line (not 2 parts after split): {line}")
            elif line: # Non-empty line that doesn't match format
                print(f"Skipping non-domain line from Gemini output: {line}")
        
        # For any concepts Gemini didn't provide a domain for, assign "General"
        for concept in concepts_list:
            if concept not in domain_map:
                print(f"Gemini did not provide domain for '{concept}'. Defaulting to 'General'.")
                domain_map[concept] = "General"
        
        print(f"Gemini suggested domains: {domain_map}")
        return domain_map

    except Exception as e:
        print(f"Error calling Gemini API or parsing domain response: {e}")
        st.error(f"Error during Gemini domain suggestion: {e}")
        # Fallback: assign "General" to all concepts
        return {concept: "General" for concept in concepts_list}

# --- Initialize Application State (Connections, Models, FAISS) ---
def initialize_app_state():
    # ... existing code ...
    # ...
    # def get_domains_for_concepts_gemini(concepts_list, gemini_model_to_use): # Definition removed from here
    # ...
    # --- Initialize Application State (Connections, Models, FAISS) ---
    # ...
    # --- Streamlit UI ---

    # --- Tab Definitions ---
    tab1, tab2, tab3 = st.tabs(["📝 Gemini Knowledge Extraction", "🔎 Analogical Search & Query", "�� Graph Overview"])
    # ... (rest of the Streamlit UI code) 