import logging
from typing import List, Dict, Any, Optional, Tuple

# Attempt to import project-specific modules
try:
    from ... import config
    from ..knowledge_graph.graph_handler import GraphHandler
    from ..vector_store.faiss_handler import FaissHandler # Contains embedding generation and FAISS ops
    # We'll also need a Gemini model for explanations
    import google.generativeai as genai

except ImportError:
    # Fallback for running script directly or if polymind_v2 is in PYTHONPATH
    import sys
    import os
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    polymind_core_dir = os.path.dirname(current_script_dir)
    project_root = os.path.dirname(polymind_core_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    import config
    from polymind_core.knowledge_graph.graph_handler import GraphHandler
    from polymind_core.vector_store.faiss_handler import FaissHandler
    import google.generativeai as genai


# Configure logging
log_level = getattr(config, 'LOG_LEVEL', 'INFO')
log_format = getattr(config, 'LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.basicConfig(level=log_level, format=log_format)
logger = logging.getLogger(__name__)

_gemini_model_analogy = None

def _get_gemini_model_for_analogy():
    """Initializes or retrieves an existing Gemini model instance for analogy explanations."""
    global _gemini_model_analogy
    if _gemini_model_analogy is None:
        # Use the standard placeholder string for the API key check
        if config.GEMINI_API_KEY and config.GEMINI_API_KEY.strip() != "YOUR_GEMINI_API_KEY_HERE" and config.GEMINI_API_KEY.strip() != "":
            try:
                genai.configure(api_key=config.GEMINI_API_KEY)
                _gemini_model_analogy = genai.GenerativeModel(config.DEFAULT_GEMINI_MODEL)
                logger.info(f"AnalogyFinder: Gemini model '{config.DEFAULT_GEMINI_MODEL}' initialized successfully.")
            except Exception as e:
                logger.error(f"AnalogyFinder: Error configuring Gemini API or initializing model: {e}")
                _gemini_model_analogy = None
        else:
            logger.warning("AnalogyFinder: Gemini API Key not set or is a placeholder. Analogy explanations will be unavailable.")
            _gemini_model_analogy = None
    return _gemini_model_analogy


def find_analogous_concepts_vector(
    query_concept_name: str,
    faiss_handler: FaissHandler,
    graph_handler: GraphHandler, # To get domain/properties of results
    k: int = 5,
    excluded_concept_names: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Finds concepts analogous to the query concept based on vector similarity.
    Retrieves concept details (name, domain) for the similar concepts.

    Args:
        query_concept_name (str): The name of the concept to find analogues for.
        faiss_handler (FaissHandler): Instance of FaissHandler.
        graph_handler (GraphHandler): Instance of GraphHandler to fetch concept details.
        k (int): Number of analogous concepts to return.
        excluded_concept_names (Optional[List[str]]): List of concept names to exclude from results.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, where each dict contains
                              'name', 'domain', 'score' (similarity score),
                              and potentially other properties for an analogous concept.
                              Returns empty list on failure or if no analogues found.
    """
    logger.info(f"Finding vector analogies for '{query_concept_name}', k={k}")
    if not faiss_handler._faiss_index or not faiss_handler._concept_list:
        logger.error("FAISS index or concept list not loaded in FaissHandler. Cannot perform vector search.")
        return []

    try:
        # 1. Get the embedding for the query concept name itself.
        #    The FaissHandler's create_concept_embeddings method is suitable here (renamed from create_embeddings if that was a typo)
        #    Assuming create_concept_embeddings is the correct method name in FaissHandler
        query_embedding = faiss_handler.create_concept_embeddings([query_concept_name])
        if query_embedding is None or query_embedding.shape[0] == 0:
            logger.error(f"Could not generate embedding for query concept '{query_concept_name}'.")
            return []

        # 2. Search in FAISS
        #    Assuming search_similar is the correct method in FaissHandler for searching
        #    The FaissHandler has find_similar_concepts which returns (name, score) list.
        #    We need distances and indices for the current logic structure. Let's assume a lower-level search method or adapt.
        #    For now, let's check if FaissHandler has a method like search_raw_faiss or similar
        #    Looking at faiss_handler.py, it has self._faiss_index.search()
        #    And the concept list is self._concept_list

        # Adjusting to use the available FaissHandler methods correctly:
        # search_similar in FaissHandler might be high level. The logic here needs raw distances/indices.
        # The current find_analogous_concepts_vector tries to replicate parts of FaissHandler.find_similar_concepts
        # but needs direct access to faiss_index.search and concept_list for its current filtering logic.
        
        # Correcting direct access to _faiss_index.search and _concept_list
        if faiss_handler._faiss_index is None:
            logger.error("FaissHandler._faiss_index is None. Cannot search.")
            return []

        # Fetch more to filter, then limit to k
        num_to_fetch = k + (len(excluded_concept_names) if excluded_concept_names else 0) + 1 
        actual_k_for_search = min(num_to_fetch, faiss_handler._faiss_index.ntotal)
        
        if actual_k_for_search <= 0:
            logger.warning(f"Adjusted k for search is {actual_k_for_search}. Cannot perform FAISS search for '{query_concept_name}'. Index might be empty or k too small.")
            return []

        distances, indices = faiss_handler._faiss_index.search(query_embedding, actual_k_for_search)

        if indices is None or distances is None or indices.shape[1] == 0:
            logger.warning(f"No vector analogies found from FAISS search for '{query_concept_name}'.")
            return []

        analogous_concepts = []
        current_excluded_names = list(excluded_concept_names) if excluded_concept_names else []
        
        # Add the query concept itself to excluded names to avoid self-match as top result
        if query_concept_name not in current_excluded_names:
            current_excluded_names.append(query_concept_name)

        for i in range(len(indices[0])):
            idx = indices[0][i]
            dist = distances[0][i]
            
            if idx < 0 or idx >= len(faiss_handler._concept_list):
                logger.warning(f"Invalid index {idx} from FAISS search. Skipping.")
                continue

            retrieved_concept_name = faiss_handler._concept_list[idx]

            if retrieved_concept_name in current_excluded_names:
                continue

            # 3. Get details for the found concept (domain, etc.)
            #    We need a method in GraphHandler like get_concept_details(name)
            #    For now, let's assume get_concept_features can give us enough, or we add one.
            #    A simpler get_concept_properties might be better.
            
            # Placeholder: Assuming graph_handler has a method to get basic properties
            # concept_details = graph_handler.get_concept_properties(retrieved_concept_name) # Ideal
            concept_details_full = graph_handler.get_concept_features(retrieved_concept_name) # max_hops=0 is default in get_concept_features if not specified or if it was removed as a param.
                                                                                             # Assuming get_concept_features can be called with just concept_name.
            
            if concept_details_full and "properties" in concept_details_full:
                concept_props = concept_details_full["properties"]
                analogous_concepts.append({
                    "name": retrieved_concept_name,
                    "domain": concept_props.get("domain", "Unknown"),
                    "description": concept_props.get("description", ""), # If available
                    "score": 1 - dist # Convert distance to similarity score (0-1, higher is better)
                })
            else:
                 analogous_concepts.append({
                    "name": retrieved_concept_name,
                    "domain": "Unknown",
                    "description": "",
                    "score": 1 - dist 
                })
            
            if len(analogous_concepts) >= k:
                break
        
        logger.info(f"Found {len(analogous_concepts)} vector analogies for '{query_concept_name}'.")
        return analogous_concepts

    except Exception as e:
        logger.error(f"Error during vector analogy search for '{query_concept_name}': {e}", exc_info=True)
        return []


def find_analogous_concepts_structural_graph(
    query_concept_name: str,
    graph_handler: GraphHandler,
    k: int = 5,
    max_hops: int = 1, # Currently, logic focuses on 1-hop direct neighbors
    excluded_concept_names: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Finds concepts that are structurally analogous to the query concept in the graph.
    This implementation uses a Cypher query to find concepts that share common patterns
    of relationships (type and neighbor's primary domain) with the query concept's 1-hop neighborhood.

    Args:
        query_concept_name (str): The name of the concept to find analogues for.
        graph_handler (GraphHandler): Instance of GraphHandler.
        k (int): Number of analogous concepts to return.
        max_hops (int): Defines the neighborhood to compare (currently fixed at 1-hop).
        excluded_concept_names (Optional[List[str]]): List of concept names to exclude.

    Returns:
        List[Dict[str, Any]]: List of analogous concepts with their details (name, domain, score).
                              The score represents the count of shared structural patterns.
    """
    if max_hops != 1:
        logger.warning(f"Structural analogy currently only supports max_hops=1. Using max_hops=1.")
        # max_hops is kept for future extension but current query is 1-hop specific.

    logger.info(f"Finding structural graph analogies for '{query_concept_name}', k={k}, hops=1")

    if excluded_concept_names is None:
        excluded_concept_names = []
    # Ensure the query concept itself is excluded from results
    if query_concept_name not in excluded_concept_names:
        excluded_concept_names.append(query_concept_name)

    # This query aims to find concepts that share similar types of relationships
    # to similar types of neighboring concepts (based on their 'domain' property).
    # It's a heuristic for structural similarity.
    #
    # Explanation of the Cypher query parts:
    # 1. Get outgoing structural patterns of the query concept (qc):
    #    MATCH (qc:Concept {name: $query_name})-[r_out:ALLOWED_RELATIONSHIP_TYPES]->(n_out:Concept)
    #    Collect distinct pairs of [type(r_out), n_out.domain] as qc_outgoing_patterns.
    #    (Assuming ALLOWED_RELATIONSHIP_TYPES is a label on relationships, or adjust to use type(r))
    #    For simplicity, we will use type(r) directly if ALLOWED_RELATIONSHIP_TYPES label isn't standard.
    #
    # 2. Get incoming structural patterns of the query concept (qc):
    #    MATCH (qc:Concept {name: $query_name})<-[r_in:ALLOWED_RELATIONSHIP_TYPES]-(n_in:Concept)
    #    Collect distinct pairs of [type(r_in), n_in.domain] as qc_incoming_patterns.
    #
    # 3. Find other concepts (oc) and their patterns:
    #    Iterate through other concepts (oc).
    #    For each oc, get its outgoing_patterns and incoming_patterns similarly.
    #
    # 4. Calculate similarity score:
    #    Score = count of common elements in qc_outgoing_patterns and oc_outgoing_patterns
    #          + count of common elements in qc_incoming_patterns and oc_incoming_patterns.
    #
    # 5. Order by score and limit.

    # Simpler approach for this iteration: Focus on common neighbors and relationship types.
    # This query counts how many neighbors a candidate concept shares with the query concept,
    # considering the relationship type and direction.
    
    # Step 1: Get the 1-hop neighborhood signature of the query concept
    # (RelationshipType, Direction, NeighborDomain)
    # Direction: 'OUTGOING' or 'INCOMING'
    query_signature_cypher = """
    MATCH (qc:Concept {name: $query_name}) 
    OPTIONAL MATCH (qc)-[r_out]->(n_out:Concept) 
    WITH qc, COLLECT(DISTINCT {type: type(r_out), domain: n_out.domain, direction: 'OUTGOING'}) AS qc_out_patterns 
    OPTIONAL MATCH (qc)<-[r_in]-(n_in:Concept) 
    WITH qc, qc_out_patterns, COLLECT(DISTINCT {type: type(r_in), domain: n_in.domain, direction: 'INCOMING'}) AS qc_in_patterns 
    RETURN qc_out_patterns + qc_in_patterns AS qc_signature
    """
    
    try:
        # Use the new run_cypher_query method
        qc_signature_result = graph_handler.run_cypher_query(query_signature_cypher, {"query_name": query_concept_name})
        if not qc_signature_result or not qc_signature_result[0]["qc_signature"]:
            logger.warning(f"Could not retrieve neighborhood signature for query concept '{query_concept_name}'.")
            return []
        
        # Filter out null entries that can occur if a concept has no outgoing or no incoming relationships
        query_concept_signature = [p for p in qc_signature_result[0]["qc_signature"] if p['type'] is not None and p['domain'] is not None]
        if not query_concept_signature:
            logger.info(f"Query concept '{query_concept_name}' has no defined neighborhood patterns to match.")
            return []

        # Step 2: Find other concepts and score them based on shared signature elements
        # We pass the signature as a parameter to the main query.
        # Note: Passing complex structures like lists of maps as Cypher parameters can be tricky or unsupported
        # depending on the driver version and Neo4j version. An alternative is to construct parts of the query string
        # or multiple queries. For now, let's try a more direct comparison approach in a single query if possible,
        # or simplify by matching specific patterns from the signature.

        # Revised strategy: Iterate through all other concepts and calculate their signatures on the fly,
        # then compare with the query_concept_signature in Python. This is less efficient for large graphs
        # but more robust than trying to pass complex list structures to Cypher or building huge queries.

        # For an initial, simpler Cypher-based approach that does scoring in the DB:
        # This query finds other concepts and counts how many of its 1-hop relationships
        # (type, neighbor_domain, direction) match any of those in the query_concept_signature.
        # This is a bit of a simplification of true structural isomorphism but a good start.

        structural_analogy_query = """
        // Get all patterns for the query concept (qc)
        MATCH (qc:Concept {name: $query_name})
        OPTIONAL MATCH (qc)-[r_out]->(n_out:Concept)
        WITH qc, COLLECT(DISTINCT {type: type(r_out), domain: n_out.domain, direction: 'OUTGOING'}) AS qc_out_p
        OPTIONAL MATCH (qc)<-[r_in]-(n_in:Concept)
        WITH qc, qc_out_p, COLLECT(DISTINCT {type: type(r_in), domain: n_in.domain, direction: 'INCOMING'}) AS qc_in_p
        WITH qc, [p IN qc_out_p + qc_in_p WHERE p.type IS NOT NULL AND p.domain IS NOT NULL] AS qc_patterns
        
        // Iterate over other concepts (oc)
        MATCH (oc:Concept)
        WHERE oc <> qc AND NOT oc.name IN $excluded_names // Exclude query concept and specified others
        
        // Get all patterns for the other concept (oc)
        OPTIONAL MATCH (oc)-[r_oc_out]->(n_oc_out:Concept)
        WITH qc, qc_patterns, oc, COLLECT(DISTINCT {type: type(r_oc_out), domain: n_oc_out.domain, direction: 'OUTGOING'}) AS oc_out_p
        OPTIONAL MATCH (oc)<-[r_oc_in]-(n_oc_in:Concept)
        WITH qc, qc_patterns, oc, oc_out_p, COLLECT(DISTINCT {type: type(r_oc_in), domain: n_oc_in.domain, direction: 'INCOMING'}) AS oc_in_p
        WITH qc_patterns, oc, [p IN oc_out_p + oc_in_p WHERE p.type IS NOT NULL AND p.domain IS NOT NULL] AS oc_patterns
        
        // Calculate similarity score (count of shared patterns)
        WITH oc, qc_patterns, oc_patterns,
             REDUCE(score = 0, p_qc IN qc_patterns | score + CASE WHEN p_qc IN oc_patterns THEN 1 ELSE 0 END) AS shared_pattern_count
        WHERE shared_pattern_count > 0 // Only consider concepts with at least one shared pattern
        
        RETURN oc.name AS name, oc.domain AS domain, shared_pattern_count AS score
        ORDER BY shared_pattern_count DESC, name ASC // Secondary sort for consistent ordering
        LIMIT $k
        """

        params = {
            "query_name": query_concept_name,
            "excluded_names": excluded_concept_names,
            "k": k
        }

        # Use the new run_cypher_query method
        results = graph_handler.run_cypher_query(structural_analogy_query, params)
        
        analogous_concepts = []
        if results:
            for record in results:
                analogous_concepts.append({
                    "name": record["name"],
                    "domain": record.get("domain", "Unknown"), # oc.domain might be null
                    "score": record["score"],
                    "reason": f"Shares {record['score']} structural neighborhood patterns with '{query_concept_name}'."
                })
        
        logger.info(f"Found {len(analogous_concepts)} structural analogies for '{query_concept_name}'.")
        return analogous_concepts

    except Exception as e:
        logger.error(f"Error during structural graph analogy search for '{query_concept_name}': {e}", exc_info=True)
        return []


def generate_analogy_explanation_gemini(
    concept1_name: str,
    concept2_name: str,
    concept1_context: Optional[Dict[str, Any]] = None, # From feature_extractor
    concept2_context: Optional[Dict[str, Any]] = None, # From feature_extractor
    analogy_type: str = "general" # "vector" or "structural" or "general"
) -> Optional[str]:
    """
    Uses Gemini to generate an explanation for why two concepts might be analogous.

    Args:
        concept1_name (str): Name of the first concept.
        concept2_name (str): Name of the second concept.
        concept1_context (Optional[Dict[str, Any]]): Rich context for concept 1.
        concept2_context (Optional[Dict[str, Any]]): Rich context for concept 2.
        analogy_type (str): Hint about the type of analogy found.

    Returns:
        Optional[str]: A textual explanation of the analogy, or None if failed.
    """
    logger.info(f"Generating Gemini explanation for analogy between '{concept1_name}' and '{concept2_name}'.")
    gemini_model = _get_gemini_model_for_analogy()
    if not gemini_model:
        logger.error("Gemini model not available. Cannot generate analogy explanation.")
        return None

    prompt_lines = [
        f"Explain the analogy between the concept \"{concept1_name}\" and \"{concept2_name}\".",
        "Consider their potential similarities in structure, function, or the domains they belong to.",
        f"The type of analogy detected was: {analogy_type}.",
        "" # Empty line for spacing
    ]

    if concept1_context and concept1_context.get("properties"):
        prompt_lines.append(f"Context for {concept1_name}:")
        prompt_lines.append(f"- Domain: {concept1_context['properties'].get('domain', 'N/A')}")
        if concept1_context.get("outgoing_relationships"):
            related_to_str = ", ".join([
                f"{r.get('target_properties',{}).get('name','Unknown')} (as {r.get('type','related')})"
                for r in concept1_context['outgoing_relationships'][:3]
            ])
            prompt_lines.append(f"- Related to: {related_to_str}")
        if concept1_context.get("incoming_relationships"):
            target_of_str = ", ".join([
                f"{r.get('source_properties',{}).get('name','Unknown')} (as {r.get('type','related')})"
                for r in concept1_context['incoming_relationships'][:3]
            ])
            prompt_lines.append(f"- Is a target of: {target_of_str}")
        prompt_lines.append("") # Empty line for spacing

    if concept2_context and concept2_context.get("properties"):
        prompt_lines.append(f"Context for {concept2_name}:")
        prompt_lines.append(f"- Domain: {concept2_context['properties'].get('domain', 'N/A')}")
        if concept2_context.get("outgoing_relationships"):
            related_to_str_2 = ", ".join([
                f"{r.get('target_properties',{}).get('name','Unknown')} (as {r.get('type','related')})"
                for r in concept2_context['outgoing_relationships'][:3]
            ])
            prompt_lines.append(f"- Related to: {related_to_str_2}")
        if concept2_context.get("incoming_relationships"):
            target_of_str_2 = ", ".join([
                f"{r.get('source_properties',{}).get('name','Unknown')} (as {r.get('type','related')})"
                for r in concept2_context['incoming_relationships'][:3]
            ])
            prompt_lines.append(f"- Is a target of: {target_of_str_2}")
        prompt_lines.append("") # Empty line for spacing

    prompt_lines.append("Provide a concise explanation highlighting the key aspects of their analogy.")
    prompt = "\\n".join(prompt_lines)

    logger.debug(f"Analogy explanation prompt for Gemini:\\n{prompt}")

    try:
        response = gemini_model.generate_content(prompt)
        explanation = response.text.strip()
        logger.info(f"Successfully generated analogy explanation for '{concept1_name}' and '{concept2_name}'.")
        return explanation
    except Exception as e:
        logger.error(f"Error calling Gemini for analogy explanation: {e}", exc_info=True)
        return None


if __name__ == '__main__':
    logger.info("Running analogy_finder.py direct tests...")

    # --- Setup --- 
    graph_handler_instance = None
    faiss_handler_instance = None
    try:
        graph_handler_instance = GraphHandler(
            uri=config.NEO4J_URI, 
            user=config.NEO4J_USER, 
            password=config.NEO4J_PASSWORD
        )
        faiss_handler_instance = FaissHandler(
            embedding_model_name=config.EMBEDDING_MODEL_NAME, 
            faiss_index_path=config.FAISS_INDEX_PATH, 
            concept_list_path=config.FAISS_CONCEPT_LIST_PATH
        )
        faiss_handler_instance.load_index()
        from polymind_core.synthesis_engine.feature_extractor import get_concept_context

    except Exception as e:
        logger.error(f"Failed to initialize handlers for testing: {e}", exc_info=True)
        # Ensure instances are None if setup fails partway
        if graph_handler_instance and graph_handler_instance._graph_db_connection is None:
             graph_handler_instance = None # If connection failed within GraphHandler init
        # No specific check for faiss needed here as error is general

    # Check if GraphHandler has a valid connection and FaissHandler has loaded its index
    if graph_handler_instance and graph_handler_instance._graph_db_connection is not None and \
       faiss_handler_instance and faiss_handler_instance._faiss_index is not None and faiss_handler_instance._concept_list is not None:
        
        test_concept = "Artificial intelligence" # Choose a concept likely in your graph
        logger.info(f"--- Testing with concept: '{test_concept}' ---")

        # Test Vector Analogies
        logger.info(f"\n--- Finding Vector Analogies for '{test_concept}' ---")
        vector_analogies = find_analogous_concepts_vector(
            test_concept, 
            faiss_handler_instance, 
            graph_handler_instance, 
            k=3
        )
        if vector_analogies:
            for ana in vector_analogies:
                logger.info(f"  - Name: {ana['name']}, Domain: {ana.get('domain', 'N/A')}, Score: {ana['score']:.4f}")
                if ana['name'] != test_concept:
                    logger.info(f"    Generating explanation for analogy with '{ana['name']}'...")
                    ctx1 = get_concept_context(graph_handler_instance, test_concept)
                    ctx2 = get_concept_context(graph_handler_instance, ana['name'])
                    explanation = generate_analogy_explanation_gemini(test_concept, ana['name'], ctx1, ctx2, "vector")
                    logger.info(f"    Explanation: {explanation if explanation else 'Could not generate.'}")
                    break 
        else:
            logger.info(f"No vector analogies found for '{test_concept}'.")

        # Test Structural Analogies
        logger.info(f"\n--- Finding Structural Analogies for '{test_concept}' ---")
        structural_analogies = find_analogous_concepts_structural_graph(
            test_concept, 
            graph_handler_instance, 
            k=3
        )
        if structural_analogies:
            for ana in structural_analogies:
                logger.info(f"  - Name: {ana['name']}, Domain: {ana.get('domain', 'N/A')}, Score: {ana['score']}, Reason: {ana.get('reason')}")
                if ana['name'] != test_concept:
                    logger.info(f"    Generating explanation for analogy with '{ana['name']}'...")
                    ctx1 = get_concept_context(graph_handler_instance, test_concept)
                    ctx2 = get_concept_context(graph_handler_instance, ana['name'])
                    explanation = generate_analogy_explanation_gemini(test_concept, ana['name'], ctx1, ctx2, "structural")
                    logger.info(f"    Explanation: {explanation if explanation else 'Could not generate.'}")
                    break
        else:
            logger.info(f"No structural analogies found for '{test_concept}'.")
        
    else:
        logger.error("GraphHandler or FaissHandler not properly initialized (or Neo4j connection failed / FAISS index missing). Skipping direct test run.")
        # Attempt to close graph_handler if it was instantiated but connection failed or other setup issue
        # No explicit close needed for py2neo.Graph in this handler's design
        # if graph_handler_instance and hasattr(graph_handler_instance, 'close') and callable(getattr(graph_handler_instance, 'close')):
        #     logger.info("Attempting to close graph_handler_instance post-error.")
        #     graph_handler_instance.close()
            
    logger.info("analogy_finder.py direct tests completed.") 