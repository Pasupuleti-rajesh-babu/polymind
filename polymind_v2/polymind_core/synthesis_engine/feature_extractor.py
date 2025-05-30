import logging
from typing import List, Dict, Any, Optional

# Attempt to import project-specific modules
try:
    from ... import config # For base project path context
    from ..knowledge_graph.graph_handler import GraphHandler 
except ImportError:
    # Fallback for running script directly from this directory or if polymind_v2 is in PYTHONPATH
    import sys
    import os
    # Add the parent directory of polymind_core (which is polymind_v2) to sys.path
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    polymind_core_dir = os.path.dirname(current_script_dir)
    project_root = os.path.dirname(polymind_core_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    import config
    from polymind_core.knowledge_graph.graph_handler import GraphHandler

# Configure logging
log_level = getattr(config, 'LOG_LEVEL', 'INFO')
log_format = getattr(config, 'LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.basicConfig(level=log_level, format=log_format)
logger = logging.getLogger(__name__)

def get_concept_context(graph_handler_instance: GraphHandler, concept_name: str) -> Optional[Dict[str, Any]]:
    """
    Retrieves a comprehensive context for a concept from Neo4j.
    This includes its properties, and its 1-hop incoming/outgoing relationships 
    along with the properties of connected concepts.

    Args:
        graph_handler_instance (GraphHandler): An initialized instance of GraphHandler.
        concept_name (str): The name of the concept.

    Returns:
        Optional[Dict[str, Any]]: A dictionary containing the concept's context, 
                                   or None if the concept is not found or an error occurs.
        The structure is:
        {
            "concept_name": "NormalizedName",
            "properties": {prop1: val1, ...}, // Properties of the main concept
            "outgoing_relationships": [
                {"type": "REL_TYPE", "target_properties": {name: "TargetName", domain: "DomainX", ...}},
                ...
            ],
            "incoming_relationships": [
                {"type": "REL_TYPE", "source_properties": {name: "SourceName", domain: "DomainY", ...}},
                ...
            ]
        }
    """
    if not graph_handler_instance:
        logger.error("GraphHandler instance not provided to get_concept_context.")
        return None

    graph = graph_handler_instance._get_graph() # Use the provided instance
        
    if not graph:
        logger.error("Neo4j connection not available (GraphHandler._get_graph() returned None). Cannot get concept context.")
        return None

    norm_concept_name = concept_name.strip().capitalize()
    context_data: Dict[str, Any] = {
        "concept_name": norm_concept_name,
        "properties": {},
        "outgoing_relationships": [],
        "incoming_relationships": []
    }

    try:
        # 1. Get properties of the main concept
        concept_query = "MATCH (c:Concept {name: $name}) RETURN properties(c) AS props"
        concept_result = graph.run(concept_query, name=norm_concept_name).data()
        if not concept_result or 'props' not in concept_result[0] or concept_result[0]['props'] is None:
            logger.warning(f"Concept '{norm_concept_name}' not found in Neo4j.")
            return None 
        context_data["properties"] = concept_result[0]['props']
        logger.info(f"Retrieved properties for concept '{norm_concept_name}'.")

        # 2. Get outgoing relationships and target concept properties
        outgoing_query = """
        MATCH (source:Concept {name: $name})-[r]->(target:Concept)
        RETURN type(r) AS relationship_type, properties(target) AS target_properties
        """
        outgoing_results = graph.run(outgoing_query, name=norm_concept_name).data()
        for record in outgoing_results:
            context_data["outgoing_relationships"].append({
                "type": record["relationship_type"],
                "target_properties": record["target_properties"]
            })
        logger.info(f"Retrieved {len(outgoing_results)} outgoing relationships for '{norm_concept_name}'.")

        # 3. Get incoming relationships and source concept properties
        incoming_query = """
        MATCH (source:Concept)-[r]->(target:Concept {name: $name})
        RETURN type(r) AS relationship_type, properties(source) AS source_properties
        """
        incoming_results = graph.run(incoming_query, name=norm_concept_name).data()
        for record in incoming_results:
            context_data["incoming_relationships"].append({
                "type": record["relationship_type"],
                "source_properties": record["source_properties"]
            })
        logger.info(f"Retrieved {len(incoming_results)} incoming relationships for '{norm_concept_name}'.")
        
        return context_data

    except Exception as e:
        logger.error(f"Error retrieving context for concept '{norm_concept_name}' from Neo4j: {e}", exc_info=True)
        return None

if __name__ == '__main__':
    import pprint
    logger.info("Running feature_extractor.py direct tests...")

    if not config.NEO4J_PASSWORD or config.NEO4J_PASSWORD == "YOUR_NEO4J_PASSWORD_HERE":
        logger.warning("NEO4J_PASSWORD in config.py seems to be a placeholder. Ensure Neo4j is accessible with correct credentials.")

    test_graph_handler = None
    try:
        test_graph_handler = GraphHandler(uri=config.NEO4J_URI, user=config.NEO4J_USER, password=config.NEO4J_PASSWORD)
        logger.info("Successfully connected to Neo4j for testing (via GraphHandler).")
        
        # --- Test with a concept likely ingested by wikipedia_ingestor.py ---
        test_concept_name_1 = "Artificial intelligence" 
        logger.info(f"\n--- Attempting to get context for: '{test_concept_name_1}' ---")
        context1 = get_concept_context(test_graph_handler, test_concept_name_1)
        if context1:
            logger.info(f"Successfully retrieved context for '{test_concept_name_1}'.")
            logger.info(f"  Properties: {context1.get('properties')}")
            logger.info(f"  Outgoing relationships count: {len(context1.get('outgoing_relationships', []))}")
            logger.info(f"  Incoming relationships count: {len(context1.get('incoming_relationships', []))}")
            if context1.get('outgoing_relationships'):
                logger.info(f"    Sample outgoing: Type='{context1['outgoing_relationships'][0]['type']}', Target='{context1['outgoing_relationships'][0]['target_properties'].get('name')}'")
            if context1.get('incoming_relationships'):
                 logger.info(f"    Sample incoming: Type='{context1['incoming_relationships'][0]['type']}', Source='{context1['incoming_relationships'][0]['source_properties'].get('name')}'")
        else:
            logger.warning(f"Could not retrieve context for '{test_concept_name_1}'. It might not be in the graph or an error occurred.")

        # --- Test with a concept that might not exist ---
        test_concept_name_2 = "NonExistentConcept123"
        logger.info(f"\n--- Attempting to get context for non-existent concept: '{test_concept_name_2}' ---")
        context2 = get_concept_context(test_graph_handler, test_concept_name_2)
        if context2 is None: 
            logger.info(f"Correctly received None for non-existent concept '{test_concept_name_2}'.")
        else:
            logger.error(f"Unexpectedly received context for non-existent concept '{test_concept_name_2}': {context2}")

        # --- Test with another potentially ingested concept ---
        test_concept_name_3 = "Dna"
        logger.info(f"\n--- Attempting to get context for: '{test_concept_name_3}' ---")
        context3 = get_concept_context(test_graph_handler, test_concept_name_3)
        if context3:
            logger.info(f"Successfully retrieved context for '{test_concept_name_3}'.")
            logger.info(f"  Properties: {context3.get('properties')}")
            logger.info(f"  Outgoing relationships count: {len(context3.get('outgoing_relationships', []))}")
            logger.info(f"  Incoming relationships count: {len(context3.get('incoming_relationships', []))}")
        else:
            logger.warning(f"Could not retrieve context for '{test_concept_name_3}'.")

    except Exception as e:
        logger.error(f"Failed to initialize GraphHandler for testing or error during tests: {e}", exc_info=True)
    finally:
        if test_graph_handler and hasattr(test_graph_handler, 'close'): # If GraphHandler gets a close method
            logger.info("Closing test GraphHandler connection.")
            test_graph_handler.close()


    logger.info("\nfeature_extractor.py direct tests completed.") 