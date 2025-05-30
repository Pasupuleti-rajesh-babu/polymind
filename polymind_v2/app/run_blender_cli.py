import argparse
import logging
import sys
import os
import pprint

# Adjust sys.path to allow importing from polymind_core
# This assumes the script is in polymind_v2/app/
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir) # This should be polymind_v2/
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from polymind_core.synthesis_engine import feature_extractor, blender
    from polymind_core.knowledge_graph import graph_handler # To initialize connection
    import config # For logging setup and API key checks
except ImportError as e:
    print(f"Error importing polymind_core modules: {e}")
    print("Please ensure that the polymind_v2 directory is in your PYTHONPATH or run this script from within the polymind_v2/app directory.")
    sys.exit(1)

# Configure logging (can be minimal for CLI, or use config)
log_level = getattr(config, 'LOG_LEVEL', 'INFO')
log_format = getattr(config, 'LOG_FORMAT', '%(asctime)s - %(levelname)s - %(message)s')
# Configure root logger or specific loggers if preferred
# For CLI, sometimes simpler logging is fine, but using project's config is consistent.
# logging.basicConfig(level=log_level, format=log_format)
# Instead of basicConfig, get specific loggers to avoid overriding Streamlit's default if this grows
cli_logger = logging.getLogger("run_blender_cli")
# Set level for this specific logger; handlers might need to be added if not configured by root
# For now, rely on other modules initializing their loggers based on config.

def main():
    parser = argparse.ArgumentParser(description="Perform conceptual blending on two concepts from the knowledge graph.")
    parser.add_argument("concept_a_name", type=str, help="The name of the first concept for blending.")
    parser.add_argument("concept_b_name", type=str, help="The name of the second concept for blending.")
    args = parser.parse_args()

    cli_logger.info(f"Starting conceptual blending process for '{args.concept_a_name}' and '{args.concept_b_name}'.")

    # Check for API keys and Neo4j config before proceeding
    if not config.GEMINI_API_KEY or config.GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        cli_logger.error("GEMINI_API_KEY is not configured in .env or config.py. Blending will fail.")
        sys.exit(1)
    if not config.NEO4J_PASSWORD or config.NEO4J_PASSWORD == "YOUR_NEO4J_PASSWORD_HERE":
        cli_logger.warning("NEO4J_PASSWORD in .env or config.py seems to be a placeholder. Ensure Neo4j is accessible.")

    # Ensure Neo4j connection is attempted (it's handled globally by graph_handler)
    graph_db_conn = graph_handler.get_neo4j_graph()
    if not graph_db_conn:
        cli_logger.error("Failed to connect to Neo4j. Cannot retrieve concept contexts. Aborting.")
        sys.exit(1)
    cli_logger.info("Successfully connected to Neo4j.")

    # 1. Get context for Concept A
    cli_logger.info(f"Retrieving context for Concept A: '{args.concept_a_name}'...")
    context_a = feature_extractor.get_concept_context(args.concept_a_name)
    if not context_a:
        cli_logger.error(f"Could not retrieve context for Concept A: '{args.concept_a_name}'. Ensure it exists in the graph.")
        sys.exit(1)
    cli_logger.info(f"Successfully retrieved context for Concept A: '{args.concept_a_name}'.")

    # 2. Get context for Concept B
    cli_logger.info(f"Retrieving context for Concept B: '{args.concept_b_name}'...")
    context_b = feature_extractor.get_concept_context(args.concept_b_name)
    if not context_b:
        cli_logger.error(f"Could not retrieve context for Concept B: '{args.concept_b_name}'. Ensure it exists in the graph.")
        sys.exit(1)
    cli_logger.info(f"Successfully retrieved context for Concept B: '{args.concept_b_name}'.")

    # 3. Perform conceptual blend
    cli_logger.info(f"Performing conceptual blend for '{args.concept_a_name}' and '{args.concept_b_name}'...")
    blended_result = blender.blend_concepts_gemini(context_a, context_b)

    # 4. Print the result
    if blended_result:
        print("\n--- Conceptual Blend Result ---")
        print(f"**Blended Concept Name:** {blended_result.get('blended_name', 'N/A')}")
        print("\n**Blended Concept Description:**")
        print(blended_result.get('description', 'N/A'))
        print("\n**Key Combined Features:**")
        if blended_result.get("combined_features"):
            for feature in blended_result["combined_features"]:
                print(f"- {feature}")
        else:
            print("N/A")
        print("\n**Potential Emergent Properties:**")
        if blended_result.get("emergent_properties"):
            for prop in blended_result["emergent_properties"]:
                print(f"- {prop}")
        else:
            print("N/A")
        print("\n-----------------------------")
    else:
        cli_logger.error("Conceptual blending failed or did not produce a parseable result.")
        sys.exit(1)

    cli_logger.info("Conceptual blending process completed.")

if __name__ == "__main__":
    main() 