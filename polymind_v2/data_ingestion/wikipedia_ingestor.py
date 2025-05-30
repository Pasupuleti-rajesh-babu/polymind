import wikipedia
import logging
from typing import List, Tuple, Optional

# Attempt to import project-specific modules
try:
    from .. import config
    from ..polymind_core.meaning_extraction.text_processor import TextProcessor
    from ..polymind_core.knowledge_graph.graph_handler import GraphHandler
    from ..polymind_core.vector_store.faiss_handler import FaissHandler
except ImportError:
    # Fallback for running script directly from data_ingestion directory or if polymind_v2 is in PYTHONPATH
    # This assumes that polymind_v2 directory is effectively the root for these imports
    import sys
    import os
    # Add the parent directory (polymind_v2) to sys.path to find config and polymind_core
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir) 
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    import config # Now this should work if config.py is in polymind_v2/
    from polymind_core.meaning_extraction.text_processor import TextProcessor
    from polymind_core.knowledge_graph.graph_handler import GraphHandler
    from polymind_core.vector_store.faiss_handler import FaissHandler

# Configure logging - use settings from config if available, otherwise basic
log_level = getattr(config, 'LOG_LEVEL', 'INFO')
log_format = getattr(config, 'LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.basicConfig(level=log_level, format=log_format)
logger = logging.getLogger(__name__)

def ingest_wikipedia_article(
    page_title: str, 
    graph_handler_instance: GraphHandler, 
    text_processor_instance: TextProcessor, 
    faiss_handler_instance: FaissHandler
) -> Tuple[bool, str]:
    """
    Fetches content for a single Wikipedia page title, extracts information,
    and loads it into Neo4j and FAISS using the provided handler instances.

    Args:
        page_title (str): The title of the Wikipedia page to ingest.
        graph_handler_instance (GraphHandler): An initialized instance of GraphHandler.
        text_processor_instance (TextProcessor): An initialized instance of TextProcessor.
        faiss_handler_instance (FaissHandler): An initialized instance of FaissHandler.

    Returns:
        Tuple[bool, str]: (success_status, message)
    """
    logger.info(f"--- Processing Wikipedia page: {page_title} ---")
    try:
        page = wikipedia.page(page_title, auto_suggest=False, redirect=True)
        content = page.content
        logger.info(f"Successfully fetched content for '{page_title}' (approx. {len(content)} chars).")

        if not text_processor_instance._get_gemini_model(): # Check if Gemini model is available
            msg = f"Gemini model not available in TextProcessor. Cannot process '{page_title}'."
            logger.error(msg)
            return False, msg

        logger.info(f"Extracting meaning atoms from content of '{page_title}'...")
        extracted_concepts, extracted_triples, raw_gemini_output = \
            text_processor_instance.extract_meaning_atoms_from_text(content)
        
        if not extracted_concepts and not extracted_triples:
            msg = f"No concepts or triples extracted by Gemini for '{page_title}'. Raw output: {raw_gemini_output[:200]}..."
            logger.warning(msg)
            # return True, msg # Still counts as "processed" but with no data. Or False if this is an error.
            # For UI, let's say it's a partial success (processed, but no data found)

        # Process extracted concepts
        if extracted_concepts:
            logger.info(f"Suggesting domains for {len(extracted_concepts)} concepts from '{page_title}'...")
            unique_page_concepts = sorted(list(set(extracted_concepts)))
            
            suggested_domains_map = text_processor_instance.suggest_domains_for_concepts(unique_page_concepts)
            
            concepts_added_to_graph_count = 0
            for concept_name in unique_page_concepts:
                domain = suggested_domains_map.get(concept_name, config.DEFAULT_DOMAIN)
                if graph_handler_instance.add_concept(concept_name, domain):
                    concepts_added_to_graph_count += 1
            logger.info(f"Added/merged {concepts_added_to_graph_count} unique concepts from '{page_title}' to Neo4j.")

            logger.info(f"Adding {len(unique_page_concepts)} unique concepts from '{page_title}' to FAISS index...")
            if not faiss_handler_instance.add_concepts_to_faiss(unique_page_concepts):
                logger.warning(f"Failed to add some or all concepts from '{page_title}' to FAISS.")
        else:
            logger.info(f"No concepts extracted from '{page_title}'.")

        # Process extracted triples
        if extracted_triples:
            logger.info(f"Adding {len(extracted_triples)} triples from '{page_title}' to Neo4j...")
            triples_added_count = 0
            for s, r, o in extracted_triples:
                if graph_handler_instance.add_relationship(s, r, o):
                    triples_added_count +=1
            logger.info(f"Successfully added {triples_added_count} triples from '{page_title}' to Neo4j.")
        else:
            logger.info(f"No triples extracted from '{page_title}'.")
        
        msg = f"Successfully processed '{page_title}'. Concepts found: {len(extracted_concepts)}, Triples found: {len(extracted_triples)}."
        logger.info(f"--- Finished processing Wikipedia page: {page_title} ---")
        return True, msg

    except wikipedia.exceptions.PageError:
        msg = f"Wikipedia page '{page_title}' not found."
        logger.error(msg)
        return False, msg
    except wikipedia.exceptions.DisambiguationError as e:
        msg = f"Wikipedia page '{page_title}' is a disambiguation page. Options: {e.options[:5]}. Please be more specific."
        logger.warning(msg)
        return False, msg
    except Exception as e:
        msg = f"An unexpected error occurred while processing '{page_title}': {str(e)}"
        logger.error(msg, exc_info=True)
        return False, msg

def ingest_wikipedia_pages(
    page_titles: List[str], 
    graph_handler_instance: GraphHandler, 
    text_processor_instance: TextProcessor, 
    faiss_handler_instance: FaissHandler
):
    """
    Iterates through a list of Wikipedia page titles and ingests each one
    using the provided handler instances.
    """
    if not page_titles:
        logger.info("No page titles provided for ingestion.")
        return

    for title in page_titles:
        success, message = ingest_wikipedia_article(
            title, 
            graph_handler_instance, 
            text_processor_instance, 
            faiss_handler_instance
        )
        if success:
            logger.info(message)
        else:
            logger.error(f"Failed to ingest '{title}': {message}")


if __name__ == '__main__':
    logger.info("Starting Wikipedia Ingestion Process (Batch Mode)...")

    # Initialize handlers for batch mode
    try:
        gh = GraphHandler(uri=config.NEO4J_URI, user=config.NEO4J_USER, password=config.NEO4J_PASSWORD)
        tp = TextProcessor(api_key=config.GEMINI_API_KEY, model_name=config.GEMINI_MODEL_NAME)
        fh = FaissHandler(
            faiss_index_path=config.FAISS_INDEX_PATH, 
            concept_list_path=config.FAISS_CONCEPT_LIST_PATH, # Make sure this is defined in config or derived
            embedding_model_name=config.DEFAULT_EMBEDDING_MODEL
        )
    except Exception as e:
        logger.error(f"Failed to initialize core handlers for batch ingestion: {e}")
        logger.error("Aborting batch ingestion.")
        sys.exit(1)

    if not tp._get_gemini_model(): # Check if Gemini is actually available
        logger.error("Gemini model could not be initialized in TextProcessor. Batch ingestion requires Gemini. Aborting.")
        sys.exit(1)
    
    if fh._faiss_index is None: # Check if FAISS is initialized
        logger.error("FAISS index could not be initialized in FaissHandler. Batch ingestion requires FAISS. Aborting.")
        sys.exit(1)

    pages_to_ingest = [
        "Artificial intelligence", "Machine learning", "Deep learning",
        "Category theory", "Semiotics", "Cognitive linguistics",
        "Conceptual blending", "Embodied cognition", "Knowledge graph",
        "Neo4j", "Sentence embedding",
        "Photosynthesis", "Market economy", "Democracy", "DNA"
    ]

    if not config.GEMINI_API_KEY or config.GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        logger.error("GEMINI_API_KEY is not configured. Batch ingestion will likely fail or be limited.")
    if not config.NEO4J_PASSWORD or config.NEO4J_PASSWORD == "YOUR_NEO4J_PASSWORD_HERE":
        logger.warning("NEO4J_PASSWORD seems to be a placeholder. Ensure Neo4j is accessible.")

    ingest_wikipedia_pages(pages_to_ingest, gh, tp, fh)

    logger.info("Wikipedia Ingestion Process (Batch Mode) Completed.") 