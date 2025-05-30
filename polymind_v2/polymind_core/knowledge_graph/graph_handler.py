from py2neo import Graph, Node, Relationship, Subgraph
from typing import Optional, List, Dict, Tuple
import logging

# Attempt to import configuration from the new project structure
try:
    from ... import config # If graph_handler is used as part of a larger package run
except ImportError:
    # Fallback for direct execution or if the above relative import fails
    # This assumes config.py is in a directory that Python can find (e.g., polymind_v2/)
    # or that polymind_v2 is in PYTHONPATH or the script is run from polymind_v2's parent.
    # For robustness, direct imports from a known package root are better if polymind_v2 is a package.
    import config # Make sure polymind_v2 directory is in PYTHONPATH if running scripts directly from subdirs

# Configure logging
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

class GraphHandler:
    def __init__(self, uri: str, user: Optional[str] = None, password: Optional[str] = None):
        """
        Initializes the GraphHandler and establishes a connection to Neo4j.

        Args:
            uri (str): The URI for the Neo4j database.
            user (Optional[str]): Username for Neo4j authentication.
            password (Optional[str]): Password for Neo4j authentication.
        """
        self._graph_db_connection = None
        logger.info(f"Attempting to connect to Neo4j: {uri}")
        try:
            if user and password:
                self._graph_db_connection = Graph(uri, auth=(user, password))
            else:
                self._graph_db_connection = Graph(uri)
            # Test connection
            self._graph_db_connection.run("MATCH (n) RETURN count(n) AS count").data()
            logger.info("Successfully connected to Neo4j.")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            self._graph_db_connection = None # Ensure it's reset on failure
            raise # Reraise the exception to signal connection failure

    def _get_graph(self) -> Optional[Graph]:
        """Returns the active Neo4j graph connection."""
        if self._graph_db_connection is None:
            logger.error("Neo4j connection is not available. Was initialization successful?")
        return self._graph_db_connection

    def add_concept(self, name: str, domain: str, properties: Optional[dict] = None) -> bool:
        """
        Adds or merges a Concept node in Neo4j.

        Args:
            name (str): The name of the concept (will be capitalized).
            domain (str): The domain of the concept.
            properties (Optional[dict]): Additional properties for the node.

        Returns:
            bool: True if successful, False otherwise.
        """
        graph = self._get_graph()
        if not graph:
            logger.error("Neo4j connection not available. Cannot add concept.")
            return False

        try:
            normalized_name = name.strip().capitalize()
            normalized_domain = domain.strip().capitalize() if domain else config.DEFAULT_DOMAIN

            node_properties = {"name": normalized_name, "domain": normalized_domain}
            if properties:
                node_properties.update(properties)
            
            query = "MERGE (c:Concept {name: $name}) SET c += $props"
            graph.run(query, name=normalized_name, props=node_properties)
            logger.info(f"Successfully merged concept: '{normalized_name}' with domain '{normalized_domain}'")
            return True
        except Exception as e:
            logger.error(f"Error adding/merging concept '{name}': {e}")
            return False

    def add_relationship(self, source_concept_name: str, rel_type: str, target_concept_name: str, properties: Optional[dict] = None) -> bool:
        """
        Adds or merges a relationship between two Concept nodes.
        Nodes are created if they don't exist.

        Args:
            source_concept_name (str): Name of the source concept.
            rel_type (str): Type of the relationship (should be from ALLOWED_RELATIONSHIP_TYPES after canonicalization).
            target_concept_name (str): Name of the target concept.
            properties (Optional[dict]): Additional properties for the relationship.

        Returns:
            bool: True if successful, False otherwise.
        """
        graph = self._get_graph()
        if not graph:
            logger.error("Neo4j connection not available. Cannot add relationship.")
            return False

        try:
            norm_source_name = source_concept_name.strip().capitalize()
            norm_target_name = target_concept_name.strip().capitalize()

            if rel_type not in config.ALLOWED_RELATIONSHIP_TYPES:
                logger.warning(f"Relationship type '{rel_type}' is not in ALLOWED_RELATIONSHIP_TYPES. Defaulting to RELATED_TO.")
                logger.warning(f"Original triple: ({norm_source_name})-[:{rel_type}]->({norm_target_name})")
                original_rel_type = rel_type
                rel_type = "RELATED_TO"
                if properties is None:
                    properties = {}
                properties["original_type"] = original_rel_type
            
            query = (
                "MERGE (s:Concept {name: $source_name}) "
                "ON CREATE SET s.domain = $default_domain "
                "MERGE (t:Concept {name: $target_name}) "
                "ON CREATE SET t.domain = $default_domain "
                f"MERGE (s)-[r:{rel_type}]->(t) "
                "SET r += $props"
            )
            
            rel_props_to_set = properties if properties is not None else {}

            graph.run(query, 
                      source_name=norm_source_name, 
                      target_name=norm_target_name, 
                      props=rel_props_to_set, 
                      default_domain=config.DEFAULT_DOMAIN)
            logger.info(f"Successfully merged relationship: ('{norm_source_name}')-[:{rel_type}]->('{norm_target_name}')")
            return True
        except Exception as e:
            logger.error(f"Error adding/merging relationship ('{source_concept_name}')-[:{rel_type}]->('{target_concept_name}'): {e}")
            return False

    def get_all_concept_names(self) -> List[str]:
        """Retrieves all distinct concept names from Neo4j."""
        graph = self._get_graph()
        if not graph:
            logger.error("Neo4j connection not available.")
            return []
        try:
            query_result = graph.run("MATCH (c:Concept) RETURN DISTINCT c.name AS name").data()
            concepts = [record['name'] for record in query_result if record['name'] is not None]
            logger.info(f"Retrieved {len(concepts)} distinct concept names from Neo4j.")
            return concepts
        except Exception as e:
            logger.error(f"Error fetching all concepts from Neo4j: {e}")
            return []

    def get_concept_features(self, concept_name: str, max_features: int = 10) -> List[Dict[str, str]]:
        """
        Retrieves 1-hop features (relationships and connected nodes) for a concept from Neo4j.
        Returns a list of dictionaries, each representing a triple involving the concept.
        e.g., [{"source": "ConceptA", "relationship": "IS_A", "target": "Entity"}]
        This method is used by the UI to display details and by the synthesis engine.
        """
        graph = self._get_graph()
        if not graph:
            logger.error("Neo4j connection not available.")
            return []

        norm_concept_name = concept_name.strip().capitalize()
        features = []
        try:
            # Query for outgoing and incoming relationships
            # Ensure relationship types are from our schema for consistency in feature description.
            allowed_types_list = list(config.ALLOWED_RELATIONSHIP_TYPES)
            # Corrected query to fetch properties and full relationship details
            # This version also fetches the Concept properties for source, target, and the central concept.
            # For the streamlit_app, we primarily need relationship type and target concept name.
            # The `get_concept_context` in feature_extractor.py is more comprehensive for blending.
            # This function, as called by streamlit_app for "Browse & View", needs to list relationships.
            
            # Query for current concept's properties
            concept_props_query = "MATCH (c:Concept {name: $concept_name}) RETURN properties(c) as props"
            concept_props_result = graph.run(concept_props_query, concept_name=norm_concept_name).data()
            
            concept_data = {"name": norm_concept_name, "properties": {}, "relationships": []}
            if concept_props_result and concept_props_result[0]['props']:
                concept_data["properties"] = dict(concept_props_result[0]['props'])


            # Query for relationships (outgoing and incoming)
            # We want to show type and target for outgoing, type and source for incoming
            # For the UI, it's simpler to just show the connection from the perspective of the current concept.
            
            outgoing_query = f"""
            MATCH (c:Concept {{name: $concept_name}})-[r]->(neighbor:Concept)
            WHERE type(r) IN $allowed_types
            RETURN type(r) AS relationship_type, neighbor.name AS target_concept, properties(r) as rel_props
            LIMIT $limit
            """
            outgoing_rels = graph.run(outgoing_query, 
                                     concept_name=norm_concept_name, 
                                     limit=max_features,
                                     allowed_types=allowed_types_list).data()
            
            for rel in outgoing_rels:
                concept_data["relationships"].append({
                    "type": rel["relationship_type"],
                    "direction": "outgoing",
                    "target_concept": rel["target_concept"],
                    "properties": dict(rel['rel_props']) if rel['rel_props'] else {}
                })

            incoming_query = f"""
            MATCH (neighbor:Concept)-[r]->(c:Concept {{name: $concept_name}})
            WHERE type(r) IN $allowed_types
            RETURN type(r) AS relationship_type, neighbor.name AS source_concept, properties(r) as rel_props
            LIMIT $limit
            """
            incoming_rels = graph.run(incoming_query,
                                     concept_name=norm_concept_name,
                                     limit=max_features,
                                     allowed_types=allowed_types_list).data()

            for rel in incoming_rels:
                concept_data["relationships"].append({
                    "type": rel["relationship_type"],
                    "direction": "incoming",
                    "source_concept": rel["source_concept"], # Source from perspective of relationship
                    "target_concept": norm_concept_name, # Target is the current concept
                    "properties": dict(rel['rel_props']) if rel['rel_props'] else {}
                })
            
            # Trim if total relationships > max_features (though limit is applied per query part)
            # For the UI, it might be better to show up to max_features outgoing and max_features incoming
            # The current structure of streamlit_app expects a flatter list of relationships for the selected concept.
            # The previous get_concept_features was returning a list of triples, this now returns a dict with properties and relationships.
            # Let's adjust the Streamlit app to handle this new structure.
            # For now, this method returns the richer concept_data structure.

            logger.info(f"Retrieved details for concept '{norm_concept_name}'. Properties: {len(concept_data['properties'])}, Relationships: {len(concept_data['relationships'])}.")
            return concept_data # Return the structured data

        except Exception as e:
            logger.error(f"Error retrieving details for '{norm_concept_name}' from Neo4j: {e}")
            # Return structure expected on error or if not found, with empty values
            return {"name": norm_concept_name, "properties": {}, "relationships": []}

    def run_cypher_query(self, query: str, parameters: Optional[Dict] = None) -> List[Dict]:
        """
        Executes a given Cypher query with parameters and returns the results.

        Args:
            query (str): The Cypher query string.
            parameters (Optional[Dict]): A dictionary of parameters for the query.

        Returns:
            List[Dict]: A list of result records (dictionaries), or an empty list on error or no results.
        """
        graph = self._get_graph()
        if not graph:
            logger.error("Neo4j connection not available. Cannot execute query.")
            return []
        try:
            if parameters is None:
                parameters = {}
            results = graph.run(query, parameters).data()
            logger.debug(f"Successfully executed Cypher query. Query: {query[:100]}... Params: {parameters}. Results: {len(results)} records.")
            return results
        except Exception as e:
            logger.error(f"Error executing Cypher query: {e}. Query: {query[:100]}... Params: {parameters}")
            return []

# Keep the test routines, but they will need to be updated to instantiate GraphHandler
if __name__ == '__main__':
    logger.info("Running graph_handler.py tests...")
    try:
        # Instantiate GraphHandler using config (ensure config.py is accessible)
        gh = GraphHandler(uri=config.NEO4J_URI, user=config.NEO4J_USER, password=config.NEO4J_PASSWORD)
        logger.info("GraphHandler instantiated successfully for tests.")
        
        # Test add_concept
        gh.add_concept("TestConcept1", "TestDomain", {"description": "A concept for testing"})
        gh.add_concept("testconcept2", "testdomain") 
        gh.add_concept("TestConcept1", "UpdatedTestDomain", {"new_prop": "value"})

        # Test add_relationship
        gh.add_relationship("TestConcept1", "IS_A", "TestConcept2", {"confidence": 0.9})
        gh.add_relationship("TestConcept2", "RELATED_TO", "TestConcept1")
        gh.add_relationship("TestConcept1", "HAS_NON_CANONICAL_LINK", "TestConcept2")

        # Test get_all_concept_names
        all_concepts = gh.get_all_concept_names()
        logger.info(f"All concepts in DB: {all_concepts}")
        assert "Testconcept1" in all_concepts 
        assert "Testconcept2" in all_concepts

        # Test get_concept_features (now returns a dict)
        details_tc1 = gh.get_concept_features("TestConcept1", max_features=5)
        logger.info(f"Details for TestConcept1: {details_tc1}")
        assert details_tc1['name'] == "Testconcept1"
        assert "description" in details_tc1['properties']
        assert len(details_tc1['relationships']) > 0


        details_tc2 = gh.get_concept_features("TestConcept2")
        logger.info(f"Details for TestConcept2: {details_tc2}")
        assert details_tc2['name'] == "Testconcept2"
        assert len(details_tc2['relationships']) > 0

        logger.info("graph_handler.py tests completed successfully.")
    except Exception as e:
        logger.error(f"Neo4j connection or test failed: {e}")
        logger.error("Please ensure Neo4j is running and config.py has correct credentials.") 