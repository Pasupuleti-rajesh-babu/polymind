import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import pickle
import logging
from typing import List, Tuple, Optional

# Attempt to import configuration
try:
    from ... import config
except ImportError:
    import config

# Configure logging
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

class FaissHandler:
    def __init__(self, faiss_index_path: str, concept_list_path: Optional[str] = None, embedding_model_name: Optional[str] = None):
        """
        Initializes the FaissHandler, loads or creates the FAISS index and concept list,
        and loads the sentence transformer model.

        Args:
            faiss_index_path (str): Path to the FAISS index file.
            concept_list_path (Optional[str]): Path to the concept list pickle file. 
                                             Defaults to faiss_index_path + '_concepts.pkl'.
            embedding_model_name (Optional[str]): Name of the SentenceTransformer model.
                                                Defaults to config.DEFAULT_EMBEDDING_MODEL.
        """
        self.faiss_index_path = faiss_index_path
        self.concept_list_path = concept_list_path if concept_list_path else faiss_index_path + "_concepts.pkl"
        self.embedding_model_name = embedding_model_name if embedding_model_name else config.DEFAULT_EMBEDDING_MODEL
        
        self._embedding_model: Optional[SentenceTransformer] = None
        self._faiss_index: Optional[faiss.Index] = None
        self._concept_list: List[str] = []

        self._load_embedding_model()
        self.load_index(force_reload=False) # Load or initialize index

    def _load_embedding_model(self) -> None:
        """Loads the sentence transformer model."""
        if self._embedding_model is None:
            try:
                logger.info(f"Loading sentence transformer model: {self.embedding_model_name}")
                self._embedding_model = SentenceTransformer(self.embedding_model_name)
                logger.info("Sentence transformer model loaded successfully.")
            except Exception as e:
                logger.error(f"Error loading sentence transformer model '{self.embedding_model_name}': {e}")
                self._embedding_model = None # Ensure it's reset
                # Consider raising an error if model loading is critical for initialization

    def get_embedding_model(self) -> Optional[SentenceTransformer]:
        """Returns the loaded sentence transformer model."""
        if self._embedding_model is None:
            logger.error("Embedding model is not loaded.")
        return self._embedding_model

    def load_index(self, force_reload: bool = False) -> None:
        """
        Loads the FAISS index and concept list from disk.
        If files don't exist, initializes an empty index and list.
        """
        if self._faiss_index is not None and self._concept_list and not force_reload:
            logger.debug("FAISS index and concept list already in memory.")
            return

        index_loaded = False
        if os.path.exists(self.faiss_index_path) and os.path.exists(self.concept_list_path):
            try:
                logger.info(f"Loading FAISS index from: {self.faiss_index_path}")
                self._faiss_index = faiss.read_index(self.faiss_index_path)
                logger.info(f"Loading concept list from: {self.concept_list_path}")
                with open(self.concept_list_path, "rb") as f:
                    self._concept_list = pickle.load(f)
                logger.info(f"FAISS index ({self._faiss_index.ntotal} vectors) and concept list ({len(self._concept_list)} concepts) loaded.")
                if self._faiss_index.ntotal != len(self._concept_list):
                    logger.warning("Mismatch between FAISS index size and concept list length. Re-synchronization might be needed.")
                index_loaded = True
            except Exception as e:
                logger.error(f"Error loading FAISS index or concept list: {e}. Initializing empty.")
                self._faiss_index = None 
                self._concept_list = []
        else:
            logger.info("FAISS index or concept list file not found. Initializing empty.")

        if not index_loaded or self._faiss_index is None:
            model = self.get_embedding_model()
            if model:
                try:
                    dummy_embedding = model.encode(["test"])
                    embedding_dim = dummy_embedding.shape[1]
                    self._faiss_index = faiss.IndexFlatL2(embedding_dim)
                    logger.info(f"Initialized new empty FAISS IndexFlatL2 with dimension {embedding_dim}.")
                except Exception as e:
                    logger.error(f"Could not determine embedding dimension or initialize FAISS index: {e}")
                    self._faiss_index = None
            else:
                logger.error("Embedding model not available. Cannot initialize FAISS index.")
            self._concept_list = []
        
    def save_index(self) -> bool:
        """Saves the current FAISS index and concept list to disk."""
        if self._faiss_index is None:
            logger.warning("FAISS index is not initialized. Cannot save.")
            return False
        try:
            os.makedirs(os.path.dirname(self.faiss_index_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.concept_list_path), exist_ok=True)
            
            logger.info(f"Saving FAISS index to: {self.faiss_index_path}")
            faiss.write_index(self._faiss_index, self.faiss_index_path)
            logger.info(f"Saving concept list to: {self.concept_list_path}")
            with open(self.concept_list_path, "wb") as f:
                pickle.dump(self._concept_list, f)
            logger.info("FAISS index and concept list saved successfully.")
            return True
        except Exception as e:
            logger.error(f"Error saving FAISS index or concept list: {e}")
            return False

    def create_concept_embeddings(self, concepts: List[str]) -> Optional[np.ndarray]:
        """Generates embeddings for a list of concept strings."""
        model = self.get_embedding_model()
        if not model:
            logger.error("Embedding model not available. Cannot create embeddings.")
            return None
        if not concepts:
            logger.info("Empty list of concepts provided for embedding.")
            return np.array([]) 
        
        try:
            logger.info(f"Creating embeddings for {len(concepts)} concepts.")
            embeddings = model.encode(concepts, convert_to_numpy=True)
            return embeddings.astype('float32')
        except Exception as e:
            logger.error(f"Error creating concept embeddings: {e}")
            return None

    def add_concepts_to_faiss(self, concepts_to_add: List[str], embeddings_to_add: Optional[np.ndarray] = None) -> bool:
        """
        Adds new concepts and their embeddings to the FAISS index and concept list.
        Avoids adding duplicate concepts. Saves the index after adding.
        """
        if self._faiss_index is None:
            logger.error("FAISS index not initialized. Cannot add concepts. Try load_index() first.")
            return False
        if not concepts_to_add:
            logger.info("No concepts provided to add to FAISS.")
            return True

        current_indexed_concepts_set = set(self._concept_list)
        unique_concepts_to_process = []
        for concept in concepts_to_add:
            if concept not in current_indexed_concepts_set:
                unique_concepts_to_process.append(concept)
                current_indexed_concepts_set.add(concept)
            else:
                logger.debug(f"Concept '{concept}' already in FAISS index. Skipping.")

        if not unique_concepts_to_process:
            logger.info("All provided concepts are already in the FAISS index.")
            return True

        if embeddings_to_add is None:
            embeddings_for_new = self.create_concept_embeddings(unique_concepts_to_process)
        else:
            if embeddings_to_add.shape[0] == len(unique_concepts_to_process):
                embeddings_for_new = embeddings_to_add.astype('float32')
            else:
                logger.warning(f"Mismatch in length or format of provided embeddings. Regenerating for {len(unique_concepts_to_process)} unique concepts.")
                embeddings_for_new = self.create_concept_embeddings(unique_concepts_to_process)

        if embeddings_for_new is None or embeddings_for_new.shape[0] == 0:
            logger.error("No valid embeddings generated or provided for new concepts. Cannot add to FAISS.")
            return False

        try:
            self._faiss_index.add(embeddings_for_new)
            self._concept_list.extend(unique_concepts_to_process)
            logger.info(f"Added {len(unique_concepts_to_process)} new concepts to FAISS index. Total concepts: {len(self._concept_list)}.")
            return self.save_index()
        except Exception as e:
            logger.error(f"Error adding embeddings to FAISS index: {e}")
            return False

    def find_similar_concepts(self, query_concept: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Finds k most similar concepts to the query_concept using the FAISS index.
        Returns a list of (concept_name, distance_score) tuples.
        """
        if self._faiss_index is None or self._faiss_index.ntotal == 0:
            logger.warning("FAISS index is not initialized or is empty. Cannot find similar concepts.")
            return []
        
        query_embedding = self.create_concept_embeddings([query_concept])
        if query_embedding is None or query_embedding.shape[0] == 0:
            logger.error(f"Could not create embedding for query concept: '{query_concept}'")
            return []

        try:
            actual_k = min(k, self._faiss_index.ntotal)
            if actual_k == 0:
                 logger.warning("FAISS index is empty, cannot perform search.")
                 return []
            if actual_k < k:
                logger.info(f"Requested k={k} similar concepts, but index only contains {actual_k}. Searching for {actual_k}.")

            logger.info(f"Searching for {actual_k} concepts similar to '{query_concept}'")
            distances, indices = self._faiss_index.search(query_embedding, actual_k)
            
            results = []
            for i in range(indices.shape[1]):
                idx = indices[0, i]
                dist = distances[0, i]
                if 0 <= idx < len(self._concept_list):
                    results.append((self._concept_list[idx], float(dist)))
                else:
                    logger.warning(f"FAISS returned index {idx} out of bounds for concept_list (len: {len(self._concept_list)})")
            
            logger.info(f"Found similar concepts: {results}")
            return results
        except Exception as e:
            logger.error(f"Error searching FAISS index for '{query_concept}': {e}")
            return []

if __name__ == '__main__':
    logger.info("Running faiss_handler.py direct tests...")
    
    # Use specific test paths to avoid overwriting production/dev data if config points there
    test_faiss_path = "./test_data/faiss_index_test.faiss"
    test_concepts_path = "./test_data/faiss_concepts_test.pkl"
    os.makedirs("./test_data", exist_ok=True)

    # Clean up old test files before starting
    if os.path.exists(test_faiss_path): os.remove(test_faiss_path)
    if os.path.exists(test_concepts_path): os.remove(test_concepts_path)

    fh = FaissHandler(faiss_index_path=test_faiss_path, concept_list_path=test_concepts_path, embedding_model_name=config.DEFAULT_EMBEDDING_MODEL)
    logger.info(f"Initial index size: {fh._faiss_index.ntotal if fh._faiss_index else 'None'}, concepts: {len(fh._concept_list)}")

    concepts1 = ["Artificial Intelligence", "Machine Learning", "Deep Learning", "Neural Networks"]
    logger.info(f"\n--- Test: Adding first batch of concepts ---")
    success_add1 = fh.add_concepts_to_faiss(concepts1)
    assert success_add1
    assert fh._faiss_index is not None and fh._faiss_index.ntotal == len(concepts1)
    assert len(fh._concept_list) == len(concepts1)
    logger.info(f"Index size after batch 1: {fh._faiss_index.ntotal}, concepts: {len(fh._concept_list)}")

    concepts2 = ["Machine Learning", "Natural Language Processing", "Computer Vision", "Robotics"]
    expected_new_in_batch2 = ["Natural Language Processing", "Computer Vision", "Robotics"] # "Machine Learning" is a duplicate
    logger.info(f"\n--- Test: Adding second batch of concepts (with duplicates) ---")
    success_add2 = fh.add_concepts_to_faiss(concepts2)
    assert success_add2
    assert fh._faiss_index.ntotal == len(concepts1) + len(expected_new_in_batch2)
    assert len(fh._concept_list) == len(concepts1) + len(expected_new_in_batch2)
    logger.info(f"Index size after batch 2: {fh._faiss_index.ntotal}, concepts: {len(fh._concept_list)}")

    logger.info(f"\n--- Test: Finding similar concepts ---")
    if fh._faiss_index.ntotal > 0:
        similar = fh.find_similar_concepts("AI", k=3)
        logger.info(f"Concepts similar to 'AI': {similar}")
        assert len(similar) <= 3
        if similar:
             assert isinstance(similar[0], tuple) and len(similar[0]) == 2
             assert isinstance(similar[0][0], str) and isinstance(similar[0][1], float)

        similar_ml = fh.find_similar_concepts("Machine Learning", k=2)
        logger.info(f"Concepts similar to 'Machine Learning': {similar_ml}")
        # Expect "Machine Learning" itself to be the top hit if present
        if similar_ml:
            assert similar_ml[0][0] == "Machine Learning"
    else:
        logger.warning("Skipping similarity search test as index is empty.")

    # Test saving and loading works by re-instantiating
    logger.info(f"\n--- Test: Saving and Re-loading Index ---")
    del fh # Delete current instance
    fh_reloaded = FaissHandler(faiss_index_path=test_faiss_path, concept_list_path=test_concepts_path)
    assert fh_reloaded._faiss_index is not None
    assert fh_reloaded._faiss_index.ntotal == len(concepts1) + len(expected_new_in_batch2)
    assert len(fh_reloaded._concept_list) == len(concepts1) + len(expected_new_in_batch2)
    logger.info(f"Reloaded index size: {fh_reloaded._faiss_index.ntotal}, concepts: {len(fh_reloaded._concept_list)}")
    logger.info("FAISS handler tests completed successfully.")

    # Clean up test files after tests
    if os.path.exists(test_faiss_path): os.remove(test_faiss_path)
    if os.path.exists(test_concepts_path): os.remove(test_concepts_path)
    if os.path.exists("./test_data") and not os.listdir("./test_data"):
        os.rmdir("./test_data") 