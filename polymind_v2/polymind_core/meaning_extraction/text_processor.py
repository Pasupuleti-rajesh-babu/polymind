import google.generativeai as genai
from typing import List, Tuple, Dict, Optional
import logging
import re # For more robust parsing if needed

# Attempt to import configuration
try:
    from ... import config
except ImportError:
    import config

# Configure logging
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

class TextProcessor:
    def __init__(self, api_key: str, model_name: Optional[str] = None):
        """
        Initializes the TextProcessor and configures the Gemini model.

        Args:
            api_key (str): The Gemini API key.
            model_name (Optional[str]): The specific Gemini model to use. 
                                      Defaults to config.DEFAULT_GEMINI_MODEL if None.
        """
        self.model_name = model_name if model_name else config.DEFAULT_GEMINI_MODEL
        self._gemini_model = None
        if not api_key or api_key == "YOUR_GEMINI_API_KEY_HERE":
            logger.warning("Gemini API Key not provided or is a placeholder. Gemini functionality will be unavailable.")
            return
        try:
            genai.configure(api_key=api_key)
            self._gemini_model = genai.GenerativeModel(self.model_name)
            logger.info(f"Gemini model '{self.model_name}' initialized successfully.")
        except Exception as e:
            logger.error(f"Error configuring Gemini API or initializing model '{self.model_name}': {e}")
            self._gemini_model = None
            # Potentially raise this or handle it so the app knows init failed

    def _get_gemini_model(self):
        """Returns the initialized Gemini model instance."""
        if self._gemini_model is None:
            logger.error("Gemini model was not initialized successfully. Cannot perform operation.")
        return self._gemini_model

    def _canonicalize_relationship(self, raw_rel_type: str) -> str:
        """Canonicalizes a raw relationship type using the mapping and allowed types from config."""
        upper_raw_rel = raw_rel_type.upper().replace(" ", "_").replace("-", "_").strip()
        
        if upper_raw_rel in config.CANONICAL_REL_MAP:
            return config.CANONICAL_REL_MAP[upper_raw_rel]
        
        if upper_raw_rel in config.ALLOWED_RELATIONSHIP_TYPES:
            return upper_raw_rel
            
        logger.warning(f"Raw relationship type '{raw_rel_type}' (normalized: '{upper_raw_rel}') not found in CANONICAL_REL_MAP or ALLOWED_RELATIONSHIP_TYPES. Defaulting to RELATED_TO.")
        return "RELATED_TO"

    def extract_meaning_atoms_from_text(self, text_content: str) -> Tuple[List[str], List[Tuple[str, str, str]], str]:
        """
        Extracts concepts and relationships (triples) from text using the Gemini API.
        Normalizes concepts and canonicalizes relationships.

        Args:
            text_content (str): The text to process.

        Returns:
            Tuple[List[str], List[Tuple[str, str, str]], str]: 
                - A list of unique normalized concept names.
                - A list of (normalized_subject, canonical_relationship, normalized_object) triples.
                - The raw text output from Gemini.
        """
        gemini_model = self._get_gemini_model()
        if not gemini_model:
            logger.error("Gemini model not available. Cannot extract meaning atoms.")
            return [], [], ""

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
            logger.info(f"Sending text to Gemini for meaning atom extraction (length: {len(text_content)} chars)")
            response = gemini_model.generate_content(prompt)
            raw_gemini_output = response.text
            logger.debug(f"Gemini raw output for atom extraction: {raw_gemini_output}")

            extracted_concepts_set = set()
            extracted_triples_list = []

            for line in raw_gemini_output.split('\n'):
                line = line.strip()
                if line.startswith("(") and line.endswith(")") and line.count(";") == 2:
                    try:
                        content = line[1:-1]
                        parts = [p.strip() for p in content.split(';')]
                        if len(parts) == 3:
                            subject, relationship_raw, object_raw = parts[0], parts[1], parts[2]
                            
                            if subject and relationship_raw and object_raw:
                                norm_subject = subject.strip().capitalize()
                                norm_object = object_raw.strip().capitalize()
                                canonical_rel = self._canonicalize_relationship(relationship_raw)
                                
                                if norm_subject and canonical_rel and norm_object:
                                    extracted_triples_list.append((norm_subject, canonical_rel, norm_object))
                                    extracted_concepts_set.add(norm_subject)
                                    extracted_concepts_set.add(norm_object)
                                else:
                                    logger.warning(f"Skipping triple due to empty parts after normalization/canonicalization: Original ({subject}, {relationship_raw}, {object_raw})")
                        else:
                            logger.warning(f"Skipping malformed triple (not 3 parts after split): {line}")
                    except Exception as e:
                        logger.error(f"Error parsing triple line '{line}': {e}")
                elif line:
                    logger.debug(f"Skipping non-triple line from Gemini atom extraction: {line}")
            
            final_concepts = sorted(list(extracted_concepts_set))
            logger.info(f"Gemini extracted {len(final_concepts)} concepts and {len(extracted_triples_list)} triples.")
            return final_concepts, extracted_triples_list, raw_gemini_output

        except Exception as e:
            logger.error(f"Error calling Gemini API or parsing response for atom extraction: {e}")
            return [], [], f"Error during Gemini processing: {e}"

    def suggest_domains_for_concepts(self, concepts: List[str]) -> Dict[str, str]:
        """
        Suggests a primary domain for each concept in a list using the Gemini API.

        Args:
            concepts (List[str]): A list of concept names.

        Returns:
            Dict[str, str]: A dictionary mapping concept_name to suggested_domain.
        """
        gemini_model = self._get_gemini_model()
        if not gemini_model or not concepts:
            if not concepts:
                return {}
            logger.error("Gemini model not available or no concepts provided. Cannot suggest domains.")
            return {concept: config.DEFAULT_DOMAIN for concept in concepts}

        concept_bullet_list = "\n".join([f"- {concept}" for concept in concepts])
        prompt = f"""For each concept in the following list, suggest a single, primary academic or general domain it belongs to.
Output each concept and its domain on a new line, strictly in the format:
Concept: Domain

Example:
DNA: Biology
Metaphysics: Philosophy
Algorithm: Computer Science

Concepts to categorize:
{concept_bullet_list}

Categorized Concepts:
"""
        try:
            logger.info(f"Sending {len(concepts)} concepts to Gemini for domain suggestion.")
            response = gemini_model.generate_content(prompt)
            response_text = response.text
            logger.debug(f"Gemini domain suggestion raw output: {response_text}")

            domain_map = {}
            for line in response_text.split('\n'):
                line = line.strip()
                if ": " in line:
                    parts = line.split(": ", 1)
                    if len(parts) == 2:
                        concept_name_from_gemini = parts[0].strip()
                        suggested_domain = parts[1].strip().capitalize()
                        matched_original_concept = None
                        for original_concept in concepts:
                            if original_concept.lower() == concept_name_from_gemini.lower():
                                matched_original_concept = original_concept
                                break
                        
                        if matched_original_concept:
                            if matched_original_concept not in domain_map: 
                               domain_map[matched_original_concept] = suggested_domain if suggested_domain else config.DEFAULT_DOMAIN
                            else:
                                logger.debug(f"Gemini provided multiple domain suggestions for '{matched_original_concept}'. Using first one: '{domain_map[matched_original_concept]}'") 
                        else:
                            logger.warning(f"Gemini suggested domain for concept '{concept_name_from_gemini}' not in original list or mismatch. Input concepts: {concepts}. Skipping.")
                    else:
                        logger.warning(f"Skipping malformed domain line (not 2 parts after split): {line}")
                elif line:
                    logger.debug(f"Skipping non-domain line from Gemini domain suggestion: {line}")
            
            for concept in concepts:
                if concept not in domain_map:
                    logger.info(f"Gemini did not provide domain for '{concept}'. Defaulting to '{config.DEFAULT_DOMAIN}'.")
                    domain_map[concept] = config.DEFAULT_DOMAIN
            
            logger.info(f"Gemini suggested domains: {domain_map}")
            return domain_map

        except Exception as e:
            logger.error(f"Error calling Gemini API or parsing domain response: {e}")
            return {concept: config.DEFAULT_DOMAIN for concept in concepts}

if __name__ == '__main__':
    logger.info("Running text_processor.py direct tests...")
    if not config.GEMINI_API_KEY or config.GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        logger.error("GEMINI_API_KEY not configured in config.py. Skipping direct tests for text_processor.")
    else:
        tp = TextProcessor(api_key=config.GEMINI_API_KEY)
        if tp._get_gemini_model(): # Check if model initialized correctly
            sample_text = ("The quick brown fox jumps over the lazy dog. The dog, named Max, is a_type_of canine. "
                           "Foxes are part of the Canidae family. Jumping causes movement.")
            
            logger.info(f"\n--- Test: extract_meaning_atoms_from_text ---")
            concepts, triples, raw_out = tp.extract_meaning_atoms_from_text(sample_text)
            logger.info(f"Extracted Concepts: {concepts}")
            logger.info(f"Extracted Triples: {triples}")
            assert isinstance(concepts, list)
            assert isinstance(triples, list)
            assert isinstance(raw_out, str)
            if triples: 
                assert len(triples[0]) == 3
                assert "RELATED_TO" in config.ALLOWED_RELATIONSHIP_TYPES

            if concepts:
                logger.info(f"\n--- Test: suggest_domains_for_concepts ---")
                suggested_domains = tp.suggest_domains_for_concepts(concepts[:3])
                logger.info(f"Suggested Domains: {suggested_domains}")
                assert isinstance(suggested_domains, dict)
                if concepts[:3]:
                    assert len(suggested_domains) == len(concepts[:3])
                    for concept_name in concepts[:3]:
                        assert concept_name in suggested_domains
            else:
                logger.warning("No concepts extracted, skipping domain suggestion test.")
            logger.info("\ntext_processor.py direct tests completed.")
        else:
            logger.error("Gemini model failed to initialize for TextProcessor. Tests aborted.") 