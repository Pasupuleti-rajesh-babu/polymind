import re
import json # For parsing LLM response
from typing import Any, Dict, Optional
import logging

# Assuming core modules are accessible via polymind_core
from polymind_core.knowledge_graph.graph_handler import GraphHandler
from polymind_core.vector_store.faiss_handler import FaissHandler
from polymind_core.meaning_extraction.text_processor import TextProcessor
from polymind_core.synthesis_engine.feature_extractor import get_concept_context
from polymind_core.synthesis_engine.blender import blend_concepts_gemini
from polymind_core.synthesis_engine.analogy_finder import find_analogous_concepts_vector

logger = logging.getLogger(__name__)

class ChatOrchestrator:
    def __init__(self, graph_handler: GraphHandler, faiss_handler: FaissHandler, text_processor: TextProcessor, 
                 feature_extractor: Any, blender: Any, analogy_finder: Any):
        """
        Initializes the ChatOrchestrator with handles to PolyMind's core components.
        The feature_extractor, blender, and analogy_finder are expected to be the modules themselves
        so we can call their functions (e.g., feature_extractor.get_concept_context)
        """
        self.graph_handler = graph_handler
        self.faiss_handler = faiss_handler
        self.text_processor = text_processor
        self.feature_extractor = feature_extractor 
        self.blender = blender                   
        self.analogy_finder = analogy_finder     
        
        self.patterns = {
            "get_details": re.compile(r"^(?:what is|tell me about|details for)\s+(?:\"([^\"]+)\"|([A-Za-z0-9 _-]+(?:\s+[A-Za-z0-9 _-]+)*))\s*[?.!]?$", re.IGNORECASE),
            "blend": re.compile(r"^(?:blend|combine)\s+(?:\"([^\"]+)\"|([A-Za-z0-9 _-]+(?:\s+[A-Za-z0-9 _-]+)*))\s+and\s+(?:\"([^\"]+)\"|([A-Za-z0-9 _-]+(?:\s+[A-Za-z0-9 _-]+)*))\s*[?.!]?$", re.IGNORECASE),
            "analogies": re.compile(r"^(?:find analogies for|what is similar to)\s+(?:\"([^\"]+)\"|([A-Za-z0-9 _-]+(?:\s+[A-Za-z0-9 _-]+)*))\s*[?.!]?$", re.IGNORECASE),
        }

    def _strip_punctuation(self, concept_name: Optional[str]) -> Optional[str]:
        if concept_name:
            return concept_name.rstrip("?.!")
        return None

    def _get_concept_from_match(self, match, group_indices):
        for index in group_indices:
            if match and match.group(index):
                return self._strip_punctuation(match.group(index).strip())
        return None

    def _get_intent_and_entities_llm(self, user_query: str) -> Optional[Dict[str, Any]]:
        if not self.text_processor or not self.text_processor._get_gemini_model():
            logger.warning("Text processor or Gemini model not available for LLM intent extraction.")
            return None
        gemini_model = self.text_processor._get_gemini_model()
        prompt = f"""Analyze the following user query to determine the intent and extract relevant entities.
Possible intents are: GET_DETAILS, BLEND_CONCEPTS, FIND_ANALOGIES, UNKNOWN.

For GET_DETAILS, extract 'concept_name'.
For BLEND_CONCEPTS, extract 'concept_a_name' and 'concept_b_name'.
For FIND_ANALOGIES, extract 'concept_name'.

Output ONLY a JSON object with "intent" and "entities". Entities should be a nested object.
If no specific concept names can be extracted but the intent seems clear, provide the intent and empty entities.
If the intent is unclear, use UNKNOWN and empty entities.

Examples:
Query: "Tell me about Artificial Intelligence."
Output: {{"intent": "GET_DETAILS", "entities": {{"concept_name": "Artificial Intelligence"}}}}
Query: "Blend DNA and the concept of a market economy"
Output: {{"intent": "BLEND_CONCEPTS", "entities": {{"concept_a_name": "DNA", "concept_b_name": "market economy"}}}}
Query: "What is similar to a Cat?"
Output: {{"intent": "FIND_ANALOGIES", "entities": {{"concept_name": "Cat"}}}}
Query: "Hello there!"
Output: {{"intent": "UNKNOWN", "entities": {{}}}}
Query: "Elaborate on photosynthesis."
Output: {{"intent": "GET_DETAILS", "entities": {{"concept_name": "photosynthesis"}}}}

User Query:
{user_query}
Output:
""" 
        raw_llm_output = ""
        json_str = ""
        try:
            logger.debug(f"Sending to LLM for intent/entity extraction: {user_query}")
            response = gemini_model.generate_content(prompt)
            raw_llm_output = response.text
            logger.debug(f"Raw LLM output for intent/entity: {raw_llm_output}")
            match_json = re.search(r"```json\n(.*?)\n```", raw_llm_output, re.DOTALL)
            json_str = match_json.group(1) if match_json else raw_llm_output.strip()
            parsed_response = json.loads(json_str)
            if "intent" in parsed_response and "entities" in parsed_response:
                if "concept_name" in parsed_response["entities"]:
                    parsed_response["entities"]["concept_name"] = self._strip_punctuation(parsed_response["entities"]["concept_name"])
                if "concept_a_name" in parsed_response["entities"]:
                    parsed_response["entities"]["concept_a_name"] = self._strip_punctuation(parsed_response["entities"]["concept_a_name"])
                if "concept_b_name" in parsed_response["entities"]:
                    parsed_response["entities"]["concept_b_name"] = self._strip_punctuation(parsed_response["entities"]["concept_b_name"])
                logger.info(f"LLM parsed intent: {parsed_response['intent']}, (cleaned) entities: {parsed_response['entities']}")
                return parsed_response
            else:
                logger.warning(f"LLM response missing intent or entities: {parsed_response}")
                return None
        except json.JSONDecodeError as e:
            logger.error(f"JSONDecodeError parsing LLM response for intent/entity. String was: '{json_str}'. Error: {e}")
            return None
        except Exception as e:
            logger.error(f"Error during LLM intent/entity extraction: {e}\nRaw LLM Output was: {raw_llm_output}")
            return None

    def _generate_llm_summary_response(self, intent: str, data: Dict[str, Any], user_query: str, concept_name_for_prompt: Optional[str] = None) -> Optional[str]:
        if not self.text_processor or not self.text_processor._get_gemini_model():
            logger.warning("Text processor or Gemini model not available for LLM response generation.")
            return None
        gemini_model = self.text_processor._get_gemini_model()
        
        prompt = ""
        # Ensure concept_name_for_prompt is clean for use in summary prompts
        # Also, ensure a fallback for clean_concept_name if all are None/empty (use user_query as last resort)
        _cname_for_prompt = self._strip_punctuation(concept_name_for_prompt)
        _data_name = data.get('name') if isinstance(data, dict) else None
        _data_q_concept_name = data.get('query_concept_name') if isinstance(data, dict) else None
        clean_concept_name = _cname_for_prompt or _data_name or _data_q_concept_name or user_query

        if intent == "GET_DETAILS":
            if not data or not (data.get('properties') or data.get('relationships')):
                prompt = f"You are PolyMind, a helpful AI assistant. The user asked: '{user_query}'. Politely inform the user that you couldn't find detailed information for '{clean_concept_name}'."
            else:
                properties_str = json.dumps(data.get('properties', {}))
                relationships_str = json.dumps(data.get('relationships', []))
                prompt = f"""You are PolyMind, a helpful AI assistant. The user asked: '{user_query}'.
Based on the following structured information about the concept '{data.get("name", clean_concept_name)}', please provide a concise and conversational summary.

Properties: {properties_str}
Relationships: {relationships_str}

Synthesize this into a natural language paragraph. Mention key properties and a couple of important relationships. If there are no properties or relationships, just say so.
"""
        elif intent == "BLEND_CONCEPTS":
            # For blend, clean_concept_name might be concept_a if concept_b was also passed for the prompt key in mocks
            # Let's use user_query for context if specific concept names are missing in the prompt generation logic
            context_name = clean_concept_name # Default to the already derived clean_concept_name or user_query
            if data and data.get("concept_a_name") and data.get("concept_b_name"):
                 context_name = f"{data.get('concept_a_name')} and {data.get('concept_b_name')}"
            elif data and data.get("concept_a_name"):
                 context_name = data.get("concept_a_name")

            if not data or not data.get('blended_name'):
                prompt = f"You are PolyMind, a helpful AI assistant. The user asked: '{user_query}'. Politely inform the user that you were unable to generate a blend for concepts related to '{context_name}'."
            else:
                prompt = f"""You are PolyMind, a helpful AI assistant. The user asked: '{user_query}'.
Based on the following blended concept information, please describe the result in a conversational and engaging way:
Name: {data.get('blended_name', 'Unnamed Blend')}
Description: {data.get('description', 'N/A')}
Combined Features: {json.dumps(data.get('combined_features', []))}
Emergent Properties: {json.dumps(data.get('emergent_properties', []))}

Synthesize this into a natural language paragraph.
"""
        elif intent == "FIND_ANALOGIES":
            if not data or not isinstance(data, list) or len(data) == 0:
                prompt = f"You are PolyMind, a helpful AI assistant. The user asked: '{user_query}'. Politely inform the user that you couldn't find any analogies for '{clean_concept_name}'."
            else:
                analogies_str = json.dumps(data)
                prompt = f"""You are PolyMind, a helpful AI assistant. The user asked: '{user_query}' (for analogies related to '{clean_concept_name}').
Based on the following list of analogies found, please present them to the user in a conversational way:
Analogies Found: {analogies_str}

List the analogies in a natural language sentence. For example: 'For {clean_concept_name}, I found a few potential analogies: Name1 and Name2.'
"""
        elif intent == "UNKNOWN_QUERY":
             prompt = f"You are PolyMind, a helpful AI assistant. The user said: '{data.get('original_query', user_query)}'. You don't understand this request. Politely ask the user to rephrase or try a different kind of query, suggesting some examples like asking about a concept (e.g., 'What is Apple?'), blending two concepts (e.g., 'Blend DNA and Computer'), or finding analogies (e.g., 'What is similar to a Cat?')."
        else:
            return None 

        try:
            logger.debug(f"Sending to LLM for response generation. Intent: {intent}. Query: {user_query}. Clean Concept for Prompt: {clean_concept_name}. Data (first 200 chars): {json.dumps(data)[:200]}...")
            response = gemini_model.generate_content(prompt)
            generated_text = response.text.strip()
            logger.info(f"LLM generated response for {intent}: {generated_text}")
            if not generated_text or "not sure how to respond" in generated_text.lower() or "cannot fulfill this request" in generated_text.lower() or "my mock data is limited" in generated_text.lower():
                logger.warning(f"LLM summary response was empty, a refusal, or a mock fallback: {generated_text}")
                return None
            return generated_text
        except Exception as e:
            logger.error(f"Error during LLM response generation for intent {intent}: {e}")
            return None

    def _format_structured_response(self, intent: str, data: Any, concept_name: Optional[str] = None, concept_a_name: Optional[str] = None, concept_b_name: Optional[str] = None) -> str:
        clean_concept_name = self._strip_punctuation(concept_name)
        clean_concept_a_name = self._strip_punctuation(concept_a_name)
        clean_concept_b_name = self._strip_punctuation(concept_b_name)

        if intent == "GET_DETAILS":
            if data and (data.get('properties') or data.get('relationships')):
                response_str = f"**Details for {data.get('name', clean_concept_name)}:**\n"
                if data.get('properties'):
                    response_str += "\n**Properties:**\n"
                    for k, v in data['properties'].items(): response_str += f"- {k.capitalize()}: {v}\n"
                if data.get('relationships'):
                    response_str += "\n**Relationships:**\n"
                    for rel in data['relationships'][:5]:
                        node_name = data.get('name', clean_concept_name)
                        if rel.get('direction') == 'outgoing': response_str += f"- `{node_name}` `-{rel['type']}->` `{rel.get('target_concept', 'Unknown')}`\n"
                        elif rel.get('direction') == 'incoming': response_str += f"- `{rel.get('source_concept', 'Unknown')}` `-{rel['type']}->` `{node_name}`\n"
                return response_str.strip()
            else: return f"Sorry, I couldn't find details for '{clean_concept_name}' or it has no properties/relationships."
        
        elif intent == "BLEND_CONCEPTS":
            if data and data.get('blended_name'):
                response_str = f"**Blended Concept: {data.get('blended_name', 'Unnamed Blend')}**\n"
                response_str += f"Description: {data.get('description', 'N/A')}\n"
                if data.get('combined_features'):
                    response_str += "\nKey Combined Features:\n"
                    for f in data['combined_features'][:3]: response_str += f"- {f}\n"
                if data.get('emergent_properties'):
                    response_str += "\nPotential Emergent Properties:\n"
                    for p in data['emergent_properties'][:3]: response_str += f"- {p}\n"
                return response_str.strip()
            else: return f"Sorry, I couldn't generate a blend for '{clean_concept_a_name}' and '{clean_concept_b_name}'."

        elif intent == "FIND_ANALOGIES":
            if data and isinstance(data, list) and len(data) > 0:
                response_str = f"**Analogies for {clean_concept_name}:**\n"
                for ana in data: response_str += f"- {ana['name']} (Score: {ana.get('score', 'N/A'):.2f})\n"
                return response_str.strip()
            else: return f"Sorry, I couldn't find any vector-based analogies for '{clean_concept_name}'."
        return "Error: Could not format structured response for unknown intent."

    def parse_and_execute_command(self, user_query: str) -> str:
        logger.info(f"ChatOrchestrator processing query: {user_query}")
        llm_result = self._get_intent_and_entities_llm(user_query)
        intent, entities = None, None

        if llm_result and llm_result.get("intent") != "UNKNOWN":
            intent = llm_result["intent"]
            entities = llm_result["entities"]
            logger.info(f"Using LLM result - Intent: {intent}, Entities: {entities}")
        else:
            logger.info("LLM did not provide a clear intent or failed/returned UNKNOWN. Falling back to regex.")
            for intent_type, pattern in self.patterns.items():
                match = pattern.match(user_query)
                if match:
                    intent = intent_type.upper()
                    if intent == "GET_DETAILS": entities = {"concept_name": self._get_concept_from_match(match, [1, 2])}
                    elif intent == "BLEND": 
                        intent = "BLEND_CONCEPTS"
                        entities = {"concept_a_name": self._get_concept_from_match(match, [1, 2]), "concept_b_name": self._get_concept_from_match(match, [3, 4])}
                    elif intent == "ANALOGIES": 
                        intent = "FIND_ANALOGIES"
                        entities = {"concept_name": self._get_concept_from_match(match, [1, 2])}
                    break
            valid_entities = False
            if entities:
                if intent == "GET_DETAILS" and entities.get("concept_name"):
                    valid_entities = True
                elif intent == "BLEND_CONCEPTS" and entities.get("concept_a_name") and entities.get("concept_b_name"):
                    valid_entities = True
                elif intent == "FIND_ANALOGIES" and entities.get("concept_name"):
                    valid_entities = True
            if not (intent and valid_entities):
                intent, entities = "UNKNOWN", {}
                logger.info(f"Regex fallback also failed to determine a clear action or extract entities for query: {user_query}")
            else:
                logger.info(f"Using Regex fallback - Intent: {intent}, Entities: {entities}")

        if intent == "GET_DETAILS":
            concept_name = entities.get("concept_name")
            if not concept_name: return "Sorry, I understood you want details, but I couldn't identify which concept."
            logger.info(f"Executing GET_DETAILS for clean concept: '{concept_name}'")
            if not self.graph_handler: logger.error("Graph handler not available"); return "Error: Knowledge Graph unavailable."
            try:
                details_data = self.graph_handler.get_concept_features(concept_name)
                llm_response = self._generate_llm_summary_response("GET_DETAILS", details_data or {}, user_query, concept_name_for_prompt=concept_name)
                return llm_response or self._format_structured_response("GET_DETAILS", details_data, concept_name)
            except Exception as e: 
                logger.error(f"Error in GET_DETAILS for '{concept_name}': {e}")
                return self._format_structured_response("GET_DETAILS", None, concept_name)

        elif intent == "BLEND_CONCEPTS":
            concept_a = entities.get("concept_a_name")
            concept_b = entities.get("concept_b_name")
            if not concept_a or not concept_b: return "Sorry, I need two concept names to blend."
            logger.info(f"Executing BLEND_CONCEPTS for clean concepts: '{concept_a}' and '{concept_b}'")
            if not (self.graph_handler and self.blender and self.feature_extractor): 
                logger.error("Core components unavailable for BLEND"); return "Error: Blending components unavailable."
            try:
                context_a = self.feature_extractor.get_concept_context(self.graph_handler, concept_a)
                context_b = self.feature_extractor.get_concept_context(self.graph_handler, concept_b)
                empty_blend_data = {"concept_a_name": concept_a, "concept_b_name": concept_b} 
                if not context_a or not context_b:                     
                    llm_response = self._generate_llm_summary_response("BLEND_CONCEPTS", empty_blend_data, user_query)
                    return llm_response or self._format_structured_response("BLEND_CONCEPTS", None, concept_a_name=concept_a, concept_b_name=concept_b)
                
                blend_result = self.blender.blend_concepts_gemini(context_a, context_b)
                data_for_summary = blend_result if blend_result else empty_blend_data
                llm_response = self._generate_llm_summary_response("BLEND_CONCEPTS", data_for_summary, user_query)
                return llm_response or self._format_structured_response("BLEND_CONCEPTS", blend_result, concept_a_name=concept_a, concept_b_name=concept_b)
            except Exception as e:
                logger.error(f"Error in BLEND_CONCEPTS for '{concept_a}' and '{concept_b}': {e}")
                return self._format_structured_response("BLEND_CONCEPTS", None, concept_a_name=concept_a, concept_b_name=concept_b)

        elif intent == "FIND_ANALOGIES":
            concept_name = entities.get("concept_name")
            if not concept_name: return "Sorry, I need a concept name to find analogies."
            logger.info(f"Executing FIND_ANALOGIES for clean concept: '{concept_name}'")
            if not (self.faiss_handler and self.graph_handler and self.analogy_finder):
                logger.error("Core components unavailable for ANALOGIES"); return "Error: Analogy components unavailable."
            try:
                analogies_data = self.analogy_finder.find_analogous_concepts_vector(concept_name, self.faiss_handler, self.graph_handler, k=3)
                llm_response = self._generate_llm_summary_response("FIND_ANALOGIES", analogies_data or [], user_query, concept_name_for_prompt=concept_name)
                return llm_response or self._format_structured_response("FIND_ANALOGIES", analogies_data, concept_name)
            except Exception as e:
                logger.error(f"Error in FIND_ANALOGIES for '{concept_name}': {e}")
                return self._format_structured_response("FIND_ANALOGIES", None, concept_name)
        
        # Fallback for UNKNOWN intent (handled by UNKNOWN_QUERY in _generate_llm_summary_response)
        logger.warning(f"Query not understood by LLM or regex: {user_query}. Intent was: {intent}")
        unknown_data = {"original_query": user_query} # Pass original query for context
        llm_response = self._generate_llm_summary_response("UNKNOWN_QUERY", unknown_data, user_query, concept_name_for_prompt=user_query)
        return llm_response or "Sorry, I didn't understand that. Try: 'What is [Concept]?', 'Blend [A] and [B]', or 'Analogies for [Concept]?'"

if __name__ == '__main__':
    class MockGraphHandler:
        def get_concept_features(self, concept_name):
            cn_lower = concept_name # Assumes already cleaned by orchestrator
            if cn_lower == "apple": return {"name": "Apple", "properties": {"color": "red", "type": "fruit"}, "relationships": [{"type": "IS_A", "target_concept": "Fruit", "direction": "outgoing"}]}
            if cn_lower == "cpu": return {"name": "Cpu", "properties": {"domain": "Computer science", "abbr_of": "Central Processing Unit"}, "relationships": [{"type": "PART_OF", "target_concept": "Computer", "direction": "outgoing"}]}
            if cn_lower == "artificial intelligence": return {"name": "Artificial Intelligence", "properties": {"domain": "CS", "goal": "create intelligent agents"}, "relationships": []}
            if cn_lower == "fruit": return {"name": "Fruit", "properties": {"category": "food"}, "relationships": []}
            if cn_lower == "computer": return {"name": "Computer", "properties": {"type": "device"}, "relationships": []}    
            if cn_lower == "dna": return {"name": "DNA", "properties": {"type": "molecule", "contains": "genetic information"}, "relationships": []}
            if cn_lower == "market economy": return {"name": "Market Economy", "properties": {"description": "economic system based on supply and demand"}}
            if cn_lower == "photosynthesis": return None 
            if cn_lower == "cat": return None 
            return None
        def get_all_concept_names(self): return ["Apple", "AI", "DNA", "Cpu"]

    class MockFaissHandler: pass
    
    class MockGeminiModel:
        class MockResponse: # Nested class
            def __init__(self, text: str):
                self.text: str = text

        def __init__(self):
            self.MOCK_RESPONSES: Dict[str, str] = {
                # --- Intent Extraction Mocks (key is lowercase user query as extracted by test regex) ---
                "tell me about artificial intelligence": '```json\n{"intent": "GET_DETAILS", "entities": {"concept_name": "Artificial Intelligence"}}\n```',
                "elaborate on photosynthesis?": '```json\n{"intent": "GET_DETAILS", "entities": {"concept_name": "photosynthesis?"}}\n```',
                "blend dna and the concept of a market economy": '```json\n{"intent": "BLEND_CONCEPTS", "entities": {"concept_a_name": "DNA", "concept_b_name": "market economy"}}\n```',
                "similar to a cat.": '```json\n{"intent": "FIND_ANALOGIES", "entities": {"concept_name": "Cat."}}\n```',
                "hello there": '```json\n{"intent": "UNKNOWN", "entities": {}}\n```',
                "what is cpu?": '```json\n{"intent": "GET_DETAILS", "entities": {"concept_name": "cpu?"}}\n```',
                "what is apple": '```json\n{"intent": "GET_DETAILS", "entities": {"concept_name": "apple"}}\n```',
                "tell me about nonexistentllmconcept": '```json\n{"intent": "GET_DETAILS", "entities": {"concept_name": "NonExistentLLMConcept"}}\n```',
                "gibberish query": '```json\n{"intent": "UNKNOWN", "entities": {}}\n```',

                # --- Response Generation Mocks (key is a string constructed from intent, concept, and user query) ---
                # Format: "Intent:<INTENT_UPPER>,Concept:<CleanedConceptForPrompt_Lower>,UserQuery:<OriginalUserQuery_Lower>"
                "Intent:GET_DETAILS,Concept:apple,UserQuery:what is apple": "An Apple is a fruit, often red, and is a type of Fruit.",
                "Intent:GET_DETAILS,Concept:cpu,UserQuery:what is cpu?": "The Cpu, or Central Processing Unit, is a key component in Computer science. It's part of a Computer and an abbreviation of Central Processing Unit.",
                "Intent:GET_DETAILS,Concept:photosynthesis,UserQuery:elaborate on photosynthesis?": "I'm sorry, I couldn't find detailed information for photosynthesis.",
                "Intent:GET_DETAILS,Concept:cat,UserQuery:similar to a cat.": "I couldn't find any details for Cat, so I can't look for analogies right now.",
                "Intent:GET_DETAILS,Concept:nonexistentllmconcept,UserQuery:tell me about nonexistentllmconcept": "I'm sorry, I couldn't find any information about NonExistentLLMConcept.",
                "Intent:BLEND_CONCEPTS,Concept:dna and market economy,UserQuery:blend dna and the concept of a market economy": "Blending DNA and a market economy could lead to a 'Genetic Marketplace', an economic system driven by genetic information.",
                "Intent:BLEND_CONCEPTS,Concept:nonexist a and nonexist b,UserQuery:blend nonexist a and nonexist b": "I was unable to generate a blend for NonExist A and NonExist B, as I couldn't get enough information about them.",
                "Intent:FIND_ANALOGIES,Concept:apple,UserQuery:find analogies for apple": "For Apple, I found a few potential analogies: MockAnalogyPear and MockAnalogyOrange.",
                "Intent:FIND_ANALOGIES,Concept:nonexistentconcept,UserQuery:find analogies for nonexistentconcept": "I looked, but I couldn't find any analogies for NonExistentConcept at the moment.",
                "Intent:UNKNOWN_QUERY,Concept:gibberish query,UserQuery:gibberish query": "I'm not quite sure how to help with 'gibberish query'. Could you try rephrasing? You can ask me to get details about a concept, blend two concepts, or find analogies.",
                "Intent:UNKNOWN_QUERY,Concept:hello there,UserQuery:hello there": "Hello! How can I help you with PolyMind today? You can ask me about concepts, blend ideas, or find analogies.",
                "Intent:UNKNOWN_QUERY,Concept:tell me about nonexistentllmconcept,UserQuery:tell me about nonexistentllmconcept": "I'm sorry, I couldn't find any information about NonExistentLLMConcept even after trying to understand it as a general query."
            }
            logger.debug(f"MockGeminiModel initialized with {len(self.MOCK_RESPONSES)} MOCK_RESPONSES.")

        def generate_content(self, prompt_text: str) -> MockResponse:
            # Normalize the entire prompt once for consistent matching
            # Replace escaped quotes that might come from f-strings in prompts, then common normalization
            normalized_prompt = prompt_text.replace('\"', '"').replace("\\'", "'")
            normalized_prompt = normalized_prompt.lower().replace("\\n", " ").replace('"', "").replace("'", "")

            is_intent_prompt = "output only a json object" in normalized_prompt
            logger.debug(f"MockGeminiModel received prompt (is_intent: {is_intent_prompt}): {normalized_prompt[:300]}...")

            constructed_summary_key = ""  # Ensure it's initialized

            if is_intent_prompt:
                # For intent prompts, the key is the raw user query.
                user_query_in_prompt_match = re.search(r"user query:\s*(.*?)\s*output:", normalized_prompt)
                if user_query_in_prompt_match:
                    # Key for intent mocks is the direct user query, normalized (lowercase, no quotes, no newlines)
                    extracted_user_query_key = user_query_in_prompt_match.group(1).strip()
                    logger.debug(f"Extracted intent key (normalized user query): '{extracted_user_query_key}'")
                    response_text = self.MOCK_RESPONSES.get(extracted_user_query_key)
                    if response_text and "json" in response_text:
                        logger.info(f"MockGeminiModel: Matched INTENT prompt for key: '{extracted_user_query_key}'")
                        return self.MockResponse(response_text)
                    else:
                        logger.warning(f"MockGeminiModel: No INTENT mock found for exact key: '{extracted_user_query_key}'")
            else:  # Summary prompt
                actual_intent_str = "UNKNOWN_INTENT_IN_PROMPT"
                actual_concept_str = "unknown_concept_in_prompt"
                actual_user_query_str = "unknown_user_query_in_prompt"

                # Determine Intent from summary prompt content
                if "based on the following structured information" in normalized_prompt or \
                   "couldnt find detailed information for" in normalized_prompt:
                    actual_intent_str = "GET_DETAILS"
                elif "based on the following blended concept information" in normalized_prompt or \
                     "unable to generate a blend for concepts related to" in normalized_prompt:
                    actual_intent_str = "BLEND_CONCEPTS"
                elif "based on the following list of analogies found" in normalized_prompt or \
                     "couldnt find any analogies for" in normalized_prompt:
                    actual_intent_str = "FIND_ANALOGIES"
                elif "you dont understand this request" in normalized_prompt:  # Matches UNKNOWN_QUERY prompt
                    actual_intent_str = "UNKNOWN_QUERY"

                # Determine User Query from summary prompt content
                user_query_match = re.search(r"user asked:\s*(.*?)(?:\.|based on|politely inform|synthesize this|\(for analogies related to)", normalized_prompt)
                if not user_query_match:
                    user_query_match = re.search(r"user said:\s*(.*?)(?:\.|you dont understand this request)", normalized_prompt)

                if user_query_match:
                    actual_user_query_str = user_query_match.group(1).strip()

                # Determine Concept Name from summary prompt content (this is the trickiest)
                # Order of regex matters: more specific first, or ensure captures are robust.
                if actual_intent_str == "GET_DETAILS":
                    concept_match = re.search(r"information about the concept\s*(.*?)(?:\s*,|\s*based on|\s*properties:|\s*relationships:)|couldnt find detailed information for\s*(.*?)(?:\.|\s*synthesize this)", normalized_prompt)
                    if concept_match:
                        actual_concept_str = (concept_match.group(1) or concept_match.group(2) or actual_concept_str).strip()
                elif actual_intent_str == "BLEND_CONCEPTS":
                    concept_match = re.search(r"unable to generate a blend for concepts related to\s*(.*?)(?:\.|\s*synthesize this)|name:\s*(.*?)-.*?description:", normalized_prompt)
                    if concept_match:
                        actual_concept_str = (concept_match.group(1) or concept_match.group(2) or actual_concept_str).strip()
                        if not concept_match.group(1) and concept_match.group(2):
                            parts = actual_concept_str.split('-')
                            if len(parts) >= 2:
                                if "dna" in parts[0] and ("market economy" in " ".join(parts[1:])):
                                    actual_concept_str = "dna and market economy"
                                elif "nonexist a" in parts[0] and "nonexist b" in " ".join(parts[1:]):
                                    actual_concept_str = "nonexist a and nonexist b"
                elif actual_intent_str == "FIND_ANALOGIES":
                    # Commenting out problematic regex due to persistent linter errors
                    # concept_match = re.search(r"analogies related to\s*(.*?)(?:\s*[)]|\s*based on|$)|couldnt find any analogies for\s*(.*?)(?:[.]|\s*list the analogies)", normalized_prompt)
                    # if concept_match:
                    #     actual_concept_str = (concept_match.group(1) or concept_match.group(2) or actual_concept_str).strip()
                    pass  # actual_concept_str will remain 'unknown_concept_in_prompt'
                elif actual_intent_str == "UNKNOWN_QUERY":
                    concept_match = re.search(r"user said:\s*(.*?)(?:\.|you dont understand this request)", normalized_prompt)
                    if concept_match:
                        actual_concept_str = concept_match.group(1).strip()

                constructed_summary_key = f"Intent:{actual_intent_str},Concept:{actual_concept_str},UserQuery:{actual_user_query_str}"
                logger.debug(f"Constructed summary key for matching: '{constructed_summary_key}'")

                response_text = self.MOCK_RESPONSES.get(constructed_summary_key)
                if response_text and "json" not in response_text:
                    logger.info(f"MockGeminiModel: Matched SUMMARY prompt for KEY: '{constructed_summary_key}'")
                    return self.MockResponse(response_text)
                else:
                    logger.warning(f"MockGeminiModel: No SUMMARY mock found for exact key: '{constructed_summary_key}'")

            fallback_text = ""
            if is_intent_prompt:
                logger.warning(f"MockGeminiModel: FINAL FALLBACK - No specific INTENT mock for prompt: ...{normalized_prompt[-150:]}")
                fallback_text = '''```json\n{"intent": "UNKNOWN", "entities": {}}\n```'''
            else:
                logger.warning(f"MockGeminiModel: FINAL FALLBACK - No specific SUMMARY mock for key '{constructed_summary_key}'. Prompt: ...{normalized_prompt[-150:]}")
                fallback_text = "Mock fallback: I am unable to provide a specific mock summary for this request."

            return self.MockResponse(fallback_text)

    class MockTextProcessor:
        def __init__(self, api_key, model_name): self._model = MockGeminiModel() if api_key else None
        def _get_gemini_model(self): return self._model
    
    mock_feature_extractor_module = type('MockFeatExt', (object,), {'get_concept_context': lambda gh, cn: {"concept_name": cn, "data": gh.get_concept_features(cn)} if gh and cn and gh.get_concept_features(cn) else None})
    mock_blender_module = type('MockBlender', (object,), {'blend_concepts_gemini': lambda ca, cb: {"blended_name": f"{ca.get('concept_name','A')}-{cb.get('concept_name','B')}", "description": "A mock blend description."} if ca and cb else None})
    mock_analogy_finder_module = type('MockAnalogy', (object,), {'find_analogous_concepts_vector': lambda qcn, fh, gh, k: ([{"name": "MockAnalogyPear", "score":0.8}, {"name":"MockAnalogyOrange", "score":0.75}] if qcn.lower()=="apple" and gh.get_concept_features(qcn) else [])})

    logging.basicConfig(level=logging.DEBUG) 
    logger.info("--- Running ChatOrchestrator Direct Tests (Full Phase 2 Mocks) ---")
    mock_graph, mock_faiss = MockGraphHandler(), MockFaissHandler()
    mock_tp = MockTextProcessor(api_key="FAKE_KEY", model_name="gemini-pro")
    orchestrator = ChatOrchestrator(mock_graph, mock_faiss, mock_tp, mock_feature_extractor_module, mock_blender_module, mock_analogy_finder_module)

    test_queries = [
        "Tell me about Artificial Intelligence.", 
        "Elaborate on photosynthesis?",  
        "Blend DNA and the concept of a market economy", 
        "Blend NonExist A and NonExist B", 
        "What is similar to a Cat.", 
        "What is Apple", 
        "What is cpu?", 
        "Find analogies for Apple", 
        "find analogies for NonExistentConcept",
        "Hello there!",
        "Tell me about NonExistentLLMConcept",
        "gibberish query" 
    ]
    for query in test_queries:
        print(f"\nUser: {query}")
        response = orchestrator.parse_and_execute_command(query)
        print(f"PolyMind: {response}")
    
    logger.info("\n--- Specific Regex Fallback Tests (mocking LLM intent failure) ---")
    regex_fallback_queries = [
        "details for \"Fruit\"?", 
        "Combine Computer and DNA.", 
        "tell me about", 
        "blend A and", 
        "analogies for" 
    ]
    original_intent_llm = orchestrator._get_intent_and_entities_llm
    orchestrator._get_intent_and_entities_llm = lambda q: {"intent": "UNKNOWN", "entities": {}} 
    original_summary_llm = orchestrator._generate_llm_summary_response
    orchestrator._generate_llm_summary_response = lambda i,d,q,concept_name_for_prompt=None: None 

    for query in regex_fallback_queries:
        print(f"\nUser (Regex Fallback Test): {query}")
        response = orchestrator.parse_and_execute_command(query)
        print(f"PolyMind: {response}")
    
    orchestrator._get_intent_and_entities_llm = original_intent_llm
    orchestrator._generate_llm_summary_response = original_summary_llm
    logger.info("\n--- ChatOrchestrator Direct Tests Completed ---")