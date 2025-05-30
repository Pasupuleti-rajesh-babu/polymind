import google.generativeai as genai
from typing import List, Dict, Any, Optional, Tuple
import logging
import re

# Attempt to import project-specific modules
try:
    from ... import config
    from . import feature_extractor # To use for testing in __main__
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
    # If feature_extractor is in the same directory (synthesis_engine)
    if os.path.basename(current_script_dir) == "synthesis_engine":
        from . import feature_extractor
    else: # If script is run from polymind_v2/ for example
        from polymind_core.synthesis_engine import feature_extractor

# Configure logging
log_level = getattr(config, 'LOG_LEVEL', 'INFO')
log_format = getattr(config, 'LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.basicConfig(level=log_level, format=log_format)
logger = logging.getLogger(__name__)

_gemini_model_blender = None

def _get_gemini_model_for_blender():
    """Initializes or retrieves an existing Gemini model instance for blending."""
    global _gemini_model_blender
    if _gemini_model_blender is None:
        if config.GEMINI_API_KEY and config.GEMINI_API_KEY != "YOUR_GEMINI_API_KEY_HERE":
            try:
                genai.configure(api_key=config.GEMINI_API_KEY)
                _gemini_model_blender = genai.GenerativeModel(config.DEFAULT_GEMINI_MODEL)
                logger.info(f"Blender: Gemini model '{config.DEFAULT_GEMINI_MODEL}' initialized successfully.")
            except Exception as e:
                logger.error(f"Blender: Error configuring Gemini API or initializing model: {e}")
                _gemini_model_blender = None
        else:
            logger.warning("Blender: Gemini API Key not set or placeholder. Blending will be unavailable.")
            _gemini_model_blender = None
    return _gemini_model_blender

def _format_concept_context_for_prompt(concept_context: Dict[str, Any], concept_label: str) -> str:
    """Formats the rich context of a concept into a string for the Gemini prompt."""
    if not concept_context:
        return f"{concept_label}: Not found or no context available.\n"

    name = concept_context.get("concept_name", "Unknown Concept")
    properties = concept_context.get("properties", {})
    domain = properties.get("domain", "Unknown Domain")
    description_prop = properties.get("description", "No description provided.") # Assuming a 'description' prop might exist

    prompt_str = f"--- {concept_label}: {name} ---\n"
    prompt_str += f"Domain: {domain}\n"
    if description_prop != "No description provided.":
         prompt_str += f"Description: {description_prop}\n"
    
    prompt_str += "Key Attributes & Connections:\n"
    
    # Simplified properties list for prompt to avoid overwhelming LLM
    simplified_props = {k: v for k, v in properties.items() if k not in ['name', 'domain', 'description'] and isinstance(v, (str, int, float, bool))}
    if simplified_props:
        for prop, val in simplified_props.items():
            prompt_str += f"- Has property: {prop} = {val}\n"

    outgoing = concept_context.get("outgoing_relationships", [])
    if outgoing:
        for rel in outgoing[:config.MAX_FEATURES_FOR_PROMPT]: # Limit features shown in prompt
            target_props = rel.get("target_properties", {})
            target_name = target_props.get("name", "Unknown Target")
            prompt_str += f"- {name} --[{rel.get('type', 'RELATED_TO')}]--> {target_name}\n"
    
    incoming = concept_context.get("incoming_relationships", [])
    if incoming:
        for rel in incoming[:config.MAX_FEATURES_FOR_PROMPT]: # Limit features shown in prompt
            source_props = rel.get("source_properties", {})
            source_name = source_props.get("name", "Unknown Source")
            prompt_str += f"- {source_name} --[{rel.get('type', 'RELATED_TO')}]--> {name}\n"
            
    return prompt_str + "\n"

def blend_concepts_gemini(concept_a_context: Dict[str, Any], concept_b_context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Uses Gemini to perform a conceptual blend of two concepts based on their contexts.

    Args:
        concept_a_context (Dict[str, Any]): The rich context of Concept A from feature_extractor.
        concept_b_context (Dict[str, Any]): The rich context of Concept B from feature_extractor.

    Returns:
        Optional[Dict[str, Any]]: A dictionary with the blended concept's details 
                                   (name, description, features, emergent_properties), 
                                   or None if blending fails or parsing is unsuccessful.
    """
    gemini_model = _get_gemini_model_for_blender()
    if not gemini_model:
        logger.error("Gemini model not available. Cannot perform conceptual blend.")
        return None

    if not concept_a_context or not concept_b_context:
        logger.error("Context for one or both concepts is missing. Cannot blend.")
        return None

    concept_a_name = concept_a_context.get("concept_name", "Concept A")
    concept_b_name = concept_b_context.get("concept_name", "Concept B")

    prompt_intro = f"""Perform a creative conceptual blend of two concepts: {concept_a_name} and {concept_b_name}.
Analyze their provided contexts, identify shared structures, contrasting elements, and potential synergies.
Generate a novel blended concept that integrates key aspects of both, leading to new insights or functionalities.

Present your output strictly in the following format, with each section clearly marked:

**Blended Concept Name:** [A concise and evocative name for the new concept]

**Blended Concept Description:** [A detailed paragraph describing the blended concept, its core idea, and how it combines elements of the original concepts.]

**Key Combined Features:**
- [Feature 1 derived from the blend]
- [Feature 2 derived from the blend]
- ... (list several key features that arise from integrating the source concepts)

**Potential Emergent Properties:**
- [Emergent property 1 - a novel characteristic or capability that arises uniquely from the blend, not present in either parent]
- [Emergent property 2]
- ... (list distinct emergent properties)

Context for the concepts:
"""
    prompt_context_a = _format_concept_context_for_prompt(concept_a_context, "Concept A")
    prompt_context_b = _format_concept_context_for_prompt(concept_b_context, "Concept B")

    full_prompt = prompt_intro + "\n" + prompt_context_a + prompt_context_b
    logger.debug(f"Full prompt for conceptual blending:\n{full_prompt}")

    try:
        logger.info(f"Sending blend request to Gemini for '{concept_a_name}' and '{concept_b_name}'.")
        # Higher temperature for more creative/diverse blends
        generation_config = genai.types.GenerationConfig(temperature=0.8)
        response = gemini_model.generate_content(full_prompt, generation_config=generation_config)
        raw_response_text = response.text
        logger.info(f"Gemini blend response received (first 500 chars): {raw_response_text[:500]}...")

        # Parse the response
        blended_data: Dict[str, Any] = {
            "blended_name": "Unnamed Blend",
            "description": "", # Initialize as empty to build it up
            "combined_features": [],
            "emergent_properties": []
        }
        current_section = None
        # A flag to indicate if we are actively seeking the blended name on the next line
        expecting_blended_name_next_line = False 

        for line_num, raw_line in enumerate(raw_response_text.splitlines()):
            line = raw_line.strip()
            logger.debug(f"BLEND_PARSE Line {line_num}: '{line}' | Current Section: {current_section} | Expecting Name: {expecting_blended_name_next_line}")

            if line.startswith("**Blended Concept Name:**"):
                current_section = "name"
                name_part = line.replace("**Blended Concept Name:**", "").strip()
                if name_part: # Name is on the same line
                    blended_data['blended_name'] = name_part
                    expecting_blended_name_next_line = False
                else: # Name might be on the next line
                    expecting_blended_name_next_line = True
                continue # Move to next line or further processing

            elif line.startswith("**Blended Concept Description:**"):
                current_section = "description"
                expecting_blended_name_next_line = False # Stop expecting name if we hit description
                desc_part = line.replace("**Blended Concept Description:**", "").strip()
                if desc_part:
                    blended_data['description'] = desc_part
                # If desc_part is empty, the actual description starts on the next line
                continue

            elif line.startswith("**Key Combined Features:**"):
                current_section = "features"
                expecting_blended_name_next_line = False
                continue

            elif line.startswith("**Potential Emergent Properties:**"):
                current_section = "emergent"
                expecting_blended_name_next_line = False
                continue

            # Handle content based on current_section or if expecting name
            if expecting_blended_name_next_line and line:
                blended_data['blended_name'] = line # Assign the first non-empty line as name
                expecting_blended_name_next_line = False # Got the name
                # current_section = None # Or decide next section based on subsequent lines.
                                         # For now, let the next iteration pick up a section marker.
                continue


            if current_section == "description":
                # Append to description if it's not a new section marker
                if not (line.startswith("**") and ":**" in line):
                    if blended_data['description']: # Add space if not the first line of description
                        blended_data['description'] += " " + line
                    else:
                        blended_data['description'] = line
            
            elif current_section == "features":
                # Prioritize bullet points, but also accept plain lines if they don't look like section markers
                if line.startswith("- ") or line.startswith("* "):
                    feature = line[2:].strip() # Remove "- " or "* "
                    if feature: blended_data["combined_features"].append(feature)
                elif line and not (line.startswith("**") and ":**" in line): # Non-empty line, not a new section
                    if blended_data["combined_features"] and not blended_data["combined_features"][-1].endswith(('.', '!', '?')):
                        # Heuristic: if previous feature looks like a continuation, append
                        blended_data["combined_features"][-1] += " " + line
                    else:
                         blended_data["combined_features"].append(line)


            elif current_section == "emergent":
                if line.startswith("- ") or line.startswith("* "):
                    prop = line[2:].strip() # Remove "- " or "* "
                    if prop: blended_data["emergent_properties"].append(prop)
                elif line and not (line.startswith("**") and ":**" in line): # Non-empty line, not a new section
                    if blended_data["emergent_properties"] and not blended_data["emergent_properties"][-1].endswith(('.', '!', '?')):
                         blended_data["emergent_properties"][-1] += " " + line
                    else:
                        blended_data["emergent_properties"].append(line)
        
        # Final cleanup of description
        blended_data['description'] = blended_data['description'].strip() if blended_data['description'] else "No description provided by Gemini or parsing failed."
        if not blended_data['description'] and blended_data['blended_name'] != "Unnamed Blend": # if name was parsed but desc wasn't
            logger.warning(f"Description for '{blended_data['blended_name']}' might be missing or misparsed.")


        if blended_data['blended_name'] == "Unnamed Blend" and not blended_data['combined_features'] and not blended_data['emergent_properties']:
             logger.warning("Could not parse much from the blend response. Structure might be very different from expected.")
             # Optionally, if you want to signal a more complete failure:
             # blended_data['description'] = "Failed to parse Gemini response for blend." 

        logger.info(f"Successfully parsed blended concept: '{blended_data['blended_name']}'")
        logger.debug(f"Parsed blended_data: {blended_data}")
        return blended_data

    except Exception as e:
        logger.error(f"Error during Gemini API call or parsing for blend: {e}", exc_info=True)
        return None

if __name__ == '__main__':
    import pprint
    logger.info("Running blender.py direct tests...")

    if not config.GEMINI_API_KEY or config.GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        logger.error("GEMINI_API_KEY not configured. Skipping blender direct tests.")
    elif not config.NEO4J_PASSWORD or config.NEO4J_PASSWORD == "YOUR_NEO4J_PASSWORD_HERE":
        logger.warning("NEO4J_PASSWORD seems to be a placeholder. Ensure Neo4j is accessible for feature_extractor.")
        # Proceeding, but feature_extractor might fail to connect.
    else:
        # Test requires feature_extractor to work and get data from Neo4j
        concept1_name = "Artificial intelligence"
        concept2_name = "Dna"

        logger.info(f"Fetching context for '{concept1_name}'...")
        context1 = feature_extractor.get_concept_context(concept1_name)
        if not context1:
            logger.error(f"Failed to get context for '{concept1_name}'. Make sure it's in Neo4j. Aborting test.")
            sys.exit(1)
        
        logger.info(f"Fetching context for '{concept2_name}'...")
        context2 = feature_extractor.get_concept_context(concept2_name)
        if not context2:
            logger.error(f"Failed to get context for '{concept2_name}'. Make sure it's in Neo4j. Aborting test.")
            sys.exit(1)
        
        logger.info(f"\n--- Attempting to blend '{concept1_name}' and '{concept2_name}' ---")
        blended_result = blend_concepts_gemini(context1, context2)

        if blended_result:
            logger.info("--- Blended Concept ---_INFO")
            pprint.pprint(blended_result)
            assert blended_result.get("blended_name") != "Unnamed Blend"
            assert blended_result.get("description")
            assert isinstance(blended_result.get("combined_features"), list)
            assert isinstance(blended_result.get("emergent_properties"), list)
        else:
            logger.error("Conceptual blending failed or produced no parseable result.")

    logger.info("\nblender.py direct tests completed.") 