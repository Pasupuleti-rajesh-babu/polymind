# PolyMind Chat Feature - Development Plan

## 1. Overall Goal

To create an interactive chat interface within PolyMind that allows users to query their knowledge graph, leverage synthesis features (blending, analogy), and explore ideas using natural language. The chat should feel conversational and progressively become more intelligent in understanding user intent and orchestrating PolyMind's core capabilities.

## 2. Core Principles

*   **Iterative Development:** Start simple and add complexity in phases.
*   **Leverage Existing Core Modules:** The chat will primarily be an interface and orchestration layer over `GraphHandler`, `FaissHandler`, `TextProcessor`, `FeatureExtractor`, `Blender`, and `AnalogyFinder`.
*   **User-Centric:** Aim for intuitive interaction, clear responses, and mechanisms for clarification.
*   **Grounded Responses:** Where possible, ensure chat outputs are grounded in the knowledge derived from the user's ingested data.

## 3. Phases of Development

### Phase 1: MVP - Basic Command Handling & UI

*   **Objective:** Establish the chat UI and handle a few predefined, direct commands.
*   **Key Features:**
    1.  **UI Setup (`streamlit_app.py`):**
        *   New "PolyMind Chat" section in the Streamlit sidebar.
        *   Use `st.chat_input` for user queries.
        *   Use `st.chat_message` to display conversation history (user queries and assistant responses).
        *   Store conversation history in `st.session_state`.
    2.  **Simple Command Parser (`chat_orchestrator.py` - new module):**
        *   A new module `polymind_core/chat_interface/chat_orchestrator.py` will be created.
        *   Initial version will use regex or simple keyword matching to identify intents.
        *   Function: `parse_and_execute_command(user_query: str, graph_handler, faiss_handler, text_processor) -> str`
    3.  **Supported Commands (Initial Set):**
        *   **Get Concept Details:**
            *   User query patterns: "What is [Concept Name]?", "Tell me about [Concept Name]", "Details for [Concept Name]"
            *   Action: Call `graph_handler.get_concept_features()`. Format and return properties and relationships.
        *   **Blend Two Concepts:**
            *   User query patterns: "Blend [Concept A] and [Concept B]", "Combine [Concept A] with [Concept B]"
            *   Action: Fetch contexts for A and B using `feature_extractor.get_concept_context()`. Call `blender.blend_concepts_gemini()`. Format and return the blend.
        *   **Find Analogies:**
            *   User query patterns: "Find analogies for [Concept Name]", "What is similar to [Concept Name]?"
            *   Action: Call `analogy_finder.find_analogous_concepts_vector()` (default) or allow specifying type. Format and return analogies.
    4.  **Basic Error Handling:** If a command isn't recognized or fails, provide a polite message.
*   **Success Metrics:**
    *   User can type supported commands and see structured results from PolyMind core functions within the chat interface.
    *   Conversation history is displayed.

### Phase 2: Enhanced NLU & Response Formatting

*   **Objective:** Improve natural language understanding and make responses more conversational.
*   **Key Features:**
    1.  **LLM-based Intent Recognition (`chat_orchestrator.py`):**
        *   Integrate an LLM (e.g., via `TextProcessor` or a dedicated Gemini call) into `parse_and_execute_command` to better classify user intent and extract entities (concept names, parameters) from more varied phrasings.
        *   The prompt to the LLM would include examples of supported intents and how to extract information.
    2.  **LLM-based Response Generation (`chat_orchestrator.py`):**
        *   Instead of just formatting raw output from core modules, use an LLM to synthesize the results into a more natural, paragraph-style response.
        *   Example: For concept details, summarize key properties rather than just listing them.
    3.  **Handling "Not Found":** Improved messages when concepts are not found in the graph.
    4.  **Refinement of Supported Commands:** Expand variations of phrasing for Phase 1 commands.
*   **Success Metrics:**
    *   Chat can understand a wider range of phrasing for existing commands.
    *   Responses are more conversational and easier to read.

### Phase 3: Contextual Conversation & Clarification

*   **Objective:** Enable multi-turn conversations and allow the chat to ask for clarification.
*   **Key Features:**
    1.  **Conversation History for Context (`chat_orchestrator.py`):**
        *   Pass relevant parts of the `st.session_state` conversation history to the NLU/orchestration logic.
        *   Allow use of pronouns (e.g., "Tell me more about *it*").
    2.  **Clarification Mechanism (`chat_orchestrator.py`):**
        *   Implement logic (rule-based or LLM-assisted) to detect ambiguous or underspecified queries.
        *   If ambiguity is detected, the `parse_and_execute_command` function will return a clarifying question instead of an answer.
        *   The Streamlit app will display this question, and the user's next input will be treated as a response to it.
    3.  **Handling Clarification Responses:** The orchestrator needs to integrate the user's clarification into the original intent before proceeding.
*   **Success Metrics:**
    *   Chat can understand follow-up questions using context.
    *   Chat proactively asks for clarification when queries are ambiguous, leading to more accurate final answers.

### Phase 4: Advanced Query Orchestration & Proactive Suggestions (Longer Term)

*   **Objective:** Handle complex, multi-step queries and potentially offer proactive suggestions.
*   **Key Features:**
    1.  **Multi-Step Query Planning (`chat_orchestrator.py`):**
        *   For complex questions like "What ideas from nature could inspire a new way to stop HIV?", the orchestrator (likely LLM-driven, an "agent") would break this down:
            *   Identify "HIV" and "nature" as key entities/domains.
            *   Query `GraphHandler` / `FeatureExtractor` for context on HIV.
            *   Query `GraphHandler` / `FaissHandler` for relevant concepts/principles from "nature".
            *   Formulate a prompt for an LLM (like `blend_concepts_gemini` but more targeted) to synthesize ideas based on the gathered context.
    2.  **Access to All PolyMind Tools:** Enable the orchestrator to decide to call any relevant PolyMind core function.
    3.  **Grounding Complex Responses:** Ensure that even for creative generation, the ideas are explicitly linked back to information found within PolyMind.
    4.  **(Optional) Proactive Suggestions:** Based on conversation history or the current state of exploration, the chat might suggest related concepts, potential blends, or relevant unasked questions.
*   **Success Metrics:**
    *   Chat can attempt to answer complex, open-ended questions by intelligently combining PolyMind's tools.
    *   Users can have more exploratory and generative dialogues with their knowledge base.

## 4. Technical Considerations & Modules

*   **`streamlit_app.py`:** UI elements, session state management for conversation.
*   **`polymind_core/chat_interface/chat_orchestrator.py` (New):**
    *   `ChatOrchestrator` class.
    *   Methods for NLU (intent/entity extraction), command execution, response generation, clarification logic, context management.
    *   Will interact with all other core PolyMind modules.
*   **`config.py`:** Potentially new configuration for chat-specific LLM prompts or parameters.
*   **Error Handling & Logging:** Robust logging throughout the chat interaction flow.

## 5. Future Enhancements (Post-Phase 4)

*   Saving/Loading chat sessions.
*   User feedback mechanism for chat responses.
*   More sophisticated agentic behavior (e.g., self-correction, tool learning).
*   Integration with external data sources via search, if desired. 