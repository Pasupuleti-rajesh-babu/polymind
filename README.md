# POLYMIND - Local Knowledge Synthesis System

POLYMIND is a local prototype of a knowledge synthesis system designed to extract structured meaning from text, build a knowledge graph, and enable analogical reasoning across different domains. It runs entirely on your local machine, ensuring privacy and control over your data.

## Core Idea

The system leverages large language models (LLMs) like Google's Gemini to identify key concepts and their relationships ("meaning atoms") within text. These are then stored in a Neo4j graph database. Concepts are also converted into vector embeddings using SentenceTransformer models and indexed in a FAISS vector store for fast similarity-based analogical search. A Streamlit UI allows users to query for analogies and explore the knowledge graph.

## Key Features

*   **Text Ingestion**: Processes raw text or content from sources like Wikipedia articles.
*   **Meaning Extraction (Gemini API)**: Uses the Gemini API to extract significant concepts and their relationships in `(Subject, Relationship, Object)` triples.
*   **Graph-Based Memory**: Stores concepts as nodes and relationships as edges in a local Neo4j database.
*   **Vector Embeddings**: Generates semantic embeddings for each concept using SentenceTransformer models.
*   **Analogical Search (FAISS)**: Enables finding semantically similar concepts across different domains using FAISS for fast vector search.
*   **Interactive UI (Streamlit)**: Provides an interface to input text, ask for analogies (e.g., "What is an analogy for DNA in philosophy?"), and visualize graph context.
*   **Local & Private**: All data (graph, vectors, text) is stored and processed locally. No cloud storage dependencies for core data.

## Tech Stack

*   **Python**: 3.10+
*   **LLM**: Google Gemini API (via `google-generativeai` SDK)
*   **Graph Database**: Neo4j (local instance)
*   **Vector Search**: FAISS (`faiss-cpu`)
*   **Embeddings**: `sentence-transformers`
*   **User Interface**: Streamlit
*   **Other Libraries**: `spacy` (for potential fallback or supplementary NLP), `py2neo`, `networkx`, `matplotlib`.

## Project Structure

*   `polymind_app.py`: Main Streamlit application script containing the core logic for UI, data processing, API calls, and search.
*   `requirements.txt`: Lists all Python dependencies.
*   `README.md`: This file.

## Setup and Installation

**1. Prerequisites:**

*   **Python 3.10 or newer**: Ensure you have a compatible Python version installed.
*   **Neo4j Desktop or Server**:
    *   Download and install [Neo4j Desktop](https://neo4j.com/download/).
    *   Create a new project and a new local graph database (e.g., version 4.x or 5.x).
    *   Ensure the database is **running**.
    *   Note down the **Bolt URL** (usually `bolt://localhost:7687`), **username** (default is `neo4j`), and the **password** you set for the database. You will need to update these in `polymind_app.py`.

**2. Clone the Repository (if applicable):**
   ```bash
   # If this project were in a git repo:
   # git clone <repository_url>
   # cd polymind
   ```

**3. Set up Python Virtual Environment:**
   It's highly recommended to use a virtual environment.
   ```bash
   python3 -m venv .polymindenv
   source .polymindenv/bin/activate  # On Windows: .polymindenv\Scripts\activate
   ```

**4. Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   This will install Streamlit, Py2Neo, FAISS, SentenceTransformers, Google Generative AI SDK, and other necessary libraries.

**5. Download spaCy Model (for auxiliary NLP tasks):**
   Although the primary extraction uses Gemini, spaCy might be used for some text processing.
   ```bash
   python -m spacy download en_core_web_sm
   ```

**6. Configure API Keys and Neo4j Credentials:**

   *   **Gemini API Key**:
        *   You need a Google Gemini API key. You can obtain one from [Google AI Studio](https://aistudio.google.com/app/apikey).
        *   **IMPORTANT**: For security, it's best to set your API key as an environment variable.
            ```bash
            export GEMINI_API_KEY="YOUR_API_KEY_HERE"
            ```
            The application will attempt to read this environment variable. As a fallback, or for quick testing, you can hardcode it in `polymind_app.py` (as shown in your provided brief), but this is **not recommended for production or shared code**.
        *   The code is currently set up to use the key `AIzaSyDuSjazaZTKq4TJJGnpz8P8IvDOFqOA3cc` as provided in your brief. **Replace this with your actual key if it's different or manage it via environment variables.**

   *   **Neo4j Credentials**:
        *   Open `polymind_app.py`.
        *   Locate the Neo4j configuration section near the top:
            ```python
            NEO4J_URI = "bolt://localhost:7687"
            NEO4J_USER = "neo4j"
            NEO4J_PASSWORD = "YOUR_NEO4J_PASSWORD" # Change this!
            ```
        *   Update `NEO4J_PASSWORD` with the password you set for your Neo4j database. Ensure `NEO4J_URI` and `NEO4J_USER` match your Neo4j setup.

## How to Run POLYMIND

1.  **Ensure Neo4j is running.**
2.  **Activate your virtual environment:**
    ```bash
    source .polymindenv/bin/activate
    ```
3.  **Set your Gemini API Key (if using environment variables):**
    ```bash
    export GEMINI_API_KEY="YOUR_API_KEY_HERE"
    ```
4.  **Run the Streamlit application:**
    ```bash
    streamlit run polymind_app.py
    ```
    This will typically open the POLYMIND interface in your web browser automatically. If not, the terminal will provide a local URL (e.g., `http://localhost:8501`).

## How It Works

1.  **Initialization (`st.session_state`)**:
    *   When the Streamlit app starts, it initializes connections and models only once and stores them in `st.session_state` to persist across UI interactions.
    *   Connects to Neo4j. If the database is empty, it loads sample concepts and relationships.
    *   Initializes the `SentenceTransformer` model for embeddings.
    *   Initializes the Gemini API client (`genai.GenerativeModel`).
    *   Builds (or rebuilds if necessary) the FAISS index from concepts currently in Neo4j.

2.  **Text Ingestion & Meaning Extraction (Future Feature - currently uses sample data)**:
    *   *(The current prototype primarily uses pre-defined sample data for graph population during initialization. The UI for direct text input and Gemini processing is a planned next step based on your brief.)*
    *   **Intended Flow**:
        1.  User provides text (e.g., pastes an article snippet or provides a URL).
        2.  The text is sent to the Gemini API with a prompt like: *"Extract all significant concepts and their relationships in the form (Subject, Relationship, Object) from the following text: [TEXT_HERE]"*.
        3.  The Gemini API response (expected to be a list of triples) is parsed.

3.  **Graph Population**:
    *   Each unique concept from the extracted triples (and sample data) becomes a `Concept` node in Neo4j. Nodes have `name` and `domain` properties.
    *   Each `(Subject, Relationship, Object)` triple forms a directed relationship between the corresponding `Concept` nodes in Neo4j.

4.  **Embedding Generation & FAISS Indexing**:
    *   All unique concepts loaded into Neo4j are retrieved.
    *   The `SentenceTransformer` model generates a vector embedding for each concept name.
    *   These embeddings, along with their corresponding concept names and domains, are used to build a FAISS index. FAISS allows for very fast similarity searches (e.g., finding the most similar vectors to a query vector).
    *   Embeddings are normalized (L2 normalization) before being added to FAISS, which is common practice for cosine similarity searches using an `IndexFlatIP` (Inner Product) index.

5.  **Analogical Query Processing (Streamlit UI)**:
    *   The user types a query, typically like "Analogy for [Concept X] in [Domain Y]?" or "What is an analogy for [Concept X]?".
    *   A regular expression parses the query to extract the `source_concept` (e.g., "DNA") and an optional `target_domain` (e.g., "philosophy").
    *   **Vector Search**:
        1.  The embedding for the `source_concept` is generated (or retrieved if already indexed).
        2.  FAISS is queried to find the `k` most similar concept embeddings from its index.
        3.  Results are filtered to find concepts that are **not** in the same domain as the `source_concept`. If a `target_domain` was specified, it prioritizes results from that domain.
        4.  The top distinct-domain analogy is presented to the user.
    *   **Graph Context Visualization**:
        1.  After an analogy is found (e.g., "DNA" in Biology is analogous to "Meme" in Philosophy), the system queries Neo4j.
        2.  It fetches the `source_concept`, the `analogous_concept`, and their direct neighbors (1-hop relationships) from the graph.
        3.  A subgraph is constructed using `networkx` and visualized using `matplotlib` directly within the Streamlit UI, showing the concepts and their connections.

6.  **Data Persistence**:
    *   **Neo4j**: Data stored in Neo4j is persistent as long as your Neo4j database server is running and managing its data files.
    *   **FAISS Index**: The FAISS index and associated concept lists are currently rebuilt in memory each time the Streamlit application starts, based on the contents of Neo4j. For a more persistent FAISS index, it could be saved to disk and reloaded (e.g., using `faiss.write_index()` and `faiss.read_index()`), but this is not yet implemented.
    *   **Embeddings**: Embeddings are generated as needed and used to build the FAISS index. The raw embeddings themselves (aside from being in the FAISS index in memory) are not explicitly saved to disk in the current version.

## Modular Functions (Conceptual Mapping to `polymind_app.py`)

While `polymind_app.py` is a single script, its functions map to these modular ideas:

*   `call_gemini_api_for_extraction` (NEW - to be added): Will encapsulate calls to Gemini.
*   `load_concepts_to_neo4j`, `load_triples_to_neo4j`: Handle building/updating the graph.
*   `initialize_embedding_model`, `generate_embeddings`, `build_faiss_index`: Manage concept embeddings and the vector store.
*   `find_analogous_concept_vector`: Performs the FAISS-based analogy search.
*   The Streamlit UI part of the script (starting with `st.title(...)`) handles rendering the UI and query processing logic.

## Future Enhancements / Extra Ideas

*   **Gemini for Analogical Explanations**: After finding an analogy, call Gemini again to generate a textual explanation of *why* it's a good analogy.
*   **API Call Caching/Rate Limiting**: Implement caching for Gemini API responses to save costs and allow offline use with previously processed texts. Add rate limiting if making many calls.
*   **Persistent FAISS Index**: Save and load the FAISS index to/from disk to avoid rebuilding it on every app start, especially for large knowledge bases.
*   **Advanced Graph Traversal**: Implement more sophisticated graph algorithms for finding analogies or insights beyond simple neighborhood views (e.g., pathfinding, community detection).
*   **User Input for Text Ingestion**: Fully implement the UI for users to paste text or provide URLs for processing by Gemini.
*   **Error Handling & Robustness**: Add more comprehensive error handling throughout the application.
*   **Configuration File**: Move settings like API keys, Neo4j credentials, and model names to a configuration file (e.g., `config.ini` or `.env` file) instead of hardcoding.

## Troubleshooting

*   **Neo4j Connection Issues**:
    *   Ensure your Neo4j server is running.
    *   Verify `NEO4J_URI`, `NEO4J_USER`, and `NEO4J_PASSWORD` in `polymind_app.py` are correct.
    *   Check Neo4j logs for any errors.
*   **FAISS / Embedding Issues**:
    *   Ensure `sentence-transformers` and `faiss-cpu` are installed correctly.
    *   The embedding model (`all-MiniLM-L6-v2` by default) will be downloaded on first use; ensure you have an internet connection for this.
*   **Streamlit App Not Running**:
    *   Make sure you are in the correct directory and your virtual environment is activated.
    *   Check for any error messages in the terminal when you run `streamlit run polymind_app.py`.
*   **Gemini API Errors**:
    *   Ensure your `GEMINI_API_KEY` is correctly set (either as an environment variable or in the code).
    *   Check for network connectivity.
    *   Review Gemini API documentation for specific error codes.
*   **`TypeError: search() missing 1 required positional argument: 'string'`**: This was a previous bug in regex handling. Ensure you have the corrected version of the regex search in `polymind_app.py`. The pattern string should be defined first, then passed to `re.search()` along with the `user_query`.

This README provides a good overview. Let me know when you're ready for the next steps! 