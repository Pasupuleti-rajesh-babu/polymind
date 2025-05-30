import streamlit as st
import os
import sys

# Add project root to sys.path
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from polymind_core.knowledge_graph.graph_handler import GraphHandler
    from polymind_core.meaning_extraction.text_processor import TextProcessor
    from polymind_core.vector_store.faiss_handler import FaissHandler
    from data_ingestion.wikipedia_ingestor import ingest_wikipedia_article
    from polymind_core.synthesis_engine.feature_extractor import get_concept_context
    from polymind_core.synthesis_engine.blender import blend_concepts_gemini
    from polymind_core.synthesis_engine.analogy_finder import (
        find_analogous_concepts_vector,
        find_analogous_concepts_structural_graph,
        generate_analogy_explanation_gemini
    )
    import config
except ImportError as e:
    st.error(f"Error importing modules: {e}. Please ensure all components are correctly installed and paths are set up.")
    st.stop()

# --- Page Configuration ---
st.set_page_config(page_title="PolyMind V2", layout="wide")

# --- Helper Functions & State Management ---
@st.cache_resource
def get_graph_handler():
    try:
        return GraphHandler(uri=config.NEO4J_URI, user=config.NEO4J_USER, password=config.NEO4J_PASSWORD)
    except Exception as e:
        st.error(f"Failed to connect to Neo4j: {e}")
        return None

@st.cache_resource
def get_text_processor():
    try:
        return TextProcessor(api_key=config.GEMINI_API_KEY, model_name=config.GEMINI_MODEL_NAME)
    except Exception as e:
        st.error(f"Failed to initialize Text Processor (Gemini): {e}")
        return None

@st.cache_resource
def get_faiss_handler():
    try:
        handler = FaissHandler(
            embedding_model_name=config.EMBEDDING_MODEL_NAME,
            faiss_index_path=config.FAISS_INDEX_PATH,
            concept_list_path=config.FAISS_CONCEPT_LIST_PATH
        )
        handler.load_index() # Load existing index or create if not found
        return handler
    except Exception as e:
        st.error(f"Failed to initialize FAISS Handler: {e}")
        return None

# Initialize handlers
graph_handler = get_graph_handler()
text_processor = get_text_processor()
faiss_handler = get_faiss_handler()

if not graph_handler or not text_processor or not faiss_handler:
    st.error("Core handlers could not be initialized. Application cannot proceed.")
    st.stop()

if 'all_concepts' not in st.session_state:
    st.session_state.all_concepts = []

def refresh_concept_list():
    if graph_handler:
        st.session_state.all_concepts = graph_handler.get_all_concept_names()
    else:
        st.session_state.all_concepts = []

if not st.session_state.all_concepts:
    refresh_concept_list()

# --- Main Application ---
st.title("🧠 PolyMind V2 - Knowledge Synthesis System")

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose a section:", 
    ("Knowledge Ingestion", "Knowledge Browser", "Conceptual Blending", "Analogical Search"))

# --- 1. Knowledge Ingestion ---
if app_mode == "Knowledge Ingestion":
    st.header("📚 Knowledge Ingestion")
    st.subheader("Ingest Content from Wikipedia")

    article_title = st.text_input("Enter Wikipedia Article Title:")
    if st.button("Ingest Article"):
        if not article_title:
            st.warning("Please enter an article title.")
        elif not graph_handler or not text_processor or not faiss_handler:
            st.error("One or more core services are not available. Ingestion aborted.")
        else:
            with st.spinner(f"Ingesting '{article_title}'..."):
                try:
                    success, message = ingest_wikipedia_article(
                        article_title,
                        graph_handler,
                        text_processor,
                        faiss_handler
                    )
                    if success:
                        st.success(message)
                        refresh_concept_list() # Update concept list after ingestion
                    else:
                        st.error(message)
                except Exception as e:
                    st.error(f"An unexpected error occurred during ingestion: {e}")

# --- 2. Knowledge Browser ---
elif app_mode == "Knowledge Browser":
    st.header("🌍 Knowledge Browser")
    st.subheader("Explore Concepts in the Knowledge Graph")
    
    if not st.session_state.all_concepts:
        st.info("No concepts found. Ingest some data first.")
        if st.button("Refresh Concept List"):
            refresh_concept_list()
    else:
        selected_concept_name = st.selectbox("Select a concept to view details:", options=st.session_state.all_concepts)
        
        if selected_concept_name and graph_handler:
            with st.spinner(f"Fetching details for {selected_concept_name}..."):
                # get_concept_features now returns a dict with 'properties' and 'relationships'
                concept_details = graph_handler.get_concept_features(selected_concept_name) 
                
                if concept_details:
                    st.subheader(f"Details for: {concept_details.get('name', selected_concept_name)}")
                    
                    # Display properties
                    if concept_details.get("properties"):
                        st.markdown("**Properties:**")
                        props = concept_details["properties"]
                        prop_list = [f"- **{key.capitalize()}**: {value}" for key, value in props.items()]
                        st.markdown("\n".join(prop_list))
                    else:
                        st.markdown("No properties found for this concept.")

                    # Display relationships
                    if concept_details.get("relationships"):
                        st.markdown("**Relationships:**")
                        rels_df = []
                        for rel in concept_details["relationships"]:
                            if rel.get('direction') == 'outgoing':
                                rel_display = f"`{concept_details['name']}` `-{rel['type']}->` `{rel.get('target_concept', 'Unknown')}`"
                            elif rel.get('direction') == 'incoming':
                                rel_display = f"`{rel.get('source_concept', 'Unknown')}` `-{rel['type']}->` `{concept_details['name']}`"
                            else:
                                rel_display = "Unknown relationship structure"
                            
                            rel_props_str = ", ".join([f"{k}: {v}" for k,v in rel.get('properties', {}).items()])
                            rels_df.append({"Relationship": rel_display, "Properties": rel_props_str if rel_props_str else "N/A"})
                        
                        if rels_df:
                            st.table(rels_df)
                        else:
                            st.markdown("No relationships with allowed types found for this concept within the display limit.")
                    else:
                        st.markdown("No relationships found for this concept.")
                else:
                    st.warning(f"Could not retrieve details for {selected_concept_name}.")
        elif not graph_handler:
            st.error("Graph Handler not available.")

# --- 3. Conceptual Blending ---
elif app_mode == "Conceptual Blending":
    st.header("✨ Conceptual Blending")
    st.subheader("Create Novel Concepts by Blending Two Existing Ones")

    if not st.session_state.all_concepts or len(st.session_state.all_concepts) < 2:
        st.info("You need at least two concepts in the knowledge graph to perform blending. Ingest some data first.")
        if st.button("Refresh Concept List"):
            refresh_concept_list()
    else:
        col1, col2 = st.columns(2)
        with col1:
            concept1_name = st.selectbox("Select First Concept:", options=st.session_state.all_concepts, key="blend_c1")
        with col2:
            available_concepts_c2 = [c for c in st.session_state.all_concepts if c != concept1_name]
            concept2_name = st.selectbox("Select Second Concept:", options=available_concepts_c2, key="blend_c2")

        if concept1_name and concept2_name and concept1_name != concept2_name:
            st.markdown("--- ")
            if st.button("🔮 Blend Concepts"):
                if not graph_handler or not text_processor:
                    st.error("Graph Handler or Text Processor not available for blending.")
                else:
                    with st.spinner(f"Blending '{concept1_name}' and '{concept2_name}'..."):
                        try:
                            # Fetch context for blending
                            context1 = get_concept_context(graph_handler, concept1_name)
                            context2 = get_concept_context(graph_handler, concept2_name)

                            if not context1 or not context2:
                                st.error("Could not retrieve sufficient context for one or both concepts.")
                            else:
                                blend_result = blend_concepts_gemini(
                                    text_processor.get_model(), # Pass the Gemini model instance
                                    context1, 
                                    context2, 
                                    concept1_name, 
                                    concept2_name
                                )
                                if blend_result:
                                    st.subheader(f"Blended Concept: {blend_result.get('name', 'Unnamed Blend')}")
                                    st.markdown(f"**Description:** {blend_result.get('description', 'No description provided.')}")
                                    
                                    features = blend_result.get('features', [])
                                    if features:
                                        st.markdown("**Key Features:**")
                                        for feature in features:
                                            st.markdown(f"- {feature}")
                                    else:
                                        st.markdown("No specific features identified for the blend.")
                                    
                                    emergent_props = blend_result.get('emergent_properties', [])
                                    if emergent_props:
                                        st.markdown("**Emergent Properties:**")
                                        for prop in emergent_props:
                                            st.markdown(f"- {prop}")
                                    
                                    # Option to add blended concept to graph (placeholder)
                                    st.markdown("--- ")
                                    st.info("Future Work: Add option to formally ingest this blended concept into the Knowledge Graph.")

                                    # Display raw contexts used for transparency
                                    with st.expander("Show Raw Context Used for Blending"):
                                        st.json({"concept1_context": context1, "concept2_context": context2})
                                else:
                                    st.error("Failed to generate blend. The model might not have returned a valid result.")
                        except Exception as e:
                            st.error(f"An error occurred during blending: {e}")
        elif concept1_name and concept2_name and concept1_name == concept2_name:
            st.warning("Please select two different concepts for blending.")

# --- 4. Analogical Search ---
elif app_mode == "Analogical Search":
    st.header("🔗 Analogical Search")
    st.subheader("Find Concepts Analogous to a Query Concept")

    if not st.session_state.all_concepts:
        st.info("No concepts in the knowledge graph to search. Ingest some data first.")
        if st.button("Refresh Concept List"):
            refresh_concept_list()
    else:
        query_concept_name = st.selectbox("Select a concept to find analogies for:", options=st.session_state.all_concepts, key="analogy_query")
        
        search_type = st.radio("Select Analogy Search Type:", ("Vector-based", "Structural (Graph-based)"), key="analogy_type")
        
        num_analogies = st.slider("Number of analogies to find:", min_value=1, max_value=10, value=3, key="analogy_k")

        if query_concept_name:
            if st.button("🔍 Find Analogies"):
                if not graph_handler or not faiss_handler or not text_processor:
                    st.error("One or more core services (Graph, FAISS, Text Processor) are not available for analogy search.")
                else:
                    with st.spinner(f"Searching for analogies to '{query_concept_name}'..."):
                        analogies = []
                        try:
                            if search_type == "Vector-based":
                                analogies = find_analogous_concepts_vector(
                                    query_concept_name=query_concept_name,
                                    faiss_handler=faiss_handler,
                                    graph_handler=graph_handler,
                                    k=num_analogies,
                                    excluded_concept_names=[query_concept_name] 
                                )
                            elif search_type == "Structural (Graph-based)":
                                analogies = find_analogous_concepts_structural_graph(
                                    query_concept_name=query_concept_name,
                                    graph_handler=graph_handler,
                                    k=num_analogies,
                                    excluded_concept_names=[query_concept_name]
                                )
                            
                            if analogies:
                                st.subheader("Found Analogies:")
                                for i, ana in enumerate(analogies):
                                    col1, col2 = st.columns([3,1])
                                    with col1:
                                        st.markdown(f"**{i+1}. {ana['name']}** (Domain: {ana.get('domain', 'N/A')}) - Score: {ana.get('score', 'N/A')}")
                                        if 'reason' in ana:
                                            st.caption(f"Reason: {ana['reason']}")
                                    
                                    with col2:
                                        button_key = f"explain_analogy_{i}_{ana['name'].replace(' ','_')}"
                                        if st.button("Explain Analogy", key=button_key):
                                            with st.spinner("Generating explanation..."):
                                                # Fetch contexts for explanation
                                                ctx1 = get_concept_context(graph_handler, query_concept_name)
                                                ctx2 = get_concept_context(graph_handler, ana['name'])
                                                explanation = generate_analogy_explanation_gemini(
                                                    query_concept_name,
                                                    ana['name'],
                                                    ctx1,
                                                    ctx2,
                                                    analogy_type=search_type.lower().split('(')[0].strip() # "vector" or "structural"
                                                )
                                                if explanation:
                                                    st.info(f"**Explanation for analogy between '{query_concept_name}' and '{ana['name']}':**\n{explanation}")
                                                else:
                                                    st.warning("Could not generate explanation.")
                            else:
                                st.info(f"No {search_type.lower()} analogies found for '{query_concept_name}'.")
                        except Exception as e:
                            st.error(f"An error occurred during analogy search: {e}")
                            st.exception(e) # Show full traceback for debugging

# --- Footer or common elements ---
st.sidebar.markdown("--- ")
st.sidebar.info("PolyMind V2 - Alpha")
if st.sidebar.button("Refresh Concept List Globally"):
    refresh_concept_list()
    st.rerun() 