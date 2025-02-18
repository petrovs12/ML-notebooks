import streamlit as st
import pandas as pd
from clipboard_analyzer import read_clipboard_data, classify_text, classify_text_iterative
from clipboard_embeddings import ClipboardEmbeddings
import logging
import datetime
import os
import dspy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('clipboard_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Enable autoreload
st.set_page_config(
    page_title="Clipboard Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Model Configuration Section
def configure_model():
    """Configure and return the LLM based on user selection."""
    st.sidebar.header("Model Configuration")
    
    # Model provider selection
    provider = st.sidebar.selectbox(
        "Select Model Provider",
        options=["OpenAI", "Ollama Local"],
        index=0
    )
    
    if provider == "OpenAI":
        model_name = st.sidebar.selectbox(
            "Select OpenAI Model",
            options=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo", "gpt-4-turbo-preview"],
            index=0
        )
        lm = dspy.LM(f"openai/{model_name}", api_key=os.getenv("OPENAI_API_KEY"))
    else:
        model_name = st.sidebar.selectbox(
            "Select Local Model",
            options=["deepseek-r1:1.5b","deepseek-r1:7b","llama3.2",],
            index=0
        )
        lm = dspy.LM(
            f"ollama_chat/{model_name}",
            api_base="http://localhost:11434"
        )
    dspy.configure(lm=lm)
    
    # Store configuration in session state
    if 'model_config' not in st.session_state:
        st.session_state.model_config = {}
    
    st.session_state.model_config.update({
        'provider': provider,
        'model_name': model_name,
        'lm': lm
    })
    
    return model_name

# Configure model and get selected model name
selected_model = configure_model()

# Classification settings
use_iterative = st.sidebar.checkbox(
    "Use Iterative Classification",
    help="Process longer texts iteratively with summaries for better accuracy"
)
if use_iterative:
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.8,
        step=0.1,
        help="Minimum confidence level to stop iterative classification"
    )
logger.info(f"Selected model configuration: Provider={st.session_state.model_config['provider']}, Model={selected_model}, Iterative={use_iterative}")

# Main content
st.title("Clipboard Data Explorer")

# Load and display data
logger.info("Loading clipboard data...")
df = read_clipboard_data()
logger.info(f"Loaded {len(df)} clipboard entries")
st.write(f"Total entries: {len(df)}")

# Initialize embeddings if not already in session state
if 'embeddings_handler' not in st.session_state:
    # Check if embeddings database exists
    db_path = os.path.join(os.path.dirname(__file__), "chroma_db")
    db_exists = os.path.exists(db_path)

    
    if db_exists:
        logger.info("Found existing embeddings database")
        st.info("Using existing embeddings database")
        # Create handler using existing db
        st.session_state.embeddings_handler = ClipboardEmbeddings()
        
    else:
        with st.spinner("Initializing embeddings... This may take a few minutes."):
            st.session_state.embeddings_handler = ClipboardEmbeddings(force_recreate=True)
            st.session_state.embeddings_handler.initialize_embeddings(df)
        st.success("Embeddings initialized successfully!")
    

# Add button to recreate database
if st.sidebar.button("Recreate Embeddings Database"):
    with st.spinner("Reinitializing embeddings... This may take a few minutes."):
        st.session_state.embeddings_handler = ClipboardEmbeddings(force_recreate=True)
        st.session_state.embeddings_handler.initialize_embeddings(df)
    st.success("Embeddings reinitialized successfully!")

# Add semantic search in sidebar
st.sidebar.header("Semantic Search")
semantic_query = st.sidebar.text_input(
    "Search similar content",
    help="Enter text to find semantically similar clipboard items"
)

if semantic_query:
    with st.spinner("Searching similar items..."):
        similar_items = st.session_state.embeddings_handler.find_similar(semantic_query)
        logger.info(f"Similar items: {similar_items}")
        
        st.sidebar.subheader("Similar Items")
        # Group results by original document ID
        results_by_doc = {}
        for idx, (doc, meta, score, chunk_id) in enumerate(zip(
            similar_items['documents'][0],
            similar_items['metadatas'][0],
            similar_items['distances'][0],
            similar_items['ids'][0]
        )):
            # Safely extract original document ID and chunk index
            try:
                id_parts = chunk_id.split('-')
                orig_id = id_parts[0]
                chunk_idx = int(id_parts[1]) if len(id_parts) > 1 else 0
            except (IndexError, ValueError):
                # If ID doesn't match expected format, use the whole ID as original
                orig_id = chunk_id
                chunk_idx = 0
            
            if orig_id not in results_by_doc:
                results_by_doc[orig_id] = {
                    'chunks': [],
                    'best_score': score
                }
            results_by_doc[orig_id]['chunks'].append({
                'text': doc,
                'score': score,
                'chunk_index': chunk_idx,
                'total_chunks': meta.get('total_chunks', 1)
            })
        
        # Display results grouped by original document
        for idx, (orig_id, result) in enumerate(sorted(
            results_by_doc.items(),
            key=lambda x: x[1]['best_score']
        )):
            # Get the original document from the dataframe
            orig_doc = df.iloc[int(orig_id)]['item']
            with st.sidebar.expander(
                f"Result {idx + 1} (Similarity: {1 - result['best_score']:.3f})"
            ):
                # Show the matching chunks first
                if len(result['chunks']) > 1:
                    st.write("🔍 Matching sections:")
                    for chunk in sorted(result['chunks'], key=lambda x: x['chunk_index']):
                        st.markdown(f"""
                        **Chunk {chunk['chunk_index'] + 1}/{chunk['total_chunks']}** *(Similarity: {1 - chunk['score']:.3f})*
                        ```
                        {chunk['text']}
                        ```
                        ---
                        """)
                
                # Show the full document
                st.write("📄 Full content:")
                st.write(orig_doc[:200] + "..." if len(orig_doc) > 200 else orig_doc)

# Data filters
col1, col2,col3 = st.columns(3)
with col1:
    min_length = st.number_input("Minimum Content Length", 
                                min_value=0, 
                                max_value=df['content_length'].max(),
                                value=0)
with col2:
    max_length = st.number_input("Maximum Content Length",
                                min_value=0,
                                max_value=df['content_length'].max(),
                                value=int(df['content_length'].max()))
with col3:
    # Create a list of unique words from all items for autocomplete
    all_words = set()
    for text in df['item'].str.split():
        all_words.update([word.lower() for word in text if len(word) > 3])
    search_text = st.text_input("Search in content", 
                               value="",
                               help="Type to search. Suggestions will appear as you type.")
    if search_text:
        suggestions = [word for word in all_words if search_text.lower() in word.lower()][:5]
        if suggestions:
            selected_suggestion = st.selectbox("Suggestions:", ["Keep current"] + suggestions)
            if selected_suggestion != "Keep current":
                search_text = selected_suggestion

# Filter the dataframe
logger.info(f"Applying filters - min_length: {min_length}, max_length: {max_length}, search_text: '{search_text}'")
filtered_df = df[
    (df['content_length'] >= min_length) & 
    (df['content_length'] <= max_length)
]
if search_text:
    filtered_df = filtered_df[filtered_df['item'].str.contains(search_text, case=False, na=False)]

logger.info(f"Filtered dataframe contains {len(filtered_df)} entries")

# Display filtered dataframe with selection
# Keep the index visible and use it for selection
display_df = filtered_df.reset_index()

edited_rows = st.dataframe(
    display_df[['index', 'item', 'content_length']],
    use_container_width=True,
    column_config={
        "index": st.column_config.NumberColumn(
            "Original Index",
            required=True,
            help="Original position in the dataset"
        ),
        "item": st.column_config.TextColumn(
            "Content",
            width="large",
            help="The clipboard content"
        ),
        "content_length": st.column_config.NumberColumn(
            "Length",
            help="Length of the content in characters"
        )
    },
    hide_index=True,
    column_order=["index", "item", "content_length"],
    on_select='rerun'
)

# Classify selected text
if st.button("Classify Selected Text"):
    selected_indices = edited_rows.selection.rows
    logger.info(f"Selected rows: {selected_indices}")
    
    if not selected_indices:
        st.warning("Please select a row to classify.")
        logger.info("No rows selected")
    else:
        try:
            # Get the first selected row
            selected_idx = selected_indices[0]
            original_idx = display_df.iloc[selected_idx]['index']  # Get the original index
            logger.info(f"Selected row {selected_idx} with original index {original_idx}")
            
            # Get the text from the original dataframe using the original index
            selected_text = df.iloc[original_idx]['item']
            logger.info(f"Processing selection - Content length: {len(selected_text)}")
            logger.info(f"Selected text preview: {selected_text[:100]}...")
            
            with st.spinner("Classifying text..."):
                try:
                    # Configure DSPy with selected model
                    dspy.configure(lm=st.session_state.model_config['lm'])
                    
                    if use_iterative:
                        classification = classify_text_iterative(
                            selected_text,
                            model_name=selected_model,
                            confidence_threshold=confidence_threshold
                        )
                    else:
                        classification = classify_text(selected_text, selected_model)
                        
                    logger.info(f"Classification results - Type: {classification['document_type']}, Confidence: {classification['confidence']}")
                    
                    st.subheader("Classification Results")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Document Type", classification['document_type'])
                    with col2:
                        st.metric("Confidence", f"{classification['confidence']:.2f}")
                    with col3:
                        st.write("Tags")
                        st.write(classification['tags'])
                        
                    if 'summary' in classification:
                        st.write("Summary:")
                        st.write(classification['summary'])
                except Exception as e:
                    error_msg = f"Error during classification: {str(e)}"
                    logger.error(error_msg)
                    st.error(error_msg)
        except Exception as e:
            error_msg = f"Error processing selection: {str(e)}"
            logger.error(error_msg)
            st.error(error_msg)

# Add some styling
st.markdown("""
    <style>
    .stDataFrame {
        font-size: 14px;
    }
    </style>
""", unsafe_allow_html=True)

def initialize_embeddings(df: pd.DataFrame) -> ClipboardEmbeddings:
    """Initialize embeddings for all items in the clipboard database."""
    try:
        handler = ClipboardEmbeddings()
        
        # Check if we need to initialize
        if handler.collection.count() == 0:
            logger.info("Empty collection found, initializing embeddings")
            handler.initialize_embeddings(df)
        else:
            logger.info("Using existing embeddings")
            
        return handler
    except Exception as e:
        logger.error(f"Error in initialize_embeddings: {str(e)}")
        raise
