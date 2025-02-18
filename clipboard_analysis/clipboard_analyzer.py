import sqlite3
import pandas as pd
import openai
from typing import List, Dict
import os
from dotenv import load_dotenv
import json
import dspy
import streamlit as st
import logging
from typing import Literal


class TextClassify(dspy.Signature):
    """Classify text content type and technical level."""
    text: str = dspy.InputField()
    document_type: Literal[
            "polished technical note",
            "unpolished technical note",
            "technical discussion",
            "interview question guide",
            "customer support communication",
            "performance conversation related",
            "source code",
            "other or unknown",
            "peer discussion", 
            "status reporting",
        ] = dspy.OutputField()
    tags: list = dspy.OutputField()
    confidence: float = dspy.OutputField()
    summary: str = dspy.OutputField()

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

@st.cache_data
def read_clipboard_data() -> pd.DataFrame:
    """Read data from the SQLite database with Streamlit caching."""
    logger.info("Reading clipboard data from database")
    try:
        db_path = "./clipboard.alfdb"
        conn = sqlite3.connect(db_path)

        query = "SELECT *, LENGTH(item) as content_length FROM clipboard WHERE app NOT LIKE '%code%' AND app NOT LIKE '%Code%'"
        df = pd.read_sql_query(query, conn)
        logger.info(f"Successfully read {len(df)} rows from database")

        conn.close()
        return df
    except Exception as e:
        logger.error(f"Error reading clipboard data: {str(e)}")
        raise

@st.cache_data
def classify_text(text: str, model_name: str = "gpt-4o-mini") -> Dict:
    """Classify text using DSPy-style classification with Streamlit caching."""
    logger.info(f"Starting text classification using model: {model_name}")
    from typing import Literal


    classify = dspy.Predict(TextClassify)

    if not text or len(str(text).strip()) < 150:
        logger.warning("Text too short for classification")
        return {
            "document_type": "unknown",
            "tags": [],
            "confidence": 0.0,
        }

    try:
        
        logger.info("Running classification")
        response = classify(text=text)
        
        result = {
            'document_type': response.document_type,
            'tags': response.tags,
            'confidence': response.confidence
        }
        logger.info(f"Classification complete: {json.dumps(result)}")
        return result

    except Exception as e:
        logger.error(f"Error processing text: {str(e)}")
        return {"document_type": "error", "tags": [], "confidence": 0.0}



@st.cache_data
def classify_text_iterative(
    text: str,
    model_name: str = "gpt-4o-mini",
    initial_window: int = 1000,
    step: int = 1000,
    confidence_threshold: float = 0.8,
    max_chars: int = 20000,
) -> Dict:
    """
    Iteratively classifies text by starting with the first chunk (initial_window)
    and expanding with the previous summary plus the next chunk until the confidence
    meets the threshold or max_chars is reached.
    """
    current_window = min(len(text), initial_window)
    current_text = text[:current_window]
    iterations = 1
    result =dict()
    while result.get("confidence", 0.0) < confidence_threshold and current_window < min(len(text), max_chars):
        result = classify_text(current_text, model_name=model_name)
        logger.info(f"Current window: {current_window}, max chars: {max_chars}, iterations: {iterations}")
        summary = result.get("summary", "")
        next_window_end = min(len(text), current_window + step)
        next_chunk = text[current_window:next_window_end]
        new_input = summary + "\n" + next_chunk
        current_window = next_window_end
        result = classify_text(new_input, model_name=model_name)
        iterations += 1
    logger.info(f"completed {iterations} iterations out of {max_chars/len(text)} chars")
    return result



def finetune_classification(df, teacher_model: str = "gpt-4o", student_model:str= "gpt-4o-mini"):
    """
    Finetunes a classification model using a teacher-student approach with DSPy.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the data to be used for finetuning.
        teacher_model (str): Name of the teacher model to use
        student_model (str): Name of the student model to use

    Returns:
        dict: Statistics about the finetuning process
    """
    from dspy import BootstrapFinetune # type: ignore
    logger.info(f"Starting finetuning with teacher={teacher_model} and student={student_model}")
    # Define the classification signature (reusing existing TextClassify)
    classify = dspy.Predict(TextClassify)
    teacher_classify = classify.deepcopy()
    student_classify = classify.deepcopy()
    
    # Configure teacher model
    teacher_lm = dspy.LM(f"openai/{teacher_model}", api_key=os.getenv("OPENAI_API_KEY"))
    
    # Configure student model (local model via Ollama)
    student_lm = dspy.LM(
        model="ollama_chat/deepseek-r1:1.5b",
        api_base="http://localhost:11434"
    )
    teacher_classify.set_lm(teacher_lm)
    student_classify.set_lm(student_lm)
    
    # Prepare training data
    train_data = []
    for _, row in df.iterrows():
        if pd.isna(row['item']) or len(str(row['item']).strip()) < 150:
            continue
        train_data.append(dspy.Example(
            text=row['item'],
            document_type="unknown",  # Will be filled by teacher
            tags=[],
            confidence=0.0,
            summary=""
        ).with_inputs('text'))
    
    # Initialize the bootstrapping process
    dspy.settings.configure(lm=teacher_lm)
    bootstrapper = BootstrapFinetune(
        student_lm=student_lm,
        metric=dspy.metrics.answer_exact_match,
        max_bootstrapped_demos=10,
        max_labeled_demos=50,
        max_rounds=3
    )
    # classify_f = bootstrapper.compile(student_classify, teacher=teacher_classify, trainset=train_data)
    
    try:
        # Set random seed for reproducibility
        import numpy as np
        np.random.seed(42)
        
        # Calculate approximate token counts for stratification
        token_counts = [len(str(example.text).split()) for example in train_data]
        
        # Create bins for token counts (small, medium, large)
        bins = [0, 100, 500, 1000, float('inf')]
        labels = pd.cut(token_counts, bins=bins, labels=['xs', 'sm', 'md', 'lg','xl'])
        
        # Stratified sampling for train and validation sets
        train_size = 100
        val_size = 20
        train_indices = []
        val_indices = []
        
        for size_label in ['small', 'medium', 'large']:
            size_indices = np.where(labels == size_label)[0]
            n_samples = len(size_indices)
            n_train = int(train_size * (n_samples / len(train_data)))
            n_val = int(val_size * (n_samples / len(train_data)))
            
            # Shuffle indices for this stratum
            shuffled = np.random.permutation(size_indices)
            train_indices.extend(shuffled[:n_train])
            val_indices.extend(shuffled[n_train:n_train + n_val])
        
        logger.info("Starting bootstrap training with stratified samples")
        trained_program = bootstrapper.bootstrap(
            task=TextClassify,
            trainset=[train_data[i] for i in train_indices],
            valset=[train_data[i] for i in val_indices]
        )
        
        # Log training statistics
        stats = {
            'num_examples': len(train_data),
            'num_rounds': bootstrapper.max_rounds,
            'teacher_model': teacher_model,
            'student_model': student_model
        }
        
        logger.info(f"Finetuning completed: {json.dumps(stats)}")
        return stats
        
    except Exception as e:
        logger.error(f"Error during finetuning: {str(e)}")
        return {
            'error': str(e),
            'status': 'failed'
        }

    
