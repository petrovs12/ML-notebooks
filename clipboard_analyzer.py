import sqlite3
import pandas as pd
import openai
from typing import List, Dict
import os
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Configure OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')

def read_clipboard_data() -> pd.DataFrame:
    """Read data from the SQLite database."""
    db_path = './clipboard.alfdb'
    conn = sqlite3.connect(db_path)
    
    query = "SELECT * FROM clipboard"
    df = pd.read_sql_query(query, conn)
    
    conn.close()
    return df

def classify_text(text: str) -> Dict:
    """Classify text using DSPy-style classification to determine document type and technical score.
    
    Categories are determined based on content analysis with confidence scores.
    Only processes text longer than 150 characters.
    """
    from typing import Literal
    
    class TextClassify(dspy.Signature):
        """Classify text content type and technical level."""
        
        text: str = dspy.InputField()
        document_type: Literal[
            'technical note',
            'technical discussion',
            'interview question note', 
            'customer support communication',
            'performance conversation related',
            'source code',
            'other',
            'unknown'
        ] = dspy.OutputField()
        tags: list = dspy.OutputField()
        # technical_score: float = dspy.OutputField()
        confidence: float = dspy.OutputField()

    classify = dspy.Predict(TextClassify)
    
    if not text or len(str(text).strip()) < 150:  # Skip texts shorter than 150 chars
        return {
            'document_type': 'unknown',
            'tags': [],
            # 'technical_score': 0.0,
            'confidence': 0.0
        }
    
    try:
        lm = dspy.LM('openai/gpt-4o-mini', api_key=os.getenv('OPENAI_API_KEY'))
        dspy.configure(lm=lm)
        response = classify(text=text)
        
        # Parse the response content as JSON
        result = json.loads(response.choices[0].message.content)
        return result
    
    except Exception as e:
        print(f"Error processing text: {e}")
        return {
            'document_type': 'error',
            'tags': [],
            'confidence': 0.0
        }

def process_items(df: pd.DataFrame) -> pd.DataFrame:
    """Process each item in the dataframe and add classifications."""
    results = []
    
    for idx, row in df.iterrows():
        if idx % 10 == 0:  # Progress indicator
            print(f"Processing item {idx}...")
            
        classification = classify_text(row['item'])
        results.append(classification)
        
        # Periodically save results to temp file
        if idx % 50 == 0:  # Save every 50 items
            temp_df = df.copy()
            temp_results = results.copy()
            # Pad results list to match dataframe length
            temp_results.extend([{'document_type': '', 'tags': [], 'confidence': 0.0}] * (len(temp_df) - len(temp_results)))
            temp_df['document_type'] = [r['document_type'] for r in temp_results]
            temp_df['tags'] = [r['tags'] for r in temp_results]
            temp_df['classification_confidence'] = [r['confidence'] for r in temp_results]
            temp_df.to_csv('clipboard_analysis_temp.csv', index=False)
            print(f"Saved temporary results at item {idx}")
    # Create new columns from classification results
    df['document_type'] = [r['document_type'] for r in results]
    df['tags'] = [r['tags'] for r in results]
    df['classification_confidence'] = [r['confidence'] for r in results]
    
    return df

def main():
    # Read the data
    print("Reading database...")
    df = read_clipboard_data()
    print(f"Found {len(df)} entries in the database")
    
    # Process the data
    print("\nProcessing items with OpenAI classification...")
    enriched_df = process_items(df)
    
    # Save results
    output_file = 'clipboard_analysis_results.csv'
    enriched_df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    
    # Print summary
    print("\nDocument type distribution:")
    print(enriched_df['document_type'].value_counts())
    
    print("\nMost common tags:")
    all_tags = [tag for tags in enriched_df['tags'] for tag in tags]
    tag_counts = pd.Series(all_tags).value_counts()
    print(tag_counts.head(10))

if __name__ == "__main__":
    main() 