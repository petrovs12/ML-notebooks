import dspy
from dspy.teleprompt import BootstrapFewShot
import streamlit as st
from typing import Optional
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure models
class ModelConfig:
    def __init__(self):
        # Initialize Claude model
        self.claude = dspy.OpenAI(
            model="gpt-4",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Initialize local model via Ollama
        self.local_model = dspy.OllamaLocal(
            model="deepseek-coder:6.7b",
            api_base="http://localhost:11434"
        )

# Basic signature for classification
class SimpleClassifier(dspy.Signature):
    """Simple classifier signature."""
    input_text = dspy.InputField(desc="Text to classify")
    label = dspy.OutputField(desc="Classification label")

# Initialize Streamlit interface
def init_streamlit():
    st.title("DSPy Model Comparison")
    st.write("Compare Claude and Local Model outputs")
    
    # Text input
    user_input = st.text_area("Enter text to classify:", "")
    
    # Model selection
    model_option = st.selectbox(
        "Select Model",
        ["Claude", "Local Model", "Both"]
    )
    
    return user_input, model_option

def main():
    # Initialize models
    models = ModelConfig()
    
    # Setup Streamlit
    user_input, model_option = init_streamlit()
    
    if st.button("Classify"):
        if not user_input:
            st.warning("Please enter some text to classify")
            return
            
        if model_option in ["Claude", "Both"]:
            with st.spinner("Getting Claude's response..."):
                dspy.settings.configure(lm=models.claude)
                classifier = SimpleClassifier()
                claude_result = classifier(input_text=user_input)
                st.write("Claude's classification:", claude_result.label)
        
        if model_option in ["Local Model", "Both"]:
            with st.spinner("Getting Local Model's response..."):
                dspy.settings.configure(lm=models.local_model)
                classifier = SimpleClassifier()
                local_result = classifier(input_text=user_input)
                st.write("Local Model's classification:", local_result.label)

if __name__ == "__main__":
    main() 