
# %%
# Setup and imports
import dspy
import os
from typing import Literal
import dspy.evaluate
import pandas as pd
from clipboard_analyzer import read_clipboard_data, TextClassify
import random
from dspy.teleprompt import BootstrapFewShot


import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("moo")

logger.info("Starting DeepScaler finetuning...")
# %%
# Define the classification signature
# class DeepScaler(dspy.Signature):
#     """Classify text for scaling-related content."""
#     text = dspy.InputField()
#     category: Literal['architecture', 'performance', 'capacity', 'reliability', 'unknown'] = dspy.OutputField(desc="One of: architecture, performance, capacity, reliability, unknown")
#     confidence: float = dspy.OutputField(desc="confidence score between 0 and 1")
#     key_points = dspy.OutputField(desc="key points from the text")

# %%
# Configure teacher model (GPT-4)
teacher_lm = dspy.LM(model="openai/gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

# Configure student model (local model via Ollama)
student_lm = dspy.LM(model="llama3.2", api_base="http://localhost:11434")




#%%

#%% create classification task
classify = dspy.Predict(TextClassify)
student_classify = classify.deepcopy()
student_classify.set_lm(student_lm)

teacher_classify = classify.deepcopy()
teacher_classify.set_lm(teacher_lm)

# %%
# Load and prepare data
logger.info("Loading clipboard data...")
df = read_clipboard_data()
logger.info(f"Loaded {len(df)} clipboard entries")

# %%
# Prepare training data
raw_data = []
for _, row in df.iterrows():
    if pd.isna(row["item"]) or len(str(row["item"]).strip()) < 150 or len(str(row["item"]).strip()) > 1000:
        continue
    raw_data.append(
        dspy.Example(
            text=row["item"],
            category="unknown",  # Will be filled by teacher
            confidence=0.0,
            key_points=[],
            summary="",
        ).with_inputs("text")
    )

# Shuffle the data
random.Random(42).shuffle(raw_data)

# Take first 100 examples for training
unlabeled_trainset = raw_data[:100]
logger.info(f"Prepared {len(unlabeled_trainset)} training examples")
metric = (lambda x, y, trace=None: x.label == y.label)

# %%
# Initialize the bootstrapping process
logger.info("Configuring teacher model...")
dspy.settings.configure(lm=teacher_lm)

# Create the basic classifier
dspy.settings.configure(experimental=True)

# Initialize bootstrapper
optimizer = dspy.BootstrapFewShot(metric==metric, max_bootstrapped_demos=10, max_labeled_demos=50, max_rounds=3)  

classify_ft = optimizer.compile(student_classify, teacher=teacher_classify, trainset=unlabeled_trainset)

# if you *do* have labels, pass metric=your_metric here!

# bootstrapper = BootstrapFewShot(
#     student_lm=student_lm,
#     metric=dspy.evaluate.metrics.answer_exact_match,
#     max_bootstrapped_demos=10,
#     max_labeled_demos=50,
#     max_rounds=3,
# )

# %%
# Training process
logger.info("Starting training process...")
# Train the student model
logger.info("Starting bootstrap training...")
classify_ft = optimizer.compile(
    task=TextClassify,
    trainset=unlabeled_trainset[:100],  # Start with subset
    valset=unlabeled_trainset[100:120],  # Small validation set
)

logger.info("Training complete!")


#%%

classify_ft(text="I didn't receive my money earlier and it says the transaction is still in progress. Can you fix it?")


# %%
# Test the trained model
def test_classification(text: str):
    """Test the trained model with a given text."""
    # Configure with student model for inference
    dspy.settings.configure(lm=student_lm)
    result = trained_program(text=text)
    return {
        "category": result.category,
        "confidence": result.confidence,
        "key_points": result.key_points,
        "summary": result.summary,
    }


# Test with sample texts
test_cases = [
    "Our distributed system needs to handle 10x more traffic during peak hours. We're seeing latency spikes and need to optimize the database queries.",
    "We need to implement redundancy and failover mechanisms to ensure high availability of our critical services.",
    "The current storage system can't handle the increasing data volume. We need to implement data sharding and archival strategies.",
]

for test in test_cases:
    logger.info(f"\nTesting with: {test[:100]}...")
    result = test_classification(test)
    logger.info(f"Result: {result}")
