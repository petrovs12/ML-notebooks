# Instructions for GitHub Copilot Assistance:
 These guidelines aim to shape Copilot’s suggestions within this machine learning repo.
 Place this file at the root of your repository or in a configuration directory
 to help guide Copilot’s completions and keep them aligned with the project’s goals.

# General Approach:
 - Prefer Python code examples, especially in Jupyter Notebook cells.
 - Use standard ML libraries (e.g., scikit-learn, TensorFlow, PyTorch, pandas, numpy, matplotlib, seaborn).
 - When suggesting model training code, provide small, self-contained examples.
 - When demonstrating theory, favor concise but clear explanations using inline comments.
 - Suggest data visualization approaches that pair well with the data manipulation steps shown.

# Data Handling:
 - Generate or load sample datasets in-memory or from common public datasets.
 - Recommend the use of pandas DataFrames for data exploration and preprocessing.
 - Show best practices for handling missing or categorical data.

# Visualizations:
 - Use matplotlib or seaborn for plotting.
 - Include code cells that show initial exploratory plots, distributions, and relationships between features.
 - Provide code snippets for common visualization patterns like histograms, scatter plots, correlation heatmaps.

# Modeling:
 - Suggest code for training baseline models (e.g., linear regression, logistic regression) before complex models.
 - Demonstrate model evaluation using train/test splits, cross-validation, and standard metrics.
 - Highlight methods to interpret models (e.g., feature importances, partial dependence plots).

# Best Practices:
 - Encourage well-commented code for theoretical explanations.
 - Suggest code cells dedicated to step-by-step data preprocessing, feature engineering, model training, evaluation, and visualization.
 - Prompt use of version control best practices (small, incremental changes; meaningful commit messages).

# Interactivity & Experimentation:
 - Propose using Jupyter widgets (ipywidgets) or interactive libraries (Plotly) for better data exploration.
 - Provide code templates to repeatedly run experiments, track hyperparameters, and store results.

# Environment:
 - When suggesting installation commands, prefer standard pip or conda installation steps.
 - Show how to install missing dependencies inline within notebooks (e.g., `!pip install package`).

# Theory & Learning:
 - Add brief inline comments summarizing key theoretical concepts where appropriate.
 - Suggest references or docstrings that link to official library documentation or well-known tutorials.

# Security & Privacy:
 - Do not suggest code that exposes credentials or sensitive information.
 - Promote safe handling of any personal or proprietary data.

# Following these instructions will help maintain a consistent, educational, and practical environment for learning ML concepts in this repository.