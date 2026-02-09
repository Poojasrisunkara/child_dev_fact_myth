# child_dev_fact_myth
This Streamlit application classifies child-development statements as Fact or Myth using machine learning and natural language processing techniques. The goal is to help parents, educators, and caregivers quickly verify child-development information and reduce misinformation.

The app preprocesses user input through text normalization, including lowercasing, punctuation removal, stop-word filtering, and lemmatization. It then converts the text into numerical representations using TF-IDF for baseline models and BERT embeddings for contextual understanding.

Predictions are generated using a fine-tuned BERT model with confidence scoring, along with safety checks to handle sensitive child-development topics.
