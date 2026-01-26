import streamlit as st
import torch
import torch.nn.functional as F
from transformers import BertTokenizerFast, BertForSequenceClassification

st.set_page_config(page_title="Child Development Fact Checker")

MODEL_NAME = "poojasrisunkara/child-dev-bert"
CONFIDENCE_THRESHOLD = 0.65

@st.cache_resource
def load_model():
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

st.title("🧒 Child Development Fact vs Myth Checker")
st.write(
    "Enter a statement related to child development to check whether it is a **FACT**, "
    "**MYTH**, or **UNCERTAIN** based on model confidence."
)

user_input = st.text_area("Enter statement here:")

if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter a statement.")
    else:
        inputs = tokenizer(
            user_input,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
            confidence, prediction = torch.max(probs, dim=1)

        confidence = confidence.item()
        prediction = prediction.item()

        if confidence < CONFIDENCE_THRESHOLD:
            st.warning(f"⚠️ UNCERTAIN (confidence: {confidence:.2f})")
            st.write(
                "The model is not confident enough to classify this statement. "
                "This usually happens for vague, opinion-based, or ambiguous inputs."
            )
        else:
            if prediction == 0:
                st.success(f"✅ FACT (confidence: {confidence:.2f})")
            else:
                st.error(f"❌ MYTH (confidence: {confidence:.2f})")

st.markdown("---")
st.caption(
    "⚠️ This tool is for educational purposes. It uses a machine learning model and "
    "may be incorrect for vague, moral, or context-dependent statements."
)


