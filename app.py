import streamlit as st
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification

st.set_page_config(page_title="Child Development Fact Checker")

MODEL_NAME = "poojasrisunkara/child-dev-bert"

@st.cache_resource
def load_model():
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

st.title("🧒 Child Development Fact vs Myth Checker")
st.write(
    "Enter a statement related to child development to check whether it is a **FACT** or a **MYTH**."
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
            prediction = torch.argmax(outputs.logits, dim=1).item()

        if prediction == 0:
            st.success("✅ FACT")
        else:
            st.error("❌ MYTH")

st.markdown("---")
st.caption(
    "⚠️ This tool is for educational purposes and may not be correct in all cases."
)


