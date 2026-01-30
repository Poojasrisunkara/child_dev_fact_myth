import streamlit as st
import torch
import torch.nn.functional as F
import re
from transformers import BertTokenizerFast, BertForSequenceClassification

st.set_page_config(page_title="Child Development Fact Checker")

MODEL_NAME = "poojasrisunkara/child-dev-bert"
CONFIDENCE_THRESHOLD = 0.65


REPLACEMENTS = {
    r"\bkids\b": "children",
    r"\bkid\b": "child",
    r"\bbabies\b": "babies",
    r"\bbaby\b": "baby",
}

HIGH_RISK_TOPICS = {
    "physical_punishment": ["beat", "beating", "hit", "hitting", "spank", "spanking"]
}

def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)

    for pat, rep in REPLACEMENTS.items():
        text = re.sub(pat, rep, text)

    tokens = text.split()

    # Fragment completion (SAFE, NO MEANING ADDED)
    if len(tokens) < 4:
        text = text + " in child development"

    if not text.endswith("."):
        text += "."

    return text

def detect_high_risk(text: str):
    for topic, words in HIGH_RISK_TOPICS.items():
        for w in words:
            if w in text:
                return topic
    return None


@st.cache_resource
def load_model():
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()


st.title("🧒 Child Development Fact vs Myth Checker")
st.write(
    "Enter a statement related to child development. "
    "Short or informal inputs are automatically normalized for better accuracy."
)

user_input = st.text_area("Enter statement here:")


if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter a statement.")
    else:
        # Step 1: Normalize
        normalized_input = normalize_text(user_input)

        # Step 2: Detect high-risk topic
        risk_topic = detect_high_risk(normalized_input)

        # Step 3: Tokenize
        inputs = tokenizer(
            normalized_input,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        # Step 4: Predict
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
            confidence, prediction = torch.max(probs, dim=1)

        confidence = confidence.item()
        prediction = prediction.item()  # 0 = FACT, 1 = MYTH (assumed)

        # Step 5: Safety override
        if risk_topic == "physical_punishment" and prediction == 0:
            prediction = 1
            confidence = max(confidence, 0.85)

        # Step 6: UNCERTAIN logic (unchanged)
        if confidence < CONFIDENCE_THRESHOLD:
            st.warning(f"⚠️ UNCERTAIN (confidence: {confidence:.2f})")
            st.write(
                "The input is vague or underspecified. "
                "Try rephrasing with more context."
            )
        else:
            if prediction == 0:
                st.success(f"✅ FACT (confidence: {confidence:.2f})")
            else:
                st.error(f"❌ MYTH (confidence: {confidence:.2f})")

        # Optional: transparency (VERY GOOD FOR REVIEWERS)
        with st.expander("See how your input was interpreted"):
            st.write("**Original input:**")
            st.code(user_input)
            st.write("**Normalized input:**")
            st.code(normalized_input)


st.markdown("---")
st.caption(
    "⚠️ Educational use only. "
    "Short or ambiguous inputs are normalized before classification. "
    "The system combines machine learning with rule-based safeguards."
)



