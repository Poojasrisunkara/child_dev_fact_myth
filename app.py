import streamlit as st
import torch
import torch.nn.functional as F
import re
from transformers import BertTokenizerFast, BertForSequenceClassification

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Child Development Fact Checker")

MODEL_NAME = "poojasrisunkara/child-dev-bert"
CONFIDENCE_THRESHOLD = 0.65

# ---------------- NORMALIZATION ----------------

REPLACEMENTS = {
    r"\bkids\b": "children",
    r"\bkid\b": "child",
    r"\bbabies\b": "babies",
    r"\bbaby\b": "baby",
}

HIGH_RISK_TOPICS = {
    "physical_punishment": [
        "beat", "beating", "hit", "hitting",
        "spank", "spanking", "punish", "punishment"
    ]
}

NEGATION_WORDS = ["not", "never", "no", "should not", "do not", "does not"]

QUESTION_PATTERNS = [
    (r"is that bad\??$", "this is bad"),
    (r"is it bad\??$", "this is bad"),
    (r"is this bad\??$", "this is bad"),
    (r"is it normal\??$", "this is normal"),
    (r"is this normal\??$", "this is normal"),
]

def normalize_question(text: str) -> str:
    text = text.lower().strip()
    for pattern, replacement in QUESTION_PATTERNS:
        text = re.sub(pattern, replacement, text)
    return text

def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)

    for pat, rep in REPLACEMENTS.items():
        text = re.sub(pat, rep, text)

    tokens = text.split()

    # Safe fragment completion
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

def contains_negation(text: str) -> bool:
    return any(n in text for n in NEGATION_WORDS)

# ---------------- MODEL LOAD ----------------

@st.cache_resource
def load_model():
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# ---------------- UI ----------------

st.title("🧒 Child Development Fact vs Myth Checker")
st.write(
    "Enter a statement related to child development. "
    "Questions and short inputs are automatically normalized for better accuracy."
)

user_input = st.text_area("Enter statement here:")

# ---------------- PREDICTION ----------------

if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter a statement.")
    else:
        # Step 1: Question normalization
        question_normalized = normalize_question(user_input)

        # Step 2: Text normalization
        normalized_input = normalize_text(question_normalized)

        # Step 3: Detect high-risk topic
        risk_topic = detect_high_risk(normalized_input)
        has_negation = contains_negation(normalized_input)

        # Step 4: Tokenize
        inputs = tokenizer(
            normalized_input,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        # Step 5: Predict
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
            confidence, prediction = torch.max(probs, dim=1)

        confidence = confidence.item()
        prediction = prediction.item()  # 0 = FACT, 1 = MYTH

        # ---------------- SAFETY OVERRIDE ----------------
        if risk_topic == "physical_punishment":
            if has_negation:
                prediction = 0  # FACT
                confidence = max(confidence, 0.85)
            else:
                prediction = 1  # MYTH
                confidence = max(confidence, 0.85)

        # ---------------- UNCERTAIN LOGIC ----------------
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

        # ---------------- TRANSPARENCY ----------------
        with st.expander("See how your input was interpreted"):
            st.write("**Original input:**")
            st.code(user_input)
            st.write("**After question normalization:**")
            st.code(question_normalized)
            st.write("**Final normalized input:**")
            st.code(normalized_input)

# ---------------- FOOTER ----------------

st.markdown("---")
st.caption(
    "⚠️ Educational use only. "
    "User questions are normalized into declarative statements before classification. "
    "High-risk child welfare topics use negation-aware safety rules."
)




