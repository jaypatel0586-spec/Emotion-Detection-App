import streamlit as st
from transformers import pipeline

# =========================
# PAGE SETTINGS
# =========================

st.set_page_config(
    page_title="Advanced AI Emotion Detection",
    page_icon="🤖"
)

st.title("🤖 Advanced AI Emotion Detection")

st.write("Enter any sentence and AI will detect emotion automatically.")

# =========================
# LOAD AI MODEL
# =========================

classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None
)

# =========================
# USER INPUT
# =========================

user_input = st.text_area("Enter your sentence:")

# =========================
# BUTTON
# =========================

if st.button("Detect Emotion"):

    if user_input.strip() == "":
        st.warning("Please enter some text.")

    else:

        results = classifier(user_input)

        # Highest emotion
        best_result = max(results[0], key=lambda x: x['score'])

        emotion = best_result['label']
        confidence = best_result['score']

        emoji_dict = {
            "joy": "😊",
            "sadness": "😢",
            "anger": "😠",
            "fear": "😨",
            "surprise": "😲",
            "love": "❤️",
            "neutral": "😐"
        }

        emoji = emoji_dict.get(emotion.lower(), "🤖")

        # MAIN RESULT
        st.success(
            f"Detected Emotion: {emotion.upper()} {emoji}"
        )

        # CONFIDENCE
        st.info(f"Confidence Score: {confidence:.2f}")

        # ALL EMOTIONS
        st.subheader("All Emotion Scores")

        for item in results[0]:

            label = item['label']
            score = item['score']

            st.write(f"{label} : {score:.2f}")