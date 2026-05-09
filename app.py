import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# =========================
# PAGE SETTINGS
# =========================

st.set_page_config(
    page_title="AI Emotion Detection",
    page_icon="😊"
)

st.title("😊 AI Emotion Detection App")

st.write("Enter a sentence and AI will detect the emotion.")

# =========================
# TRAINING DATA
# =========================

training_data = [

    # HAPPY
    ("I am very happy", "happy"),
    ("Today is amazing", "happy"),
    ("I feel great", "happy"),
    ("I am excited", "happy"),
    ("This made me smile", "happy"),
    ("I am feeling awesome", "happy"),
    ("This is fantastic", "happy"),
    ("I feel wonderful", "happy"),

    # SAD
    ("I am very sad", "sad"),
    ("I feel lonely", "sad"),
    ("I am depressed", "sad"),
    ("This is terrible", "sad"),
    ("I want to cry", "sad"),
    ("Life feels bad", "sad"),
    ("I feel empty", "sad"),
    ("I feel hopeless", "sad"),

    # ANGRY
    ("I am angry", "angry"),
    ("I hate this", "angry"),
    ("I am frustrated", "angry"),
    ("This makes me mad", "angry"),
    ("I am irritated", "angry"),
    ("This is annoying", "angry"),
    ("I lost my temper", "angry"),
    ("I feel rage", "angry"),

    # FEAR
    ("I am scared", "fear"),
    ("I feel nervous", "fear"),
    ("I am worried", "fear"),
    ("This is frightening", "fear"),
    ("I am anxious", "fear"),
    ("I feel unsafe", "fear"),
    ("I am terrified", "fear"),
    ("I am stressed", "fear"),

    # SURPRISE
    ("Wow this is unexpected", "surprise"),
    ("I am shocked", "surprise"),
    ("This surprised me", "surprise"),
    ("I cannot believe this", "surprise"),
    ("What a surprise", "surprise"),
    ("This is astonishing", "surprise"),
    ("Oh my god", "surprise"),
    ("This is unbelievable", "surprise"),

    # NEUTRAL
    ("I am going to school", "neutral"),
    ("Today is Monday", "neutral"),
    ("I am eating food", "neutral"),
    ("I am reading a book", "neutral"),
    ("The weather is normal", "neutral"),
    ("I am sitting in class", "neutral"),
    ("I am doing homework", "neutral"),
    ("I woke up early", "neutral")
]

# =========================
# SEPARATE TEXTS & LABELS
# =========================

texts = [item[0] for item in training_data]
labels = [item[1] for item in training_data]

# =========================
# VECTORIZE TEXT
# =========================

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(texts)

# =========================
# TRAIN MODEL
# =========================

model = LogisticRegression(max_iter=1000)

model.fit(X, labels)

# =========================
# USER INPUT
# =========================

user_input = st.text_area("Enter your sentence:")

# =========================
# DETECT BUTTON
# =========================

if st.button("Detect Emotion"):

    if user_input.strip() == "":
        st.warning("Please enter some text.")

    else:

        input_data = vectorizer.transform([user_input])

        prediction = model.predict(input_data)[0]

        emoji = {
            "happy": "😊",
            "sad": "😢",
            "angry": "😠",
            "fear": "😨",
            "surprise": "😲",
            "neutral": "😐"
        }

        st.success(f"Detected Emotion: {prediction.upper()} {emoji[prediction]}")

        st.subheader("📋 Report")

        st.write("User Input:")
        st.info(user_input)

        st.write("Predicted Emotion:")
        st.info(prediction.upper())