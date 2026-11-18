
import streamlit as st
import joblib
import os
import re
import pandas as pd

# =========================================
# PAGE CONFIGURATION
# =========================================

st.set_page_config(
    page_title="NewsGuard AI",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "fake_news_model.pkl"
)

VECTORIZER_PATH = os.path.join(
    BASE_DIR, "models", "tfidf_vectorizer.pkl"
)

# =========================================
# CUSTOM DESIGN
# =========================================

st.markdown("""
<style>

.stApp {
    background: #F5F7FB;
    color: #172B4D;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    background: #14243D;
    border-right: 1px solid #243B5A;
}

[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}

h1, h2, h3 {
    color: #172B4D !important;
}

p {
    color: #52627A;
}

.hero {
    background: linear-gradient(120deg, #142B4D, #245B91);
    padding: 35px;
    border-radius: 18px;
    margin-bottom: 25px;
}

.hero h1 {
    color: white !important;
    font-size: 38px;
}

.hero p {
    color: #DFEAF7 !important;
    font-size: 17px;
}

.eyebrow {
    color: #8BE9F5;
    font-size: 12px;
    font-weight: bold;
    letter-spacing: 2px;
}

.card {
    background: white;
    border: 1px solid #E1E8F0;
    border-radius: 15px;
    padding: 23px;
    margin-bottom: 15px;
    box-shadow: 0 4px 15px rgba(20, 40, 80, 0.045);
}

.card h3 {
    color: #245B91 !important;
    margin-top: 0;
}

.card p {
    color: #52627A;
}

.stTextArea textarea {
    background: white !important;
    color: #172B4D !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 12px !important;
}

.stButton > button {
    background: linear-gradient(90deg, #2563EB, #0891B2);
    color: white !important;
    border: none;
    border-radius: 10px;
    min-height: 46px;
    font-weight: bold;
    width: 100%;
}

.stButton > button:hover {
    background: #1D4ED8;
    border: none;
}

[data-testid="stMetric"] {
    background: white;
    border: 1px solid #E1E8F0;
    border-radius: 13px;
    padding: 18px;
}

hr {
    border-color: #DFE6EF;
}

</style>
""", unsafe_allow_html=True)


# =========================================
# LOAD MODEL
# =========================================

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


# =========================================
# TEXT PREPROCESSING
# =========================================

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# =========================================
# REUSABLE PAGE HEADER
# =========================================

def page_header(label, title, description):
    st.markdown(f"""
    <div class="hero">
        <div class="eyebrow">{label}</div>
        <h1>{title}</h1>
        <p>{description}</p>
    </div>
    """, unsafe_allow_html=True)


# =========================================
# SIDEBAR NAVIGATION
# =========================================

st.sidebar.markdown("# 📰 NewsGuard AI")
st.sidebar.caption("INTELLIGENT NEWS ANALYSIS")
st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "🏠 Home",
        "🔍 Predict",
        "📊 Model Performance",
        "⚙️ How It Works",
        "ℹ️ About"
    ]
)

st.sidebar.divider()

st.sidebar.markdown("### 🤖 MODEL DETAILS")
st.sidebar.caption("Algorithm: Logistic Regression")
st.sidebar.caption("Vectorizer: TF-IDF")
st.sidebar.caption("Framework: Scikit-learn")

st.sidebar.divider()
st.sidebar.caption("Fake News Prediction using ML")


# =========================================
# HOME PAGE
# =========================================

if page == "🏠 Home":

    page_header(
        "ARTIFICIAL INTELLIGENCE • NLP",
        "NewsGuard AI",
        "An intelligent system to analyze news articles "
        "and identify patterns associated with fake and real news."
    )

    st.markdown("## Welcome to NewsGuard AI")

    st.write(
        "Explore how Machine Learning can help analyze "
        "news content and classify articles."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="card">
            <h3>01. Paste</h3>
            <p>Enter a news headline or complete article.</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="card">
            <h3>02. Analyze</h3>
            <p>The trained ML model analyzes the text patterns.</p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="card">
            <h3>03. Predict</h3>
            <p>Receive a Fake or Real classification and probabilities.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("## Project Highlights")

    col1, col2 = st.columns(2)

    with col1:
        st.info("🧠 Machine Learning: Logistic Regression")

    with col2:
        st.info("🔤 NLP: TF-IDF Text Vectorization")

    st.warning(
        "This is a machine learning classification project. "
        "It identifies learned writing patterns and does not "
        "independently verify facts."
    )

    st.markdown("### Ready to analyze news?")

    if st.button("🔍 Go to News Prediction"):
        st.session_state["navigation_page"] = "🔍 Predict"
        st.rerun()


# =========================================
# PREDICTION PAGE
# =========================================

elif page == "🔍 Predict":

    page_header(
        "AI NEWS ANALYZER",
        "Analyze a News Article",
        "Enter news content and let the trained model analyze it."
    )

    st.markdown("## 📝 News Input")

    news = st.text_area(
        "Paste your news article",
        height=250,
        placeholder="Enter your news headline and complete article here..."
    )

    st.caption(
        "Tip: A complete article usually provides more text "
        "for the model to analyze."
    )

    if st.button("🚀 ANALYZE NEWS"):

        if len(news.strip()) < 30:
            st.warning("Please enter at least 30 characters.")

        else:
            try:
                model, vectorizer = load_model()

                cleaned = clean_text(news)

                if not cleaned:
                    st.warning("Please enter valid news text.")

                else:
                    vector = vectorizer.transform([cleaned])

                    prediction = model.predict(vector)[0]
                    probabilities = model.predict_proba(vector)[0]

                    probability_map = {
                        int(label): float(prob)
                        for label, prob in zip(
                            model.classes_, probabilities
                        )
                    }

                    fake_prob = probability_map.get(0, 0) * 100
                    real_prob = probability_map.get(1, 0) * 100
                    confidence = max(fake_prob, real_prob)

                    st.divider()
                    st.markdown("## 📊 Prediction Result")

                    if prediction == 0:
                        st.error("🚨 FAKE NEWS PATTERN DETECTED")
                    else:
                        st.success("✅ REAL NEWS PATTERN DETECTED")

                    st.metric(
                        "Model Confidence",
                        f"{confidence:.2f}%"
                    )

                    st.markdown("### Probability Breakdown")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**🚨 Fake News**")
                        st.progress(int(round(fake_prob)))
                        st.write(f"{fake_prob:.2f}%")

                    with col2:
                        st.markdown("**✅ Real News**")
                        st.progress(int(round(real_prob)))
                        st.write(f"{real_prob:.2f}%")

                    st.info(
                        "These values are model prediction probabilities, "
                        "not proof of factual accuracy. Always verify "
                        "important news using reliable sources."
                    )

            except FileNotFoundError:
                st.error(
                    "Model files not found. Check your models folder."
                )

            except Exception as e:
                st.error(f"Prediction error: {e}")


# =========================================
# MODEL PERFORMANCE PAGE
# =========================================

elif page == "📊 Model Performance":

    page_header(
        "MODEL EVALUATION",
        "Model Performance",
        "Evaluation results from your trained model's test dataset."
    )

    st.markdown("## 📈 Training Evaluation")

    # Values from your actual confusion matrix:
    # [[3510, 70], [32, 4154]]

    accuracy = (3510 + 4154) / (3510 + 70 + 32 + 4154) * 100

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Test Accuracy", f"{accuracy:.2f}%")

    with col2:
        st.metric("Correct Predictions", "7,664")

    with col3:
        st.metric("Test Samples", "7,766")

    st.markdown("## Confusion Matrix")

    st.write(
        "Rows represent actual classes; columns represent predicted classes."
    )

    matrix = pd.DataFrame(
        [
            [3510, 70],
            [32, 4154]
        ],
        index=["Actual Fake", "Actual Real"],
        columns=["Predicted Fake", "Predicted Real"]
    )

    st.dataframe(matrix, use_container_width=True)

    st.markdown("## Classification Metrics")

    metrics = pd.DataFrame({
        "Metric": [
            "Fake Precision",
            "Fake Recall",
            "Real Precision",
            "Real Recall"
        ],
        "Score": [
            "99.10%",
            "98.04%",
            "98.34%",
            "99.23%"
        ]
    })

    st.dataframe(
        metrics,
        hide_index=True,
        use_container_width=True
    )

    st.caption(
        "These metrics are calculated from the confusion matrix "
        "produced during your model training run. They measure "
        "performance on the held-out test dataset, not on all "
        "real-world news."
    )


# =========================================
# HOW IT WORKS PAGE
# =========================================

elif page == "⚙️ How It Works":

    page_header(
        "SYSTEM WORKFLOW",
        "How NewsGuard AI Works",
        "Five stages take the news article from text input to prediction."
    )

    steps = [
        (
            "01",
            "📰 News Input",
            "The user enters a news article into the application."
        ),
        (
            "02",
            "🧹 Text Cleaning",
            "URLs, special characters and unnecessary spaces are removed."
        ),
        (
            "03",
            "🔤 TF-IDF",
            "The text is converted into numerical features."
        ),
        (
            "04",
            "🧠 Logistic Regression",
            "The trained classifier analyzes the extracted features."
        ),
        (
            "05",
            "🎯 Prediction",
            "The system outputs a Fake or Real classification."
        )
    ]

    for number, title, description in steps:
        st.markdown(f"""
        <div class="card">
            <h3>{number} &nbsp; {title}</h3>
            <p>{description}</p>
        </div>
        """, unsafe_allow_html=True)


# =========================================
# ABOUT PAGE
# =========================================

elif page == "ℹ️ About":

    page_header(
        "PROJECT INFORMATION",
        "About NewsGuard AI",
        "Fake News Prediction Using Machine Learning."
    )

    st.markdown("## 🎯 Project Objective")

    st.markdown("""
    <div class="card">
        <p>
        The objective of this project is to develop a web-based
        Machine Learning system that analyzes news article text
        and classifies it into Fake News or Real News categories.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## 🛠 Technologies Used")

    technologies = [
        ("🐍 Python", "Core programming language"),
        ("📊 Pandas", "Dataset processing"),
        ("🧠 Scikit-learn", "Machine Learning"),
        ("🔤 NLP", "Natural Language Processing"),
        ("📈 TF-IDF", "Text feature extraction"),
        ("🌐 Streamlit", "Web application interface")
    ]

    col1, col2 = st.columns(2)

    for i, (name, description) in enumerate(technologies):
        with (col1 if i % 2 == 0 else col2):
            st.markdown(f"""
            <div class="card">
                <h3>{name}</h3>
                <p>{description}</p>
            </div>
            """, unsafe_allow_html=True)


# =========================================
# FOOTER
# =========================================

st.divider()

st.markdown("""
<p style="text-align:center; color:#718096;">
    <b>NewsGuard AI</b> | Fake News Prediction Using Machine Learning
    <br>
    Developed with Python, NLP and Streamlit
</p>
""", unsafe_allow_html=True)
