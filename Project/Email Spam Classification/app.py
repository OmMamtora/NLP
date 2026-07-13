import streamlit as st
import pickle
import re
import string
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="AI Email Spam Detector",
    page_icon="📧",
    layout="wide"
)

# ==========================================
# LOAD MODEL
# ==========================================

model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# ==========================================
# NLP PREPROCESSING
# ==========================================

punctuation = string.punctuation
stop_word = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_data(text):

    text = text.lower()

    text = re.sub(
        r'https?://\S+|www\.\S+',
        '',
        text
    )

    text = text.translate(
        str.maketrans('', '', punctuation)
    )

    words = text.split()

    words = [
        word
        for word in words
        if word not in stop_word
    ]

    words = [
        lemmatizer.lemmatize(word)
        for word in words
    ]

    return " ".join(words)


# ==========================================
# PREDICTION FUNCTION
# ==========================================

def predict_email(text):

    cleaned_text = clean_data(text)

    vectorized_text = vectorizer.transform([cleaned_text])

    prediction = model.predict(vectorized_text)[0]

    score = model.decision_function( vectorized_text)[0]

    confidence = 1 / (1 + np.exp(-score))

    return prediction, confidence, cleaned_text


# ==========================================
# HEADER
# ==========================================

st.markdown(
    """
    <h1 style='text-align:center'>
        📧 AI Email Spam Detector
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# ==========================================
# LAYOUT
# ==========================================

left_col, center_col, right_col = st.columns(
    [1.2, 3.5, 1.2]
)

prediction = None
confidence = 0
cleaned_text = ""

# ==========================================
# LEFT PANEL
# ==========================================

with left_col:

    st.subheader("ℹ️ About")

    st.info(
        """
        - NLP Preprocessing
        - TF-IDF Vectorization
        - Linear Support Vector Machine
        to classify emails as Spam or Not Spam.
        """
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.subheader("🧪 Examples")

    st.success(
        "Example Spam\n\n"
        "You have won ₹50,000. "
        "Claim your prize now."
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.info(
        "Example Not Spam\n\n"
        "Meeting tomorrow at 10 AM."
    )

# ==========================================
# CENTER PANEL
# ==========================================

with center_col:

    email = st.text_area(
        "📩 Email Content",
        height=250,
        placeholder="Paste email content here..."
    )

    predict_btn = st.button(
        "🚀 Predict",
        use_container_width=True
    )

    if predict_btn and email.strip():

        prediction, confidence, cleaned_text = predict_email(
            email
        )

        st.markdown("---")

    st.subheader("🔍 Processed Text")

    st.text_area(
        "",
        value=cleaned_text,
        height=120,
        disabled=True
    )

# ==========================================
# RIGHT PANEL
# ==========================================

with right_col:

    st.subheader("📊 Result")

    if prediction is not None:

        if prediction == 1:

            st.error("Spam")

        else:

            st.success("Not Spam")

        st.metric(
            "Confidence",
            f"{confidence:.2%}"
        )

        

# ==========================================
# PROCESSED TEXT SECTION
# ==========================================

# if prediction is not None:

#     st.markdown("---")

#     st.subheader("🔍 Processed Text")

#     st.text_area(
#         "",
#         value=cleaned_text,
#         height=120,
#         disabled=True
#     )

# # ==========================================
# # FOOTER
# # ==========================================

# st.markdown("---")

# st.markdown(
#     """
#     <center>
#     Built with ❤️ using Streamlit
#     </center>
#     """,
#     unsafe_allow_html=True
# )