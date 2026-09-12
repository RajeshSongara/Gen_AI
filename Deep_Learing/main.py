from pathlib import Path
import re

import streamlit as st
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence


MODEL_PATH = Path(__file__).resolve().parent / "simple_rnn_imdb.h5"
MAX_FEATURES = 10_000
MAX_LEN = 500
DECISION_THRESHOLD = 0.49


@st.cache_resource
def load_sentiment_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_word_index():
    return imdb.get_word_index()


def preprocess_text(text, word_index):
    cleaned_text = re.sub(r"[^a-zA-Z\s]", "", text.lower())
    words = cleaned_text.split()
    unknown_token = 2 + 3
    encoded_review = []
    for word in words:
        token_id = word_index.get(word, 2) + 3
        encoded_review.append(
            token_id if token_id < MAX_FEATURES else unknown_token
        )
    return sequence.pad_sequences([encoded_review], maxlen=MAX_LEN)


def predict_sentiment(review, model, word_index):
    model_input = preprocess_text(review, word_index)
    score = float(model.predict(model_input, verbose=0)[0][0])
    sentiment = "Positive" if score >= DECISION_THRESHOLD else "Negative"
    return sentiment, score


def main():
    st.set_page_config(
        page_title="ReelFeel | IMDb Sentiment",
        page_icon="🎞️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');

        :root {
            --ink: #071018;
            --panel: rgba(16, 31, 41, .82);
            --line: rgba(177, 232, 224, .18);
            --mint: #a5f3df;
            --coral: #ff8c79;
            --cream: #f2eee5;
        }
        .stApp {
            background:
                radial-gradient(circle at 78% 14%, rgba(255, 140, 121, .16), transparent 24%),
                radial-gradient(circle at 8% 76%, rgba(165, 243, 223, .11), transparent 28%),
                linear-gradient(135deg, #071018 0%, #0c1822 52%, #132329 100%);
            color: var(--cream);
            font-family: 'Space Grotesk', sans-serif;
        }
        .block-container { max-width: 1180px; padding: 2.5rem 2rem 4rem; }
        .hero { display: grid; grid-template-columns: 1.1fr .9fr; gap: 2rem; align-items: center; margin-bottom: 2rem; }
        .eyebrow { color: var(--mint); font-family: 'DM Mono', monospace; font-size: .72rem; letter-spacing: .16em; text-transform: uppercase; }
        .hero h1 { color: var(--cream); font-size: clamp(2.8rem, 6vw, 5.5rem); line-height: .92; letter-spacing: -.06em; margin: .6rem 0 1rem; }
        .hero p { color: #a5b7ba; font-size: 1.05rem; max-width: 34rem; line-height: 1.6; }
        .scene { height: 270px; perspective: 900px; position: relative; }
        .reel { position: absolute; inset: 18px 22px 18px 10px; transform: rotateY(-20deg) rotateX(8deg) rotateZ(3deg); transform-style: preserve-3d; }
        .reel-card { position: absolute; border: 1px solid var(--line); border-radius: 14px; box-shadow: 22px 26px 45px rgba(0,0,0,.35); }
        .back-card { inset: 26px 0 0 34px; background: linear-gradient(145deg, #17343b, #0c1b25); transform: translateZ(-50px); }
        .front-card { inset: 0 34px 26px 0; background: linear-gradient(145deg, rgba(34, 66, 71, .94), rgba(12, 26, 34, .97)); padding: 24px; transform: translateZ(35px); }
        .front-card .signal { color: var(--coral); font-family: 'DM Mono', monospace; font-size: .7rem; }
        .front-card .wave { height: 72px; margin-top: 28px; background: repeating-linear-gradient(90deg, transparent 0 8px, var(--mint) 9px 11px, transparent 12px 18px); opacity: .72; clip-path: polygon(0 58%, 6% 38%, 12% 69%, 18% 20%, 24% 73%, 31% 43%, 37% 62%, 44% 14%, 51% 78%, 58% 31%, 65% 60%, 72% 25%, 80% 71%, 87% 42%, 94% 56%, 100% 18%, 100% 100%, 0 100%); }
        .workspace { border: 1px solid var(--line); border-radius: 18px; padding: 1.2rem 1.35rem 1.4rem; background: var(--panel); box-shadow: 0 24px 60px rgba(0,0,0,.2); }
        .workspace-label { color: #a5b7ba; font-family: 'DM Mono', monospace; font-size: .72rem; letter-spacing: .11em; text-transform: uppercase; margin-bottom: .65rem; }
        div[data-testid="stTextArea"] textarea { background: rgba(4, 13, 19, .72); color: var(--cream); border: 1px solid rgba(165, 243, 223, .25); border-radius: 12px; font-family: 'Space Grotesk', sans-serif; font-size: 1rem; }
        div[data-testid="stTextArea"] textarea:focus { border-color: var(--mint); box-shadow: 0 0 0 1px var(--mint); }
        div.stButton > button { background: var(--mint); color: var(--ink); border: 0; border-radius: 10px; font-weight: 700; padding: .7rem 1.25rem; transition: transform .2s ease, box-shadow .2s ease; }
        div.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 10px 26px rgba(165, 243, 223, .2); }
        .result { margin-top: 1.4rem; border-radius: 14px; padding: 1rem 1.2rem; border: 1px solid var(--line); background: rgba(8, 19, 26, .8); }
        .result-positive { border-color: rgba(165, 243, 223, .55); }
        .result-negative { border-color: rgba(255, 140, 121, .55); }
        .result-kicker { color: #9cafb1; font-family: 'DM Mono', monospace; font-size: .7rem; text-transform: uppercase; letter-spacing: .13em; }
        .result-value { color: var(--cream); font-size: 2rem; font-weight: 700; margin-top: .25rem; }
        .score-card { display: flex; justify-content: space-between; align-items: end; gap: 1rem; margin-top: 1rem; padding: .95rem 1.1rem; border-radius: 12px; background: rgba(4, 13, 19, .62); border: 1px solid var(--line); }
        .score-label { color: #9cafb1; font-family: 'DM Mono', monospace; font-size: .7rem; letter-spacing: .1em; text-transform: uppercase; }
        .score-value { font-family: 'DM Mono', monospace; font-size: 1.45rem; font-weight: 500; }
        .score-positive { color: var(--mint); }
        .score-negative { color: var(--coral); }
        @media (max-width: 760px) { .hero { grid-template-columns: 1fr; } .scene { height: 210px; } .block-container { padding: 1.5rem 1rem 3rem; } }
        </style>
        <div class="hero">
            <div>
                <div class="eyebrow">ReelFeel / neural readout</div>
                <h1>Feel the<br>review.</h1>
                <p>A compact IMDb-trained recurrent model that turns the emotional texture of a movie review into a probability signal.</p>
            </div>
            <div class="scene" aria-hidden="true">
                <div class="reel">
                    <div class="reel-card back-card"></div>
                    <div class="reel-card front-card">
                        <div class="signal">LIVE SENTIMENT SIGNAL</div>
                        <div class="wave"></div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        model = load_sentiment_model()
        word_index = load_word_index()
    except Exception as error:
        st.error(f"Could not load the sentiment model: {error}")
        st.stop()

    st.markdown('<div class="workspace"><div class="workspace-label">01 / Feed the model</div>', unsafe_allow_html=True)
    review = st.text_area(
        "Movie review",
        value="This movie was fantastic! The acting was great and the plot was thrilling.",
        height=150,
        label_visibility="collapsed",
    )

    if st.button("Analyze sentiment", type="primary"):
        if not review.strip():
            st.warning("Please enter a review first.")
            return

        sentiment, score = predict_sentiment(review, model, word_index)
        result_class = "result-positive" if sentiment == "Positive" else "result-negative"
        st.markdown(
            f'<div class="result {result_class}"><div class="result-kicker">02 / Classification</div><div class="result-value">{sentiment}</div></div>',
            unsafe_allow_html=True,
        )
        score_class = "score-positive" if sentiment == "Positive" else "score-negative"
        st.markdown(
            f'<div class="score-card"><div class="score-label">Positive probability<br><small>Threshold: {DECISION_THRESHOLD:.0%}</small></div><div class="score-value {score_class}">{score:.2%}</div></div>',
            unsafe_allow_html=True,
        )
        st.progress(min(max(score, 0.0), 1.0))

    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
