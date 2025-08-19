import streamlit as st
import pickle
import os
import numpy as np
from gensim.models import Word2Vec
from scipy.sparse import hstack, csr_matrix, vstack
from lime.lime_text import LimeTextExplainer


base_path = os.path.dirname(__file__)


@st.cache_resource
def load_model():
    with open(os.path.join(base_path, "best_model.pkl"), "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_vectorizer():
    with open(os.path.join(base_path, "tfidf_vectorizer.pkl"), "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_w2v():
    return Word2Vec.load(os.path.join(base_path, "w2v_model.model"))

model = load_model()
tfidf = load_vectorizer()
w2v_model = load_w2v()


def get_avg_w2v(tokens, model, vector_size=None):
    if vector_size is None:
        vector_size = model.vector_size
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if len(vectors) == 0:
        return np.zeros(vector_size)
    return np.mean(vectors, axis=0)

def preprocess(text):
    tokens = text.split()
    X_tfidf = tfidf.transform([text])
    vector_size = w2v_model.vector_size
    X_w2v = get_avg_w2v(tokens, w2v_model, vector_size).reshape(1, -1)
    X_w2v = csr_matrix(X_w2v)
    return hstack([X_tfidf, X_w2v])


st.title("Fake Job Detector")
job_text = st.text_area("Paste job description here:")

if st.button("Predict"):
    if job_text.strip() == "":
        st.error("Please enter a job description!")
    else:
        X = preprocess(job_text)
        if X.shape[1] != model.n_features_in_:
            st.error(
                f"Feature size mismatch! Model expects {model.n_features_in_}, "
                f"but got {X.shape[1]}"
            )
        else:
            prediction = model.predict(X)[0]
            proba = model.predict_proba(X)[0].tolist()
            st.session_state["prediction"] = prediction
            st.session_state["proba"] = proba


if "prediction" in st.session_state:
    st.write("Prediction:", st.session_state["prediction"])
    st.write("**Probabilities:**")
    st.json({"Real": st.session_state["proba"][0], "Fake": st.session_state["proba"][1]})


if st.button("Explain with LIME"):
    if job_text.strip() == "":
        st.error("Please enter a job description!")
    else:
        explainer = LimeTextExplainer(class_names=["Real", "Fake"])

        def predict_proba_for_lime(texts):
            X_list = [preprocess(t) for t in texts]
            X = vstack(X_list)
            return model.predict_proba(X)

        exp = explainer.explain_instance(
            job_text,
            predict_proba_for_lime,
            num_features=6
        )
        
        st.session_state["lime_exp"] = exp


if "lime_exp" in st.session_state:
    st.write("LIME Explanation")
    st.components.v1.html(
        st.session_state["lime_exp"].as_html(),
        height=800,
        scrolling=True
    )
if job_text.strip():
    st.write("Model expects:", model.n_features_in_)
    st.write("Preprocessed features:", preprocess(job_text).shape[1])