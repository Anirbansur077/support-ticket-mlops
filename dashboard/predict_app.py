import joblib
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Ticket Escalation Predictor", page_icon="🎫")
st.title("Ticket Escalation Predictor")
st.caption("Enter a support ticket and the model estimates its escalation risk.")


@st.cache_resource
def load_options():
    encoder = joblib.load("src/serving/artifacts/encoder.pkl")
    top_tags = joblib.load("src/serving/artifacts/top_tags.pkl")
    types, queues = encoder.categories_  # order matches cat_cols = ['type', 'queue']
    return list(types), list(queues), list(top_tags)


types, queues, top_tags = load_options()

with st.form("ticket_form"):
    subject = st.text_input("Subject", placeholder="e.g. Production server down")
    body = st.text_area("Body", height=150, placeholder="Describe the issue...")
    col1, col2 = st.columns(2)
    ticket_type = col1.selectbox("Type", types)
    queue = col2.selectbox("Queue", queues)
    tags = st.multiselect("Tags", top_tags)
    submitted = st.form_submit_button("Predict escalation risk")

if submitted:
    payload = {
        "subject": subject,
        "body": body,
        "type": ticket_type,
        "queue": queue,
        "tags": tags,
    }
    try:
        response = requests.post(API_URL, json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
    except requests.exceptions.ConnectionError:
        st.error("Can't reach the API. Is `uvicorn src.serving.main:app --port 8000` running?")
        st.stop()
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    risk = result["escalation_risk"]
    st.metric("Escalation risk", f"{risk:.1%}")
    st.progress(min(max(risk, 0.0), 1.0))
    if result["escalated"]:
        st.error("Likely to be escalated: route to a senior agent.")
    else:
        st.success("Unlikely to be escalated: standard handling.")