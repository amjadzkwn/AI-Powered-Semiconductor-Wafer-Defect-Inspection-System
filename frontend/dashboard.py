import base64
import os
from io import BytesIO

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from PIL import Image

API_URL = os.getenv("API_URL", "http://localhost:8000")
TIMEOUT = 60

st.set_page_config(page_title="Wafer AI Inspection", page_icon="🔬", layout="wide")
st.title("🔬 AI-Powered Semiconductor Wafer Inspection")
st.caption("Custom CNN · FastAPI · REST API · Production Analytics")


def api_get(path, **params):
    response = requests.get(f"{API_URL}{path}", params=params, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


def show_error(error):
    st.error(f"API error: {error}")
    st.info("Confirm FastAPI is running on port 8000.")

page = st.sidebar.radio("Navigation", ["Inspection", "History", "Production Analytics"])

if page == "Inspection":
    uploaded = st.file_uploader("Upload wafer map", type=["png", "jpg", "jpeg", "bmp", "npy"])
    if uploaded is not None:
        if uploaded.type and uploaded.type.startswith("image/"):
            st.image(uploaded, caption="Uploaded wafer map", width=350)
        if st.button("Run AI Inspection", type="primary", use_container_width=True):
            try:
                with st.spinner("Running model and Grad-CAM..."):
                    response = requests.post(
                        f"{API_URL}/api/v1/upload-image",
                        files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type or "application/octet-stream")},
                        timeout=TIMEOUT,
                    )
                    response.raise_for_status()
                    result = response.json()
                c1, c2, c3 = st.columns(3)
                c1.metric("Prediction", result["prediction"])
                c2.metric("Confidence", f'{result["confidence"]:.2f}%')
                c3.metric("Inference", f'{result["inference_ms"]:.1f} ms')
                if result["is_defective"]:
                    st.error("Inspection result: DEFECTIVE")
                else:
                    st.success("Inspection result: NORMAL")
                heatmap_bytes = base64.b64decode(result["heatmap"].split(",", 1)[1])
                st.image(Image.open(BytesIO(heatmap_bytes)), caption="Grad-CAM heatmap", width=450)
                probability_df = pd.DataFrame(
                    {"Class": list(result["probabilities"].keys()), "Probability": list(result["probabilities"].values())}
                ).sort_values("Probability", ascending=True)
                st.plotly_chart(px.bar(probability_df, x="Probability", y="Class", orientation="h", title="Class probabilities (%)"), use_container_width=True)
            except Exception as error:
                show_error(error)

elif page == "History":
    try:
        payload = api_get("/api/v1/predictions", page=1, page_size=100)
        frame = pd.DataFrame(payload["items"])
        if frame.empty:
            st.info("No inspection history yet.")
        else:
            st.dataframe(frame, use_container_width=True, hide_index=True)
            st.subheader("Add ground truth")
            selected_id = st.selectbox("Prediction ID", frame["id"].tolist())
            model_info = api_get("/api/v1/model-info")
            ground_truth = st.selectbox("Actual class", model_info["classes"])
            if st.button("Save ground truth"):
                response = requests.patch(
                    f"{API_URL}/api/v1/predictions/{selected_id}/ground-truth",
                    json={"ground_truth": ground_truth}, timeout=TIMEOUT,
                )
                response.raise_for_status()
                st.success("Ground truth saved. Analytics will now include measured accuracy and false-positive rate.")
                st.rerun()
    except Exception as error:
        show_error(error)

else:
    try:
        summary = api_get("/api/v1/analytics/summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total wafers", summary["total_wafers"])
        c2.metric("Normal", summary["normal_count"])
        c3.metric("Defective", summary["defective_count"])
        c4.metric("Defect rate", f'{summary["defect_rate"]:.2f}%')
        c5, c6, c7 = st.columns(3)
        c5.metric("Most common defect", summary["most_common_defect"] or "N/A")
        c6.metric("Average confidence", f'{summary["average_confidence"]:.2f}%')
        fpr = summary["estimated_false_positive_rate"]
        c7.metric("Measured false-positive rate", "N/A" if fpr is None else f"{fpr:.2f}%")

        daily = pd.DataFrame(api_get("/api/v1/analytics/daily", days=30))
        distribution = pd.DataFrame(api_get("/api/v1/analytics/distribution"))
        if not daily.empty:
            daily["date"] = pd.to_datetime(daily["date"])
            st.plotly_chart(px.line(daily, x="date", y=["normal", "defective"], markers=True, title="Daily inspection trend"), use_container_width=True)
        if not distribution.empty:
            st.plotly_chart(px.pie(distribution, names="defect_type", values="count", title="Defect type distribution"), use_container_width=True)
    except Exception as error:
        show_error(error)
