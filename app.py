"""Streamlit demo. It loads the committed checkpoint and never trains a model."""
import json

import pandas as pd
import streamlit as st

from turbine.config import ARTIFACT_DIR, FEATURES, FEATURE_LABELS, SOURCE_PAGE
from turbine.inference import load_model, outside_training_range, predict


st.set_page_config(page_title="Turbine NOx Lab", page_icon="⚙️", layout="wide")


@st.cache_resource
def saved_model():
    return load_model()


st.title("Turbine NOx Lab")
st.write("Estimate nitrogen-oxide concentration from nine measurements of a gas turbine.")
st.caption("Same-hour estimation • Output in mg/m³ • A small neural network trained in PyTorch")
try:
    model, checkpoint = saved_model()
except (FileNotFoundError, ValueError, RuntimeError) as error:
    st.error(f"Cannot load artifacts/model.pt: {error}")
    st.stop()

with st.sidebar:
    st.header("About this model")
    st.write("9 inputs · 32 and 16 hidden neurons · 1 output")
    st.write("Training: 2011–2012\n\nValidation: 2013\n\nFinal test: 2014–2015")
    st.markdown(f"[UCI dataset and attribution]({SOURCE_PAGE})")
    st.caption("Research demo using one plant. Predictions are not calibrated uncertainty estimates.")

single_tab, batch_tab, results_tab = st.tabs(["Try a measurement", "Predict a CSV", "Test results"])
with single_tab:
    examples = pd.read_csv(ARTIFACT_DIR / "demo_inputs.csv")
    selected = st.selectbox("Start with a recorded training example", range(len(examples)),
                            format_func=lambda index: f"Example {index + 1}")
    st.caption("The examples come from the training set. Edit the measurements and select Estimate NOx.")
    with st.form("measurement_form"):
        columns = st.columns(3)
        values = {}
        for index, name in enumerate(FEATURES):
            with columns[index % 3]:
                values[name] = st.number_input(
                    FEATURE_LABELS[name], value=float(examples.iloc[selected][name]),
                    format="%.3f", key=f"{selected}_{name}")
        st.form_submit_button("Estimate NOx", type="primary")
    frame = pd.DataFrame([values])
    try:
        prediction = float(predict(frame, model, checkpoint)[0])
        st.metric("Estimated NOx", f"{prediction:.2f} mg/m³")
        outside = outside_training_range(frame, checkpoint).any()
        if outside.any():
            st.warning("Outside the training range: " + ", ".join(outside.index[outside])
                       + ". This estimate may be unreliable.")
        if prediction < 0:
            st.warning("The network returned a physically implausible negative concentration.")
    except ValueError as error:
        st.error(str(error))
    with st.expander("Input definitions and observed training ranges"):
        ranges = pd.DataFrame({
            "Sensor": FEATURES, "Description": [FEATURE_LABELS[name] for name in FEATURES],
            "Training minimum": checkpoint["input_min"], "Training maximum": checkpoint["input_max"],
        })
        st.dataframe(ranges, hide_index=True, width="stretch")
        st.caption("The UCI page labels CDP as mbar although its scale is roughly 10–15. "
                   "The model uses the original CSV values without converting pressure units.")

with batch_tab:
    st.write("Upload a CSV with these nine numeric columns. Column order does not matter.")
    st.code(",".join(FEATURES), language=None)
    st.download_button("Download example input CSV", examples.to_csv(index=False),
                       file_name="turbine_inputs.csv", mime="text/csv")
    upload = st.file_uploader("Sensor measurements", type="csv")
    if upload is not None:
        try:
            frame = pd.read_csv(upload)
            if len(frame) > 10000:
                raise ValueError("Please upload at most 10,000 rows")
            predictions = predict(frame, model, checkpoint)
            result = frame.copy()
            result["predicted_NOX_mg_m3"] = predictions
            result["outside_training_range"] = outside_training_range(frame, checkpoint).any(axis=1)
            st.dataframe(result, hide_index=True, width="stretch")
            st.download_button("Download predictions", result.to_csv(index=False),
                               file_name="nox_predictions.csv", mime="text/csv")
        except (ValueError, TypeError, pd.errors.ParserError) as error:
            st.error(f"Could not predict: {error}")

with results_tab:
    metrics_path = ARTIFACT_DIR / "test_metrics.json"
    report = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else None
    if report and report["checkpoint_sha256"] != checkpoint["sha256"]:
        st.warning("The saved test report belongs to another checkpoint. Run python evaluate.py.")
    elif report:
        st.write(f"Final evaluation on {report['test_count']:,} observations from 2014–2015.")
        summary = pd.DataFrame(report["models"]).T.rename(columns={
            "mae": "MAE (mg/m³)", "rmse": "RMSE (mg/m³)", "r2": "R²"})
        st.dataframe(summary.style.format("{:.3f}"), width="stretch")
        st.caption("Lower MAE and RMSE are better. R² = 1 is perfect; R² below zero is possible.")
        left, right = st.columns(2)
        left.image(str(ARTIFACT_DIR / "loss_curve.png"))
        right.image(str(ARTIFACT_DIR / "test_predictions.png"))
    else:
        st.info("Run python evaluate.py to generate the saved test report.")
    st.write("Limitations: one plant, older observations, possible drift over time, and correlated "
             "measurements. Changing one input does not prove a causal effect on emissions.")
