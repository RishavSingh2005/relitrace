"""ReliTrace report explorer. Shows real exported results, never demo metrics."""
from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
import streamlit as st
from report_loader import load_directory, load_zip

st.set_page_config(page_title="ReliTrace | Reliability Diagnostics", page_icon="🔎", layout="wide")

st.markdown("""
<style>
.block-container {max-width: 1200px; padding-top: 2rem;}
[data-testid="stMetric"] {background:#f3f6fc; border:1px solid #dde5f2; border-radius:12px; padding:14px;}
@media (prefers-color-scheme: dark) { [data-testid="stMetric"] {background:#1d2532; border-color:#334155;} }
</style>
""", unsafe_allow_html=True)

st.title("ReliTrace")
st.caption("Test deeper. Understand failure. Build with confidence.")
st.write("**Live CSV diagnostics + experiment evidence explorer** · Automatic experimental calibration available")
st.info("This dashboard reads saved analysis outputs. It does not train models or establish production safety. Synthetic interventions keep labels fixed.")

# Choose between raw CSV feature comparison and precomputed ZIP reports.
# CSV upload is intentionally available on the landing page, not hidden behind
# the separate diagnostics page.
mode = st.radio(
    "Choose a workflow",
    ["Analyze CSV datasets", "Explore saved experiment reports (ZIP)"],
    horizontal=True,
)

if mode == "Analyze CSV datasets":
    import hashlib
    from relitrace.csv_input import read_feature_csv, align_feature_frames
    from relitrace.diagnostics import detect_data_shift

    st.subheader("Compare two CSV files")
    st.write(
        "Upload **reference_features.csv** (historical/training features) and "
        "**current_features.csv** (new/test features). Both files should contain "
        "the same feature column names. Target labels are not required."
    )
    from relitrace.calibration import calibrate_thresholds

    cutoff_mode = st.radio(
        "Threshold selection",
        ["Automatic calibration (experimental)", "Manual thresholds"],
        horizontal=True,
        help="Automatic mode learns cutoffs from the reference CSV only."
    )
    is_auto = cutoff_mode.startswith("Automatic")
    with st.form("csv_diagnostics_form"):
        left, right = st.columns(2)
        with left:
            reference_csv = st.file_uploader(
                "Reference CSV", type=["csv"], key="landing_reference_csv"
            )
        with right:
            current_csv = st.file_uploader(
                "Current CSV", type=["csv"], key="landing_current_csv"
            )
        if is_auto:
            st.caption(
                "Uses 100 calibration splits and 40 separate validation splits of "
                "the reference dataset (seed 2026). Requires ≥80 reference rows. "
                "Repeated splits reuse the same records: warnings are not "
                "statistical guarantees."
            )
            dist_threshold = None
            missing_threshold = None
        else:
            t1, t2 = st.columns(2)
            with t1:
                dist_threshold = st.number_input(
                    "Distribution cutoff (exploratory)", min_value=0.001,
                    max_value=1.0, value=0.10, step=0.01, format="%.3f"
                )
            with t2:
                missing_threshold = st.number_input(
                    "Missingness cutoff (exploratory)", min_value=0.001,
                    max_value=1.0, value=0.05, step=0.01, format="%.3f"
                )
        submitted = st.form_submit_button("Analyze CSV datasets", type="primary")

    def upload_signature():
        if reference_csv is None or current_csv is None:
            return None
        return (
            hashlib.sha256(reference_csv.getvalue()).hexdigest(),
            hashlib.sha256(current_csv.getvalue()).hexdigest(),
            cutoff_mode,
            None if is_auto else float(dist_threshold),
            None if is_auto else float(missing_threshold),
        )

    if submitted:
        if reference_csv is None or current_csv is None:
            st.error("Please upload both CSV files. ZIP files go in the other workflow.")
        else:
            try:
                reference = read_feature_csv(reference_csv.getvalue(), "Reference CSV")
                current = read_feature_csv(current_csv.getvalue(), "Current CSV")
                reference, current = align_feature_frames(reference, current)
                calibration = None
                if is_auto:
                    with st.spinner("Estimating cutoffs from reference data..."):
                        calibration = calibrate_thresholds(
                            reference, len(current),
                            calibration_runs=100,
                            validation_runs=40,
                            alpha=0.05,
                            seed=2026,
                        )
                    used_dist = calibration.distribution_threshold
                    used_missing = calibration.missingness_threshold
                else:
                    used_dist = float(dist_threshold)
                    used_missing = float(missing_threshold)
                report = detect_data_shift(
                    reference, current,
                    distribution_threshold=used_dist,
                    missingness_threshold=used_missing,
                )
            except (TypeError, ValueError) as exc:
                st.session_state.pop("landing_csv_report", None)
                st.error(f"Unable to analyze CSV files: {exc}")
            else:
                st.session_state["landing_csv_report"] = report
                st.session_state["landing_csv_signature"] = upload_signature()
                st.session_state["landing_csv_calibration"] = calibration
                st.session_state["landing_csv_meta"] = {
                    "reference_rows": len(reference),
                    "current_rows": len(current),
                    "features": len(reference.columns),
                    "mode": "auto" if is_auto else "manual",
                    "distribution_cutoff": used_dist,
                    "missingness_cutoff": used_missing,
                    "calibration_runs": 100 if is_auto else None,
                    "validation_runs": 40 if is_auto else None,
                    "empirical_control_warning_fraction": (
                        calibration.observed_control_warning_fraction
                        if calibration is not None else None
                    ),
                    "reference_sample_rows": (
                        calibration.reference_sample_rows if calibration else None
                    ),
                    "current_sample_rows": (
                        calibration.current_sample_rows if calibration else None
                    ),
                    "notes": "Exploratory feature-level shift checks; not model-performance validation",
                }

    report = st.session_state.get("landing_csv_report")
    if report is not None:
        if upload_signature() != st.session_state.get("landing_csv_signature"):
            st.warning("Inputs or thresholds changed. Click Analyze CSV datasets to refresh.")
        else:
            meta = st.session_state["landing_csv_meta"]
            cols = st.columns(4)
            cols[0].metric("Reference rows", meta["reference_rows"])
            cols[1].metric("Current rows", meta["current_rows"])
            cols[2].metric("Features checked", meta["features"])
            cols[3].metric("Flagged features", int(report["shift_detected"].sum()))
            flagged_only = st.checkbox("Show flagged features only", key="landing_flagged_only")
            view = report.loc[report["shift_detected"]] if flagged_only else report
            st.dataframe(view, use_container_width=True, hide_index=True)
            st.bar_chart(report.set_index("feature")[["distribution_score"]])
            st.write(
                f"Thresholds used: distribution **{meta['distribution_cutoff']:.4f}**, "
                f"missingness **{meta['missingness_cutoff']:.4f}**"
            )
            calibration = st.session_state.get("landing_csv_calibration")
            if calibration is not None:
                st.metric(
                    "Unmodified-control warning fraction",
                    f"{calibration.observed_control_warning_fraction:.1%}"
                )
                st.caption(
                    f"Calibrated on {calibration.calibration_runs} partitions; "
                    f"checked on {calibration.validation_runs} more partitions. "
                    f"Each reference-only comparison uses "
                    f"{calibration.reference_sample_rows} reference and "
                    f"{calibration.current_sample_rows} current rows. "
                    "Repeated partitions from one dataset do not establish "
                    "out-of-sample false-positive control."
                )
                with st.expander("Inspect calibration control results"):
                    st.dataframe(calibration.validation_scores, hide_index=True)
                st.download_button(
                    "Download calibration controls CSV",
                    calibration.null_scores.to_csv(index=False).encode("utf-8"),
                    file_name="relitrace_calibration_controls.csv", mime="text/csv",
                    key="landing_calibration_download"
                )
                st.download_button(
                    "Download validation controls CSV",
                    calibration.validation_scores.to_csv(index=False).encode("utf-8"),
                    file_name="relitrace_validation_controls.csv", mime="text/csv",
                    key="landing_validation_download"
                )
            st.download_button(
                "Download run metadata JSON", json.dumps(meta, indent=2).encode("utf-8"),
                file_name="relitrace_run_config.json", mime="application/json",
                key="landing_meta_download"
            )
            st.caption(
                "Numerical: KS statistic; categorical: total variation distance; "
                "missing values are checked separately. Warnings do not demonstrate model failure."
            )
    st.info(
        "**Have a ZIP from Google Colab?** Choose "
        "**Explore saved experiment reports (ZIP)** above. "
        "CSV comparison and ZIP report exploration are different workflows."
    )
    st.stop()

with st.sidebar:
    st.header("Load reports")
    st.write("Upload the ZIP exported from Google Colab, or place reports under `relitrace_outputs/reports/` next to `app.py`.")
    uploaded = st.file_uploader("ReliTrace results ZIP", type=["zip"], help="Contains aggregated CSV/JSON experiment reports, not original client records.")
    st.caption("Reports are parsed in memory in the running app. Don't upload confidential files to a public deployment.")

bundle = load_zip(uploaded.getvalue()) if uploaded is not None else load_directory(Path(__file__).resolve().parent)
for message in bundle.warnings:
    st.warning(message)

if not bundle.has_data():
    st.subheader("No experiment results loaded")
    st.write("Export the `relitrace_outputs` directory from Colab as a ZIP and upload it above. No example metrics are shown because we have not loaded your actual results.")
    st.code('''from google.colab import files
import shutil
shutil.make_archive("/content/relitrace_results", "zip", root_dir="/content", base_dir="relitrace_outputs")
files.download("/content/relitrace_results.zip")''', language="python")
    st.stop()

T = bundle.tables
reliability = T.get("reliability", pd.DataFrame())
features = T.get("features", pd.DataFrame())
uncertainty = T.get("uncertainty", pd.DataFrame())
subgroups = T.get("subgroups", pd.DataFrame())
controls = T.get("controls", pd.DataFrame())
meta = bundle.metadata


def read_number(frame, column, experiment="original"):
    if frame.empty or "experiment" not in frame or column not in frame:
        return None
    result = frame.loc[frame["experiment"] == experiment, column]
    if result.empty:
        return None
    number = pd.to_numeric(result, errors="coerce").iloc[0]
    return None if pd.isna(number) else float(number)


def table_and_download(title, frame, filename):
    st.subheader(title)
    if frame.empty:
        st.info(f"Missing: {filename}. Export or rerun the corresponding notebook cell.")
        return
    st.dataframe(frame, use_container_width=True, hide_index=True)
    st.download_button(f"Download {filename}", frame.to_csv(index=False).encode("utf-8"), file_name=filename, mime="text/csv", key=f"dl_{filename}")


scenario_count = max(0, len(reliability) - ("experiment" in reliability and "original" in reliability["experiment"].values))
original_warnings = read_number(reliability, "n_warnings")
warning_fraction = meta.get("observed_control_warning_fraction")
cols = st.columns(4)
cols[0].metric("Loaded result tables", len(T))
cols[1].metric("Shift scenarios", scenario_count if not reliability.empty else "N/A")
cols[2].metric("Original-input warnings", int(original_warnings) if original_warnings is not None else "N/A")
cols[3].metric("Control warning fraction", f"{float(warning_fraction):.1%}" if isinstance(warning_fraction, (float, int)) else "N/A")

overview_tab, shift_tab, uncertainty_tab, group_tab, methods_tab = st.tabs([
    "Overview", "Shift diagnostics", "Uncertainty", "Subgroups", "Validation & methods"
])
with overview_tab:
    table_and_download("Scenario performance", reliability, "reliability_summary.csv")
    if not reliability.empty and {"experiment", "f1_drop", "roc_auc_drop", "ap_drop"}.issubset(reliability.columns):
        st.write("Metric drop = original minus shifted. Positive values indicate measured degradation.")
        chart_data = reliability.set_index("experiment")[["f1_drop", "roc_auc_drop", "ap_drop"]]
        st.bar_chart(chart_data)
    if "baseline" in T:
        table_and_download("Initial model baselines", T["baseline"], "baseline_metrics.csv")
with shift_tab:
    if features.empty:
        st.info("Feature-level diagnostics were not included in the upload.")
    else:
        if "experiment" in features:
            choices = list(features["experiment"].dropna().astype(str).unique())
            selection = st.selectbox("Experiment", choices)
            view = features[features["experiment"].astype(str) == selection].copy()
        else:
            view = features.copy()
        if "shift_detected" in view:
            only_flagged = st.checkbox("Show only flagged features", value=False)
            if only_flagged:
                view = view[view["shift_detected"].astype(str).str.lower() == "true"]
        st.dataframe(view, use_container_width=True, hide_index=True)
        st.download_button("Download all feature diagnostics", features.to_csv(index=False).encode("utf-8"), "feature_diagnostics.csv", "text/csv")
with uncertainty_tab:
    table_and_download("Paired bootstrap intervals", uncertainty, "uncertainty_summary.csv")
    st.caption("Exploratory percentile intervals conditional on the fitted model, test data and synthetic interventions; they do not measure training uncertainty.")
    if not uncertainty.empty and {"metric", "experiment", "lower_95", "upper_95", "mean_drop"}.issubset(uncertainty):
        metric = st.selectbox("Choose performance metric", sorted(uncertainty["metric"].dropna().unique()))
        subset = uncertainty.loc[uncertainty["metric"] == metric, ["experiment", "mean_drop", "lower_95", "upper_95"]]
        st.dataframe(subset, use_container_width=True, hide_index=True)
with group_tab:
    table_and_download("Age-subgroup sensitivity", subgroups, "subgroup_diagnostics.csv")
    st.caption("Age brackets are defined from the unmodified test data. This is slice analysis, not a fairness certification.")
    if not subgroups.empty and {"experiment", "age_group", "f1_drop"}.issubset(subgroups.columns):
        st.bar_chart(subgroups.pivot(index="age_group", columns="experiment", values="f1_drop"))
with methods_tab:
    st.subheader("Experiment configuration")
    if meta:
        st.json(meta)
    else:
        st.info("Experiment metadata not available.")
    table_and_download("Control runs", controls, "validation_controls.csv")
    if "calibration" in T:
        table_and_download("Calibration samples", T["calibration"], "calibration_controls.csv")
    if "validation_summary" in T:
        table_and_download("Repeated-shift validation", T["validation_summary"], "validation_summary.csv")
    st.warning("Known limitations: synthetic shifts may not resemble deployment changes; fixed labels can become unrealistic; cutoffs are heuristic/empirical and unvalidated externally; the `duration` feature needs a prediction-time decision before real-world use.")

st.divider()
st.caption("ReliTrace · Experimental analysis. Results shown come from uploaded or locally saved reports; no scores are invented.")
