"""
P2M — ML Drift Monitoring & Retrain
====================================
Streamlit application that compares a Reference and a Current dataset,
detects schema issues, data-quality problems, distribution drift,
evaluates model performance and offers automated retraining with
MLflow tracking.
"""

import streamlit as st
import pandas as pd
import numpy as np

from utils.schema import compare_schemas
from utils.data_quality import check_types, check_missing, detect_outliers
from utils.drift import detect_drift, SENSITIVITY_PRESETS
from utils.performance import (
    evaluate_model,
    check_performance_drop,
    detect_task_type,
    deployment_readiness,
)
from utils.retrain import (
    prepare_features,
    align_features,
    train_model,
    retrain_with_mlflow,
    MODELS,
    get_default_params,
    get_parameter_info,
    find_best_model_and_params,
    format_recommendation_for_display,
    ModelSelectionAgent,
)
from utils.costs import estimate_drift_cost, estimate_retrain_cost, make_retrain_decision
from utils.drift_analysis import analyze_drift_topology
from utils.plots import (
    plot_distribution_comparison,
    plot_drift_summary,
    plot_missing_values,
)

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(page_title="P2M — ML Drift Monitor", page_icon="📊", layout="wide")


# ── Helpers ──────────────────────────────────────────────────────
def detect_mode(ref_df: pd.DataFrame, curr_df: pd.DataFrame, target):
    """Return 'supervised' | 'monitoring' | 'unsupervised'."""
    if target is None:
        return "unsupervised"
    ref_has = target in ref_df.columns and ref_df[target].notna().sum() > 0
    curr_has = target in curr_df.columns and curr_df[target].notna().sum() > 0
    if ref_has and curr_has:
        return "supervised"
    if ref_has:
        return "monitoring"
    return "unsupervised"


# ═══════════════════════ SIDEBAR ═════════════════════════════════
with st.sidebar:
    st.title("📊 P2M")
    st.caption("ML Drift Monitoring & Retrain")
    st.divider()

    ref_file = st.file_uploader("📁 Reference dataset (baseline)", type=["csv"])
    curr_file = st.file_uploader("📁 Current dataset (production)", type=["csv"])

    target_col = None
    use_intersection = False
    feature_only_retrain = False
    sensitivity = "Balanced"

    if ref_file is not None and curr_file is not None:
        ref_df = pd.read_csv(ref_file)
        curr_df = pd.read_csv(curr_file)
        st.session_state["ref_df"] = ref_df
        st.session_state["curr_df"] = curr_df

        st.divider()

        # Target selector
        cols_list = ["None"] + sorted(ref_df.columns.tolist())
        target_col = st.selectbox("🎯 Target column (optional)", cols_list)
        if target_col == "None":
            target_col = None

        use_intersection = st.checkbox("Use intersection of columns only", value=False)

        # Advanced
        with st.expander("⚙️ Advanced options"):
            feature_only_retrain = st.toggle(
                "Feature-only retrain (risky)", value=False
            )
            if feature_only_retrain:
                st.warning(
                    "🟠 Feature-only retrain is risky: it may worsen "
                    "performance because no ground truth is available."
                )

        sensitivity = st.select_slider(
            "Drift sensitivity",
            options=["Loose", "Balanced", "Strict"],
            value="Balanced",
        )
        preset = SENSITIVITY_PRESETS[sensitivity]
        st.caption(
            f"PSI > {preset['psi_threshold']}  •  "
            f"KS p < {preset['ks_pvalue']}  •  "
            f"Feature drift > {int(preset['feature_drift_pct'] * 100)} %"
        )

        # Mode badge
        mode = detect_mode(ref_df, curr_df, target_col)
        st.divider()
        if mode == "supervised":
            st.success("✅ Supervised mode")
        elif mode == "monitoring":
            st.warning("⚠️ Monitoring mode — labels missing in Current")
        else:
            st.error("❌ Unsupervised-only mode")

# ═══════════════════════ WELCOME ═════════════════════════════════
if "ref_df" not in st.session_state or "curr_df" not in st.session_state:
    st.title("📊 P2M — ML Drift Monitoring")
    st.markdown(
        """
        Upload **two CSV datasets** in the sidebar to begin:

        | Step | Description |
        |------|-------------|
        | 1 | **Reference** — baseline / training data |
        | 2 | **Current** — new / production data |

        The application will automatically:
        - Compare schemas & data quality
        - Detect distribution drift (PSI, KS, Chi²)
        - Evaluate model performance (if labels available)
        - Propose and execute retraining (MLflow registry)
        """
    )
    st.stop()

# ═══════════════════════ DATA READY ══════════════════════════════
ref_df = st.session_state["ref_df"]
curr_df = st.session_state["curr_df"]
mode = detect_mode(ref_df, curr_df, target_col)

# Schema comparison (always)
schema = compare_schemas(ref_df, curr_df)
work_cols = schema.common_columns

# ── Tabs ─────────────────────────────────────────────────────────
tab_schema, tab_quality, tab_drift, tab_perf, tab_retrain = st.tabs(
    ["📋 Schema", "🔍 Data Quality", "📈 Drift", "📊 Performance", "🔄 Retrain & Registry"]
)

# ╔════════════════════════════════════════════════════════════════╗
# ║  TAB 1 — Schema                                              ║
# ╚════════════════════════════════════════════════════════════════╝
with tab_schema:
    st.header("Schema Comparison")

    c1, c2, c3 = st.columns(3)
    c1.metric("Common columns", len(schema.common_columns))
    c2.metric("Reference only", len(schema.ref_only_columns))
    c3.metric("Current only", len(schema.curr_only_columns))

    if schema.is_compatible:
        st.success("✅ Schemas are fully compatible.")
    else:
        if schema.ref_only_columns or schema.curr_only_columns:
            st.error(
                "🔴 Les datasets ne partagent pas le même schéma. "
                "Drift comparatif impossible tant que les colonnes ne matchent pas."
            )
        if schema.type_mismatches:
            st.warning(
                f"🟠 {len(schema.type_mismatches)} colonne(s) avec des types différents."
            )

    # Common columns detail
    with st.expander("Common columns", expanded=True):
        if schema.common_columns:
            st.dataframe(
                pd.DataFrame(
                    {
                        "Column": schema.common_columns,
                        "Ref Type": [str(ref_df[c].dtype) for c in schema.common_columns],
                        "Curr Type": [str(curr_df[c].dtype) for c in schema.common_columns],
                    }
                ),
                use_container_width=True,
            )
        else:
            st.info("No common columns found.")

    if schema.ref_only_columns:
        with st.expander("Columns only in Reference"):
            st.write(schema.ref_only_columns)
    if schema.curr_only_columns:
        with st.expander("Columns only in Current"):
            st.write(schema.curr_only_columns)
    if schema.type_mismatches:
        with st.expander("Type mismatches"):
            rows = [
                {"Column": c, "Ref Type": t[0], "Curr Type": t[1]}
                for c, t in schema.type_mismatches.items()
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ╔════════════════════════════════════════════════════════════════╗
# ║  TAB 2 — Data Quality                                        ║
# ╚════════════════════════════════════════════════════════════════╝
with tab_quality:
    st.header("Data Quality")

    for label, df in [("Reference", ref_df), ("Current", curr_df)]:
        st.subheader(f"{label} Dataset")

        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{len(df):,}")
        c2.metric("Columns", len(df.columns))
        c3.metric("Missing cells", f"{df.isnull().sum().sum():,}")

        # Types
        with st.expander("Column types"):
            st.dataframe(check_types(df), use_container_width=True)

        # Missing values
        missing = check_missing(df)
        if len(missing) > 0:
            with st.expander(f"🟠 Missing values — {len(missing)} column(s)"):
                st.dataframe(missing, use_container_width=True)
                fig = plot_missing_values(missing)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("✅ No missing values.")

        # Outliers
        outliers = detect_outliers(df)
        if len(outliers) > 0:
            with st.expander(f"🟡 Outliers — {len(outliers)} column(s)"):
                st.dataframe(outliers, use_container_width=True)
        else:
            st.info("✅ No outliers detected (IQR method).")

        # Target check
        if target_col and target_col in df.columns and df[target_col].isnull().any():
            st.warning(
                f"🟠 Target column '{target_col}' has "
                f"{df[target_col].isnull().sum()} missing values in {label}."
            )
        elif target_col and target_col not in df.columns:
            st.error(f"🔴 Target column '{target_col}' not found in {label} dataset.")

        st.divider()

# ╔════════════════════════════════════════════════════════════════╗
# ║  TAB 3 — Drift                                               ║
# ╚════════════════════════════════════════════════════════════════╝
drift_result = None  # used later in Retrain tab

with tab_drift:
    st.header("Drift Detection")

    if not work_cols:
        st.error("No common columns to compare. Fix schema issues first.")
    else:
        drift_result = detect_drift(
            ref_df,
            curr_df,
            columns=work_cols,
            sensitivity=sensitivity,
            target=target_col,
        )

        # KPI row
        c1, c2, c3 = st.columns(3)
        c1.metric("Drifted features", f"{drift_result.n_drifted} / {drift_result.n_total}")
        c2.metric("Drift %", f"{drift_result.drift_pct * 100:.1f} %")
        if drift_result.overall_drift:
            c3.metric("Overall drift", "⚠️ YES")
            st.error("🔴 Significant drift detected!")
        else:
            c3.metric("Overall drift", "✅ NO")
            st.success("✅ No significant drift detected.")

        # Summary chart
        fig = plot_drift_summary(drift_result)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

        # Detail table
        with st.expander("Detailed results per column"):
            detail = []
            for cr in drift_result.column_results:
                detail.append(
                    {
                        "Column": cr.column,
                        "Drifted": "🔴 Yes" if cr.drift_detected else "✅ No",
                        "Method": cr.method,
                        "Statistic": round(cr.statistic, 4),
                        "p-value": round(cr.pvalue, 4) if cr.pvalue is not None else "—",
                        "PSI": round(cr.psi, 4) if cr.psi is not None else "—",
                    }
                )
            st.dataframe(pd.DataFrame(detail), use_container_width=True)

        # Distribution comparison for drifted features
        drifted_cols = [
            cr.column for cr in drift_result.column_results if cr.drift_detected
        ]
        if drifted_cols:
            with st.expander(f"Distribution plots ({len(drifted_cols)} drifted features)"):
                for col in drifted_cols:
                    if col in ref_df.columns and col in curr_df.columns:
                        if pd.api.types.is_numeric_dtype(ref_df[col]):
                            fig = plot_distribution_comparison(
                                ref_df[col].dropna(), curr_df[col].dropna(), col
                            )
                            st.plotly_chart(fig, use_container_width=True)

# ╔════════════════════════════════════════════════════════════════╗
# ║  TAB 4 — Performance                                         ║
# ╚════════════════════════════════════════════════════════════════╝
with tab_perf:
    st.header("Model Performance")

    if mode == "unsupervised":
        st.error(
            "❌ Performance evaluation requires labeled data. "
            "Select a target column in the sidebar."
        )
        st.info("You can still use the **Drift** tab for unsupervised monitoring.")
    else:
        task_type = detect_task_type(ref_df[target_col])
        st.info(
            f"Detected task: **{task_type}**  —  "
            f"target: `{target_col}` ({ref_df[target_col].nunique()} unique values)"
        )

        model_options = list(MODELS[task_type].keys())
        selected_model = st.selectbox("Select model", model_options, key="perf_model")
        feature_cols = [c for c in work_cols if c != target_col]

        # ── Hyperparameter customization for evaluation ──────────────
        custom_params_eval = {}
        with st.expander("⚙️ Customize hyperparameters (optional)", expanded=False):
            st.write(f"**Model:** {selected_model}")

            default_params = get_default_params(selected_model)

            if selected_model == "Logistic Regression":
                custom_params_eval["C"] = st.slider(
                    "Regularization strength (C)", 0.1, 10.0, default_params.get("C", 1.0), 0.1, key="eval_C"
                )
                custom_params_eval["max_iter"] = st.slider(
                    "Max iterations", 100, 5000, default_params.get("max_iter", 1000), 100, key="eval_max_iter"
                )

            elif selected_model in ["Random Forest", "Gradient Boosting"]:
                custom_params_eval["n_estimators"] = st.slider(
                    "Number of estimators", 10, 500, default_params.get("n_estimators", 100), 10, key="eval_n_est"
                )
                custom_params_eval["max_depth"] = st.slider(
                    "Max depth", 2, 30, default_params.get("max_depth", 10), 1, key="eval_depth"
                )
                if selected_model == "Gradient Boosting":
                    custom_params_eval["learning_rate"] = st.slider(
                        "Learning rate", 0.01, 1.0, default_params.get("learning_rate", 0.1), 0.01, key="eval_lr"
                    )

            elif selected_model == "SVM":
                custom_params_eval["kernel"] = st.selectbox(
                    "Kernel", ["linear", "rbf", "poly"],
                    index=["linear", "rbf", "poly"].index(default_params.get("kernel", "rbf")), key="eval_kernel"
                )
                custom_params_eval["C"] = st.slider(
                    "Regularization (C)", 0.1, 100.0, default_params.get("C", 1.0), 0.1, key="eval_svm_c"
                )

            elif selected_model == "K-Nearest Neighbors":
                custom_params_eval["n_neighbors"] = st.slider(
                    "Number of neighbors", 1, 50, default_params.get("n_neighbors", 5), 1, key="eval_knn_n"
                )
                custom_params_eval["weights"] = st.selectbox(
                    "Weights", ["uniform", "distance"],
                    index=["uniform", "distance"].index(default_params.get("weights", "uniform")), key="eval_knn_w"
                )

            elif selected_model == "Decision Tree":
                custom_params_eval["max_depth"] = st.slider(
                    "Max depth", 2, 30, default_params.get("max_depth", 10), 1, key="eval_dt_depth"
                )
                custom_params_eval["min_samples_split"] = st.slider(
                    "Min samples to split", 2, 20, default_params.get("min_samples_split", 5), 1, key="eval_dt_split"
                )

            elif selected_model == "AdaBoost":
                custom_params_eval["n_estimators"] = st.slider(
                    "Number of estimators", 10, 200, default_params.get("n_estimators", 50), 10, key="eval_ada_n"
                )
                custom_params_eval["learning_rate"] = st.slider(
                    "Learning rate", 0.1, 2.0, default_params.get("learning_rate", 1.0), 0.1, key="eval_ada_lr"
                )

            # Display current parameters
            st.write("**Current parameters:**")
            st.json({**default_params, **custom_params_eval})

        if st.button("🚀 Train & Evaluate", key="train_eval"):
            # ── Train on Reference ──
            with st.spinner("Training model on Reference…"):
                try:
                    X_ref, y_ref, _ = prepare_features(ref_df, target_col, feature_cols)
                    model = train_model(X_ref, y_ref, task_type, selected_model, custom_params=custom_params_eval)
                    ref_perf = evaluate_model(model, X_ref, y_ref, task_type)

                    st.session_state["trained_model"] = model
                    st.session_state["X_ref_columns"] = X_ref.columns.tolist()
                    st.session_state["ref_perf"] = ref_perf
                    st.session_state["task_type"] = task_type
                    st.session_state["feature_cols"] = feature_cols
                except Exception as e:
                    st.error(f"Training failed: {e}")
                    st.stop()

            st.subheader("Reference (baseline) metrics")
            cols = st.columns(len(ref_perf.metrics))
            for i, (k, v) in enumerate(ref_perf.metrics.items()):
                cols[i].metric(k, f"{v:.4f}")

            # ── Evaluate on Current (supervised only) ──
            if mode == "supervised":
                with st.spinner("Evaluating on Current…"):
                    try:
                        X_curr, y_curr, _ = prepare_features(
                            curr_df, target_col, feature_cols
                        )
                        X_curr = align_features(X_ref, X_curr)
                        curr_perf = evaluate_model(model, X_curr, y_curr, task_type)
                        st.session_state["curr_perf"] = curr_perf
                    except Exception as e:
                        st.error(f"Current evaluation failed: {e}")
                        st.stop()

                st.subheader("Current metrics")
                cols = st.columns(len(curr_perf.metrics))
                for i, (k, v) in enumerate(curr_perf.metrics.items()):
                    cols[i].metric(k, f"{v:.4f}")

                # Comparison table
                st.subheader("Performance comparison")
                drops = check_performance_drop(ref_perf, curr_perf)
                drop_rows = []
                for metric_name, info in drops.items():
                    drop_rows.append(
                        {
                            "Metric": metric_name,
                            "Reference": info["ref"],
                            "Current": info["curr"],
                            "Change": f"{info['change']:+.4f}",
                            "Status": "🔴 Degraded" if info["degraded"] else "✅ Stable",
                        }
                    )
                st.dataframe(pd.DataFrame(drop_rows), use_container_width=True)

                any_drop = any(d["degraded"] for d in drops.values())
                if any_drop:
                    st.warning("🟠 Performance drop detected. Consider retraining.")
                else:
                    st.success("✅ Performance is stable.")
            else:
                st.info(
                    "⚠️ Current dataset has no labels — cannot evaluate on Current.\n\n"
                    "**Recommendation:** wait for labels and re-run evaluation."
                )

# ╔════════════════════════════════════════════════════════════════╗
# ║  TAB 5 — Retrain & Registry                                  ║
# ╚════════════════════════════════════════════════════════════════╝
with tab_retrain:
    st.header("Retrain & Model Registry")

    # ── Decision summary with cost analysis ──────────────────────
    st.subheader("🔍 Drift Analysis & Retrain Decision")

    drift_detected = drift_result.overall_drift if drift_result is not None else False
    perf_dropped = False

    if "curr_perf" in st.session_state and "ref_perf" in st.session_state:
        _drops = check_performance_drop(
            st.session_state["ref_perf"], st.session_state["curr_perf"]
        )
        perf_dropped = any(d["degraded"] for d in _drops.values())

    # Analyze drift topology
    topology = analyze_drift_topology(
        drift_result=drift_result,
        ref_perf=st.session_state.get("ref_perf"),
        curr_perf=st.session_state.get("curr_perf"),
        perf_threshold=0.05,
    )

    # Display drift topology
    st.subheader("Drift Topology")
    col1, col2 = st.columns(2)
    col1.metric("Drift Type", topology.drift_type)
    col2.metric("Data Drift Severity", topology.data_drift_severity)
    st.metric("Model/Concept Drift Severity", topology.model_drift_severity)
    st.info(f"**Interpretation:** {topology.interpretation}")
    st.write(f"**Action suggested:** {topology.action_suggested}")

    # Estimate business cost of drift
    st.divider()
    st.subheader("💰 Cost-Benefit Analysis")

    if "curr_perf" in st.session_state and "ref_perf" in st.session_state:
        # Estimate impact of drift (assuming 10,000 events as example)
        n_events = len(curr_df)  # Can be parameterized later
        cost_per_error = 10  # Currency units; can be adjusted

        drift_cost = estimate_drift_cost(
            st.session_state["ref_perf"],
            st.session_state["curr_perf"],
            n_events=n_events,
            cost_per_error=cost_per_error,
            task_type=st.session_state.get("task_type", "classification"),
        )

        # Estimate cost of retraining
        # Use a default model for cost estimation (Random Forest)
        retrain_cost = estimate_retrain_cost(
            model_name="Random Forest",
            n_samples=len(curr_df),
            hourly_cost=50,
            human_review_hours=2,
        )

        # Make cost-based decision
        cost_decision = make_retrain_decision(drift_cost, retrain_cost, threshold_ratio=1.5)

        col1, col2, col3 = st.columns(3)
        col1.metric("Est. Drift Cost (over ~10k events)", f"{drift_cost:,.0f} units")
        col2.metric("Est. Retrain Cost", f"{retrain_cost:,.0f} units")
        col3.metric("Net Benefit of Retrain", f"{cost_decision.net_benefit:,.0f} units")

        st.info(f"💡 {cost_decision.recommendation}")
    else:
        st.warning("Run Performance evaluation first to estimate drift cost.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Drift status", "⚠️ YES" if drift_detected else "✅ NO")
    c2.metric(
        "Performance drop",
        "⚠️ YES"
        if perf_dropped
        else ("✅ Stable" if mode == "supervised" else "— N/A"),
    )

    if drift_detected and perf_dropped:
        c3.metric("Retrain decision", "✅ Triggered")
        st.error("🔴 Drift + Performance drop → **Retrain recommended**")
    elif drift_detected and mode != "supervised":
        c3.metric("Retrain decision", "⚠️ Suggested")
        st.warning("🟠 Drift detected but no labels to confirm performance drop.")
    elif drift_detected:
        c3.metric("Retrain decision", "⚠️ Monitor")
        st.warning("🟠 Drift detected but performance stable. Monitor closely.")
    else:
        c3.metric("Retrain decision", "❌ Not needed")
        st.success("✅ No retrain needed — drift and performance are stable.")

    st.divider()

    # ── Retrain controls ─────────────────────────────────────────
    if mode == "unsupervised" and not feature_only_retrain:
        st.error(
            "❌ Retrain supervisé impossible sans labels.\n\n"
            "Options :\n"
            "1. Fournir des données labellisées\n"
            "2. Activer *Feature-only retrain* dans les options avancées (risqué)"
        )

    elif mode == "monitoring" and not feature_only_retrain:
        st.warning(
            "⚠️ Retrain supervisé : ❌ impossible (labels manquants dans Current)\n\n"
            "**Action recommandée :** attendre l'arrivée des labels et relancer le retrain.\n\n"
            "Activez *Feature-only retrain* dans les options avancées pour forcer un "
            "retrain sur les données Reference uniquement."
        )

    else:
        # supervised  OR  feature_only_retrain enabled
        if target_col:
            task_type = st.session_state.get(
                "task_type", detect_task_type(ref_df[target_col])
            )
        else:
            task_type = "classification"  # fallback for feature-only
        target_available_for_supervised = target_col is not None and target_col in ref_df.columns

        # ── AI agent: best model + params ───────────────────────────────
        st.subheader("🤖 AI Agent: Recommend Best Model & Parameters")

        col1, col2, col3, col4 = st.columns(4)
        use_quick_mode = col1.checkbox("⚡ Quick mode (faster)", value=True)
        objective = col2.selectbox(
            "Optimization objective",
            ["score_first", "balanced", "fast"],
            index=0,
            help="score_first = best CV score, balanced = score + stability + time, fast = speed-aware",
        )
        metric_options = ["accuracy", "f1_weighted"] if task_type == "classification" else ["neg_mean_squared_error", "r2"]
        default_metric = "f1_weighted" if task_type == "classification" else "neg_mean_squared_error"
        scoring_metric = col3.selectbox(
            "CV metric",
            metric_options,
            index=metric_options.index(default_metric),
        )
        include_lstm_search = col4.checkbox(
            "Include LSTM",
            value=False,
            help="Enable only for sequential/time-series data.",
        )
        find_best = st.button("🔍 Run AI Agent", key="find_best_btn")

        if not target_available_for_supervised:
            st.warning("Select a valid target column to run supervised model recommendation.")

        if find_best:
            with st.spinner("AI agent is evaluating model families and hyperparameters..."):
                try:
                    if not target_available_for_supervised:
                        raise ValueError("Target column is required for supervised model selection.")

                    feature_cols = [c for c in work_cols if c != target_col]
                    X_analysis, y_analysis, _ = prepare_features(ref_df, target_col, feature_cols)

                    context = {
                        "drift_detected": drift_detected,
                        "perf_dropped": perf_dropped,
                        "drift_pct": drift_result.drift_pct if drift_result is not None else 0.0,
                        "drift_severity": topology.data_drift_severity,
                        "need_fast_retrain": objective == "fast",
                    }

                    agent = ModelSelectionAgent(
                        task_type=task_type,
                        cv_folds=3,
                        quick_mode=use_quick_mode,
                        objective=objective,
                        scoring_metric=scoring_metric,
                        context=context,
                        include_lstm=include_lstm_search,
                    )
                    report = agent.recommend(X_analysis, y_analysis, top_k=5)

                    st.success(
                        f"✅ AI agent tested candidates and produced {len(report.leaderboard)} top recommendations."
                    )

                    display_data = []
                    for i, rec in enumerate(report.leaderboard, 1):
                        display_data.append({
                            "Rank": f"#{i}",
                            "Model": rec.model_name,
                            "CV Score": f"{rec.cv_score:.4f}",
                            "Std": f"{rec.cv_std:.4f}",
                            "Time (s)": f"{rec.training_time:.2f}",
                            "Params": str(rec.params),
                        })

                    st.subheader("Top AI Recommendations")
                    st.dataframe(pd.DataFrame(display_data), use_container_width=True)

                    best_rec = report.best
                    st.info(
                        f"**🏆 Best Recommendation:** {best_rec.model_name}\n\n"
                        f"Score ({best_rec.scoring_metric}): **{best_rec.cv_score:.4f}** +/- {best_rec.cv_std:.4f}\n\n"
                        f"Recommended parameters: `{best_rec.params}`\n\n"
                        f"Why this model: {report.rationale}"
                    )

                    st.session_state["best_recommendation"] = best_rec
                    st.session_state["all_recommendations"] = report.leaderboard

                except Exception as e:
                    st.error(f"Error running AI agent: {e}")

        # Display stored recommendations and select model
        if "best_recommendation" in st.session_state:
            st.divider()
            best_rec = st.session_state["best_recommendation"]

            # Use best recommendation checkbox
            use_best = st.checkbox(
                "✅ Use the best recommendation",
                value=True,
                key="use_best_recommendation"
            )

            if use_best:
                retrain_model_name = best_rec.model_name
                auto_params = best_rec.params
                st.success(f"✅ Using: **{retrain_model_name}** with params {auto_params}")
            else:
                model_options = list(MODELS[task_type].keys())
                retrain_model_name = st.selectbox(
                    "Or choose another model manually", model_options, key="retrain_model"
                )
                auto_params = None
        else:
            auto_params = None
            model_options = list(MODELS[task_type].keys())
            retrain_model_name = st.selectbox(
                "Model for retrain", model_options, key="retrain_model"
            )

        # ── Hyperparameter customization ────────────────────────────
        with st.expander("⚙️ Customize hyperparameters (optional)", expanded=False):
            st.write(f"**Model:** {retrain_model_name}")

            default_params = get_default_params(retrain_model_name)
            defaults = default_params.copy()

            # Show recommended params if available
            best_from_state = st.session_state.get("best_recommendation")
            if auto_params and best_from_state and retrain_model_name == best_from_state.model_name:
                st.info(f"📊 Agent-recommended params: {auto_params}")
                use_auto = st.checkbox("✅ Prefill form with recommended parameters", value=True, key="use_auto_params")
                if use_auto:
                    defaults.update(auto_params)

            custom_params = {}

            if retrain_model_name == "Logistic Regression":
                custom_params["C"] = st.slider(
                    "Regularization strength (C)", 0.1, 10.0, defaults.get("C", 1.0), 0.1
                )
                custom_params["max_iter"] = st.slider(
                    "Max iterations", 100, 5000, defaults.get("max_iter", 1000), 100
                )

            elif retrain_model_name in ["Random Forest", "Gradient Boosting"]:
                custom_params["n_estimators"] = st.slider(
                    "Number of estimators", 10, 500, defaults.get("n_estimators", 100), 10
                )
                custom_params["max_depth"] = st.slider(
                    "Max depth", 2, 30, defaults.get("max_depth", 10), 1
                )
                if retrain_model_name == "Gradient Boosting":
                    custom_params["learning_rate"] = st.slider(
                        "Learning rate", 0.01, 1.0, defaults.get("learning_rate", 0.1), 0.01
                    )

            elif retrain_model_name == "SVM":
                custom_params["kernel"] = st.selectbox(
                    "Kernel", ["linear", "rbf", "poly"],
                    index=["linear", "rbf", "poly"].index(defaults.get("kernel", "rbf"))
                )
                custom_params["C"] = st.slider(
                    "Regularization (C)", 0.1, 100.0, defaults.get("C", 1.0), 0.1
                )

            elif retrain_model_name == "K-Nearest Neighbors":
                custom_params["n_neighbors"] = st.slider(
                    "Number of neighbors", 1, 50, defaults.get("n_neighbors", 5), 1
                )
                custom_params["weights"] = st.selectbox(
                    "Weights", ["uniform", "distance"],
                    index=["uniform", "distance"].index(defaults.get("weights", "uniform"))
                )

            elif retrain_model_name == "Decision Tree":
                custom_params["max_depth"] = st.slider(
                    "Max depth", 2, 30, defaults.get("max_depth", 10), 1
                )
                custom_params["min_samples_split"] = st.slider(
                    "Min samples to split", 2, 20, defaults.get("min_samples_split", 5), 1
                )

            elif retrain_model_name == "AdaBoost":
                custom_params["n_estimators"] = st.slider(
                    "Number of estimators", 10, 200, defaults.get("n_estimators", 50), 10
                )
                custom_params["learning_rate"] = st.slider(
                    "Learning rate", 0.1, 2.0, defaults.get("learning_rate", 1.0), 0.1
                )

            elif retrain_model_name == "LSTM":
                custom_params["units"] = st.slider(
                    "Units (LSTM layer)", 16, 128, defaults.get("units", 32), 16
                )
                custom_params["epochs"] = st.slider(
                    "Epochs (training iterations)", 5, 50, defaults.get("epochs", 20), 1
                )
                custom_params["batch_size"] = st.selectbox(
                    "Batch size",
                    [16, 32, 64, 128],
                    index=[16, 32, 64, 128].index(defaults.get("batch_size", 32))
                )
                st.info("💡 **LSTM Note**: This is a neural network model suitable for sequential/temporal patterns. Training may take longer.")

            # Display current parameters
            st.write("**Current parameters:**")
            st.json({**defaults, **custom_params})

        if mode == "supervised":
            train_on = st.radio(
                "Training data",
                ["Reference only", "Current only", "Combined (Reference + Current)"],
                index=2,
            )
        else:
            train_on = "Reference only"
            st.info("ℹ️ Retrain will use **Reference data only** (labels unavailable in Current).")

        if st.button("🔄 Retrain model", key="retrain_btn"):
            feature_cols = [c for c in work_cols if c != target_col]

            with st.spinner("Retraining with MLflow logging…"):
                try:
                    if not target_available_for_supervised:
                        raise ValueError("Target column is required for supervised retraining.")

                    if train_on == "Reference only":
                        train_df = ref_df
                    elif train_on == "Current only":
                        train_df = curr_df
                    else:
                        train_df = pd.concat(
                            [ref_df[work_cols], curr_df[work_cols]], ignore_index=True
                        )

                    X, y, _ = prepare_features(train_df, target_col, feature_cols)
                    result = retrain_with_mlflow(
                        X,
                        y,
                        task_type,
                        retrain_model_name,
                        custom_params=custom_params,
                        scoring_metric=scoring_metric,
                    )
                    st.session_state["retrain_result"] = result

                    st.success(
                        f"✅ Model retrained and logged to MLflow!  \n"
                        f"Run ID: `{result.run_id}`"
                    )

                    mcols = st.columns(len(result.metrics))
                    for i, (k, v) in enumerate(result.metrics.items()):
                        mcols[i].metric(k, f"{v:.4f}")

                    # Evaluate retrained model on Current dataset and compare
                    # with previous metrics (if available)
                    new_curr_perf = None
                    if mode == "supervised" and target_col:
                        try:
                            X_curr_eval, y_curr_eval, _ = prepare_features(
                                curr_df, target_col, feature_cols
                            )
                            X_curr_eval = align_features(X, X_curr_eval)
                            new_curr_perf = evaluate_model(
                                result.model, X_curr_eval, y_curr_eval, task_type
                            )
                            st.session_state["retrain_curr_perf"] = new_curr_perf
                        except Exception as e:
                            st.warning(
                                f"Retrained model evaluation on Current failed: {e}"
                            )

                    if new_curr_perf is not None and "curr_perf" in st.session_state:
                        st.subheader("Current dataset — old vs new metrics")
                        old_perf = st.session_state["curr_perf"]
                        rows = []
                        for metric_name, new_val in new_curr_perf.metrics.items():
                            old_val = old_perf.metrics.get(metric_name)
                            change = (
                                new_val - old_val if old_val is not None else None
                            )
                            rows.append(
                                {
                                    "Metric": metric_name,
                                    "Old": round(old_val, 4)
                                    if old_val is not None
                                    else "—",
                                    "New": round(new_val, 4),
                                    "Change": f"{change:+.4f}" if change is not None else "—",
                                }
                            )
                        st.dataframe(pd.DataFrame(rows), use_container_width=True)

                        deploy_check = deployment_readiness(old_perf, new_curr_perf)
                        st.session_state["deploy_check"] = deploy_check

                        if deploy_check["can_deploy"]:
                            st.success(
                                f"✅ Deploy eligible: {deploy_check['reason']}"
                            )
                        else:
                            st.warning(
                                f"⚠️ Deploy blocked: {deploy_check['reason']}"
                            )

                        if deploy_check["can_deploy"] and st.button("🚀 Deploy retrained model", key="deploy_model_btn"):
                            st.session_state["deployed_model"] = result.model
                            st.session_state["deployed_model_name"] = result.model_name
                            st.session_state["deployed_run_id"] = result.run_id
                            st.session_state["deployed_task_type"] = result.task_type
                            st.session_state["deployed_metrics"] = new_curr_perf.metrics
                            st.success(
                                f"🚀 Model deployed: {result.model_name} (run {result.run_id})"
                            )

                except Exception as e:
                    st.error(f"Retrain failed: {e}")

    # ── MLflow information ───────────────────────────────────────
    st.divider()
    st.subheader("📦 Model Registry (MLflow)")
    st.info(
        "MLflow tracking is stored locally in `./mlruns`.  \n"
        "To browse runs:  \n"
        "```\n"
        "mlflow ui\n"
        "```\n"
        "Then open **http://localhost:5000**"
    )

    if "retrain_result" in st.session_state:
        r = st.session_state["retrain_result"]
        st.markdown(
            f"**Last retrain:** `{r.model_name}` ({r.task_type})  \n"
            f"Run ID: `{r.run_id}`"
        )
        for k, v in r.metrics.items():
            st.write(f"- {k}: **{v:.4f}**")

    if "deployed_run_id" in st.session_state:
        st.success(
            f"Production model: {st.session_state.get('deployed_model_name')} "
            f"(run {st.session_state.get('deployed_run_id')})"
        )
