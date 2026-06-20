"""
Hospitality Loyalty Campaign Propensity & Segmentation Analytics

End-to-end hospitality campaign analytics workflow using synthetic resort loyalty data.
The pipeline covers data preparation, QA, feature engineering, propensity modeling,
decile scoring, segmentation, A/B test setup, and dashboard-ready outputs.

Run from the project root:
    python src/build_project.py
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHART_DIR = OUTPUT_DIR / "charts"
SQL_DIR = PROJECT_ROOT / "sql"

AS_OF_DATE = pd.Timestamp("2026-06-20")
RANDOM_SEED = 42
N_CUSTOMERS = 20_000


def ensure_dirs() -> None:
    for path in [RAW_DIR, PROCESSED_DIR, OUTPUT_DIR, CHART_DIR, SQL_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def logistic(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def generate_synthetic_loyalty_data(n: int = N_CUSTOMERS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Create a realistic synthetic resort loyalty campaign dataset."""
    rng = np.random.default_rng(seed)

    customer_id = np.arange(100000, 100000 + n)
    tenure_days = rng.integers(30, 1_460, size=n)
    signup_date = AS_OF_DATE - pd.to_timedelta(tenure_days, unit="D")

    home_market = rng.choice(
        ["UAE", "GCC", "India", "UK/EU", "North America", "Other"],
        size=n,
        p=[0.32, 0.18, 0.18, 0.14, 0.10, 0.08],
    )
    preferred_channel = rng.choice(
        ["Email", "SMS", "WhatsApp", "App Push", "Phone"],
        size=n,
        p=[0.38, 0.20, 0.22, 0.15, 0.05],
    )

    # Latent customer value and engagement levels.
    value_score = rng.gamma(shape=2.0, scale=1.0, size=n)
    engagement_score = rng.beta(a=2.2, b=3.0, size=n)
    trip_intensity = rng.gamma(shape=1.7, scale=1.0, size=n)

    visits_30d = rng.poisson(lam=np.clip(0.3 + 0.7 * trip_intensity, 0.05, 8), size=n)
    visits_90d = visits_30d + rng.poisson(lam=np.clip(0.7 + 1.2 * trip_intensity, 0.05, 12), size=n)
    visits_180d = visits_90d + rng.poisson(lam=np.clip(1.2 + 1.8 * trip_intensity, 0.05, 18), size=n)

    hotel_nights_180d = rng.poisson(lam=np.clip(0.2 + 0.55 * trip_intensity + 0.15 * value_score, 0.05, 12), size=n)
    dining_visits_90d = rng.poisson(lam=np.clip(0.4 + 1.3 * trip_intensity, 0.05, 15), size=n)
    entertainment_visits_90d = rng.poisson(lam=np.clip(0.2 + 0.8 * trip_intensity, 0.05, 10), size=n)
    spa_visits_90d = rng.poisson(lam=np.clip(0.08 + 0.18 * value_score, 0.01, 5), size=n)

    total_spend_180d = (
        rng.gamma(shape=2.0 + value_score, scale=340, size=n)
        + hotel_nights_180d * rng.normal(900, 120, size=n).clip(300, 2_000)
        + dining_visits_90d * rng.normal(160, 35, size=n).clip(50, 400)
        + spa_visits_90d * rng.normal(450, 90, size=n).clip(100, 900)
    ).round(2)
    avg_spend_per_visit = np.divide(
        total_spend_180d,
        np.maximum(visits_180d, 1),
    ).round(2)

    last_visit_days_ago = rng.gamma(shape=2.0, scale=25, size=n).astype(int)
    last_visit_days_ago = np.clip(last_visit_days_ago - visits_30d * 4, 0, 365)
    email_open_rate = np.clip(rng.normal(0.22 + 0.45 * engagement_score, 0.12, size=n), 0, 1).round(3)
    promotion_clicks_90d = rng.poisson(lam=np.clip(0.1 + 2.8 * engagement_score, 0, 8), size=n)
    prior_offer_count_180d = rng.poisson(lam=np.clip(0.6 + 1.4 * engagement_score + 0.15 * trip_intensity, 0, 10), size=n)
    prior_offer_redemptions_180d = rng.binomial(
        n=np.maximum(prior_offer_count_180d, 0),
        p=np.clip(0.08 + 0.35 * engagement_score + 0.04 * trip_intensity, 0.02, 0.75),
    )
    service_cases_180d = rng.poisson(lam=np.clip(0.05 + 0.08 * visits_180d, 0.01, 3), size=n)
    weekend_visit_share = np.clip(rng.normal(0.45 + 0.12 * engagement_score, 0.16, size=n), 0, 1).round(3)
    booking_lead_days_avg = np.clip(rng.normal(13 + 3.5 * value_score, 8, size=n), 0, 80).round(1)

    # Tiers based on spend; this approximates loyalty status for targeting segments.
    tier_bins = np.quantile(total_spend_180d, [0, 0.55, 0.80, 0.94, 1.0])
    loyalty_tier = pd.cut(
        total_spend_180d,
        bins=tier_bins,
        labels=["Bronze", "Silver", "Gold", "Platinum"],
        include_lowest=True,
        duplicates="drop",
    ).astype(str)

    # Synthetic response probability: recent, engaged, high-value guests with prior redemptions respond more.
    tier_boost = pd.Series(loyalty_tier).map({"Bronze": 0.0, "Silver": 0.25, "Gold": 0.45, "Platinum": 0.62}).to_numpy()
    logit = (
        -2.55
        + 0.018 * visits_90d
        + 0.00013 * total_spend_180d
        + 1.85 * email_open_rate
        + 0.22 * np.minimum(promotion_clicks_90d, 5)
        + 0.35 * np.minimum(prior_offer_redemptions_180d, 3)
        - 0.006 * last_visit_days_ago
        - 0.13 * np.maximum(prior_offer_count_180d - 4, 0)
        - 0.10 * service_cases_180d
        + tier_boost
    )
    response_probability = np.clip(logistic(logit), 0.01, 0.88)
    responded_to_campaign = rng.binomial(1, response_probability)

    df = pd.DataFrame(
        {
            "customer_id": customer_id,
            "signup_date": signup_date.date.astype(str),
            "home_market": home_market,
            "preferred_channel": preferred_channel,
            "loyalty_tier": loyalty_tier,
            "last_visit_days_ago": last_visit_days_ago,
            "visits_30d": visits_30d,
            "visits_90d": visits_90d,
            "visits_180d": visits_180d,
            "hotel_nights_180d": hotel_nights_180d,
            "dining_visits_90d": dining_visits_90d,
            "entertainment_visits_90d": entertainment_visits_90d,
            "spa_visits_90d": spa_visits_90d,
            "total_spend_180d": total_spend_180d,
            "avg_spend_per_visit": avg_spend_per_visit,
            "email_open_rate": email_open_rate,
            "promotion_clicks_90d": promotion_clicks_90d,
            "prior_offer_count_180d": prior_offer_count_180d,
            "prior_offer_redemptions_180d": prior_offer_redemptions_180d,
            "service_cases_180d": service_cases_180d,
            "weekend_visit_share": weekend_visit_share,
            "booking_lead_days_avg": booking_lead_days_avg,
            "responded_to_campaign": responded_to_campaign,
        }
    )

    # Inject a small number of realistic data quality issues to demonstrate QA checks.
    missing_open_idx = rng.choice(df.index, size=int(0.015 * n), replace=False)
    df.loc[missing_open_idx, "email_open_rate"] = np.nan

    negative_spend_idx = rng.choice(df.index.difference(missing_open_idx), size=int(0.004 * n), replace=False)
    df.loc[negative_spend_idx, "total_spend_180d"] = -1 * rng.uniform(50, 600, size=len(negative_spend_idx)).round(2)

    high_visit_idx = rng.choice(df.index, size=int(0.003 * n), replace=False)
    df.loc[high_visit_idx, "visits_180d"] = rng.integers(80, 140, size=len(high_visit_idx))

    duplicate_rows = df.sample(n=40, random_state=seed).copy()
    df = pd.concat([df, duplicate_rows], ignore_index=True)

    return df


def write_sql_feature_query() -> None:
    query = """
-- customer_features.sql
-- Feature engineering query for campaign propensity scoring.
-- The query creates clean RFM, value, engagement, and targeting flags from raw loyalty data.

SELECT
    customer_id,
    signup_date,
    home_market,
    preferred_channel,
    loyalty_tier,
    CASE loyalty_tier
        WHEN 'Bronze' THEN 1
        WHEN 'Silver' THEN 2
        WHEN 'Gold' THEN 3
        WHEN 'Platinum' THEN 4
        ELSE 0
    END AS loyalty_tier_rank,
    CASE WHEN last_visit_days_ago < 0 OR last_visit_days_ago > 365 THEN NULL ELSE last_visit_days_ago END AS recency_days,
    CASE WHEN visits_30d < 0 THEN 0 ELSE visits_30d END AS visits_30d,
    CASE WHEN visits_90d < 0 THEN 0 ELSE visits_90d END AS visits_90d,
    CASE WHEN visits_180d < 0 THEN 0 WHEN visits_180d > 75 THEN 75 ELSE visits_180d END AS visits_180d_capped,
    CASE WHEN hotel_nights_180d < 0 THEN 0 ELSE hotel_nights_180d END AS hotel_nights_180d,
    CASE WHEN dining_visits_90d < 0 THEN 0 ELSE dining_visits_90d END AS dining_visits_90d,
    CASE WHEN entertainment_visits_90d < 0 THEN 0 ELSE entertainment_visits_90d END AS entertainment_visits_90d,
    CASE WHEN spa_visits_90d < 0 THEN 0 ELSE spa_visits_90d END AS spa_visits_90d,
    CASE WHEN total_spend_180d < 0 THEN NULL ELSE total_spend_180d END AS spend_180d_clean,
    CASE WHEN avg_spend_per_visit < 0 THEN NULL ELSE avg_spend_per_visit END AS avg_spend_per_visit_clean,
    CASE WHEN email_open_rate < 0 OR email_open_rate > 1 THEN NULL ELSE email_open_rate END AS email_open_rate_clean,
    promotion_clicks_90d,
    prior_offer_count_180d,
    prior_offer_redemptions_180d,
    service_cases_180d,
    weekend_visit_share,
    booking_lead_days_avg,
    CASE WHEN visits_90d >= 4 THEN 1 ELSE 0 END AS active_guest_flag,
    CASE WHEN last_visit_days_ago >= 90 THEN 1 ELSE 0 END AS reactivation_flag,
    CASE WHEN prior_offer_count_180d >= 5 THEN 1 ELSE 0 END AS offer_fatigue_flag,
    CASE WHEN total_spend_180d >= 3000 THEN 1 ELSE 0 END AS high_value_flag,
    CASE WHEN email_open_rate >= 0.45 OR promotion_clicks_90d >= 2 THEN 1 ELSE 0 END AS engaged_marketing_flag,
    responded_to_campaign
FROM raw_loyalty;
""".strip()
    (SQL_DIR / "customer_features.sql").write_text(query, encoding="utf-8")


def load_raw_to_sqlite(raw_csv: Path, db_path: Path) -> None:
    df = pd.read_csv(raw_csv)
    with sqlite3.connect(db_path) as conn:
        df.to_sql("raw_loyalty", conn, if_exists="replace", index=False)


def build_features_from_sql(db_path: Path) -> pd.DataFrame:
    query = (SQL_DIR / "customer_features.sql").read_text(encoding="utf-8")
    with sqlite3.connect(db_path) as conn:
        features = pd.read_sql_query(query, conn)
    return features.drop_duplicates(subset=["customer_id"], keep="first")


def make_data_quality_report(raw: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    checks = [
        ("raw_row_count", len(raw), "information"),
        ("unique_customers", raw["customer_id"].nunique(), "information"),
        ("duplicate_customer_rows", int(raw.duplicated(subset=["customer_id"]).sum()), "review"),
        ("missing_email_open_rate", int(raw["email_open_rate"].isna().sum()), "impute_with_median"),
        ("negative_total_spend_180d", int((raw["total_spend_180d"] < 0).sum()), "set_to_null_then_impute"),
        ("extreme_visits_180d_over_75", int((raw["visits_180d"] > 75).sum()), "cap_at_75"),
        ("feature_row_count_after_dedup", len(features), "information"),
        ("missing_spend_after_sql_cleaning", int(features["spend_180d_clean"].isna().sum()), "model_pipeline_imputes"),
        ("missing_email_after_sql_cleaning", int(features["email_open_rate_clean"].isna().sum()), "model_pipeline_imputes"),
    ]
    return pd.DataFrame(checks, columns=["check_name", "value", "action_taken"])


def get_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric_features = [
        "loyalty_tier_rank",
        "recency_days",
        "visits_30d",
        "visits_90d",
        "visits_180d_capped",
        "hotel_nights_180d",
        "dining_visits_90d",
        "entertainment_visits_90d",
        "spa_visits_90d",
        "spend_180d_clean",
        "avg_spend_per_visit_clean",
        "email_open_rate_clean",
        "promotion_clicks_90d",
        "prior_offer_count_180d",
        "prior_offer_redemptions_180d",
        "service_cases_180d",
        "weekend_visit_share",
        "booking_lead_days_avg",
        "active_guest_flag",
        "reactivation_flag",
        "offer_fatigue_flag",
        "high_value_flag",
        "engaged_marketing_flag",
    ]
    categorical_features = ["home_market", "preferred_channel", "loyalty_tier"]
    return numeric_features, categorical_features


def make_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ]
    )


def evaluate_model(name: str, model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float | str]:
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "model": name,
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, pred, zero_division=0), 4),
        "baseline_response_rate": round(float(y_test.mean()), 4),
    }


def train_models(features: pd.DataFrame) -> Tuple[Pipeline, pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame, List[str], List[str]]:
    numeric_features, categorical_features = get_feature_columns(features)
    X = features[numeric_features + categorical_features].copy()
    y = features["responded_to_campaign"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_SEED, stratify=y
    )

    logistic_model = Pipeline(
        steps=[
            ("preprocess", make_preprocessor(numeric_features, categorical_features)),
            ("model", LogisticRegression(max_iter=1_000, class_weight="balanced", random_state=RANDOM_SEED)),
        ]
    )
    random_forest_model = Pipeline(
        steps=[
            ("preprocess", make_preprocessor(numeric_features, categorical_features)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=180,
                    max_depth=9,
                    min_samples_leaf=35,
                    class_weight="balanced_subsample",
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    models = {
        "Logistic Regression": logistic_model,
        "Random Forest": random_forest_model,
    }
    metrics = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics.append(evaluate_model(name, model, X_test, y_test))

    metrics_df = pd.DataFrame(metrics).sort_values("roc_auc", ascending=False)
    best_name = metrics_df.iloc[0]["model"]
    best_model = models[best_name]

    test_scored = X_test.copy()
    test_scored["responded_to_campaign"] = y_test.values
    test_scored["predicted_response_probability"] = best_model.predict_proba(X_test)[:, 1]
    test_scored["score_decile"] = pd.qcut(
        test_scored["predicted_response_probability"].rank(method="first"),
        10,
        labels=False,
    )
    test_scored["score_decile"] = 10 - test_scored["score_decile"]

    return best_model, metrics_df, X, y, test_scored, numeric_features, categorical_features


def make_lift_table(test_scored: pd.DataFrame) -> pd.DataFrame:
    baseline = test_scored["responded_to_campaign"].mean()
    lift = (
        test_scored.groupby("score_decile")
        .agg(
            customers=("responded_to_campaign", "size"),
            response_rate=("responded_to_campaign", "mean"),
            avg_predicted_probability=("predicted_response_probability", "mean"),
        )
        .reset_index()
        .sort_values("score_decile")
    )
    lift["lift_vs_baseline"] = lift["response_rate"] / baseline
    lift["response_rate"] = lift["response_rate"].round(4)
    lift["avg_predicted_probability"] = lift["avg_predicted_probability"].round(4)
    lift["lift_vs_baseline"] = lift["lift_vs_baseline"].round(2)
    return lift


def score_all_customers(best_model: Pipeline, features: pd.DataFrame, numeric_features: List[str], categorical_features: List[str]) -> pd.DataFrame:
    X_all = features[numeric_features + categorical_features].copy()
    scored = features.copy()
    scored["predicted_response_probability"] = best_model.predict_proba(X_all)[:, 1]
    scored["score_decile"] = pd.qcut(
        scored["predicted_response_probability"].rank(method="first"), 10, labels=False
    )
    scored["score_decile"] = 10 - scored["score_decile"]

    def assign_segment(row: pd.Series) -> str:
        if row["score_decile"] <= 2 and row["high_value_flag"] == 1:
            return "High Propensity - High Value"
        if row["score_decile"] <= 3:
            return "High Propensity - Standard Value"
        if row["reactivation_flag"] == 1 and row["engaged_marketing_flag"] == 1:
            return "Reactivation Test Cell"
        if row["offer_fatigue_flag"] == 1 and row["score_decile"] >= 7:
            return "Low Priority - Suppress"
        if row["score_decile"] <= 6:
            return "Mid Propensity - Test Offer"
        return "Low Propensity - Holdout Candidate"

    scored["campaign_segment"] = scored.apply(assign_segment, axis=1)
    scored["recommended_channel"] = np.where(
        scored["preferred_channel"].isin(["WhatsApp", "SMS", "App Push"]),
        scored["preferred_channel"],
        "Email",
    )
    scored["qa_ready_flag"] = np.where(
        scored[["customer_id", "predicted_response_probability", "campaign_segment"]].notna().all(axis=1),
        1,
        0,
    )
    return scored


def make_segment_summary(scored: pd.DataFrame) -> pd.DataFrame:
    summary = (
        scored.groupby("campaign_segment")
        .agg(
            customers=("customer_id", "count"),
            avg_score=("predicted_response_probability", "mean"),
            historical_response_rate=("responded_to_campaign", "mean"),
            avg_spend_180d=("spend_180d_clean", "mean"),
            qa_ready_rate=("qa_ready_flag", "mean"),
        )
        .reset_index()
        .sort_values("avg_score", ascending=False)
    )
    for col in ["avg_score", "historical_response_rate", "avg_spend_180d", "qa_ready_rate"]:
        summary[col] = summary[col].round(4 if col != "avg_spend_180d" else 2)
    return summary


def make_ab_test_offer_calibration(scored: pd.DataFrame, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    eligible = scored[scored["campaign_segment"].isin([
        "High Propensity - High Value",
        "High Propensity - Standard Value",
        "Mid Propensity - Test Offer",
        "Reactivation Test Cell",
    ])].copy()
    eligible["test_group"] = rng.choice(["Treatment", "Control"], size=len(eligible), p=[0.70, 0.30])

    def offer(row: pd.Series) -> str:
        if row["campaign_segment"] == "High Propensity - High Value":
            return "Premium stay/dining bundle"
        if row["campaign_segment"] == "Reactivation Test Cell":
            return "Reactivation dining offer"
        if row["score_decile"] <= 3:
            return "Standard loyalty offer"
        return "Low-cost reminder offer"

    eligible["offer_type"] = eligible.apply(offer, axis=1)
    treatment_uplift = np.where(
        eligible["offer_type"].eq("Premium stay/dining bundle"), 0.055,
        np.where(eligible["offer_type"].eq("Reactivation dining offer"), 0.045,
                 np.where(eligible["offer_type"].eq("Standard loyalty offer"), 0.035, 0.018))
    )
    base_probability = eligible["predicted_response_probability"].to_numpy()
    simulated_probability = np.where(
        eligible["test_group"].eq("Treatment"),
        np.clip(base_probability + treatment_uplift, 0, 0.95),
        base_probability,
    )
    eligible["simulated_campaign_response"] = rng.binomial(1, simulated_probability)
    eligible["simulated_revenue_aed"] = np.where(
        eligible["simulated_campaign_response"].eq(1),
        np.maximum(eligible["spend_180d_clean"].fillna(eligible["spend_180d_clean"].median()) * rng.uniform(0.08, 0.18, len(eligible)), 100),
        0,
    ).round(2)

    summary = (
        eligible.groupby(["offer_type", "test_group"])
        .agg(
            customers=("customer_id", "count"),
            response_rate=("simulated_campaign_response", "mean"),
            revenue_aed=("simulated_revenue_aed", "sum"),
            avg_score=("predicted_response_probability", "mean"),
        )
        .reset_index()
    )
    pivot = summary.pivot(index="offer_type", columns="test_group", values="response_rate").reset_index()
    pivot.columns.name = None
    pivot = pivot.rename(columns={"Control": "control_response_rate", "Treatment": "treatment_response_rate"})
    pivot["incremental_response_rate"] = pivot["treatment_response_rate"] - pivot["control_response_rate"]

    counts = summary.pivot(index="offer_type", columns="test_group", values="customers").reset_index()
    counts.columns.name = None
    counts = counts.rename(columns={"Control": "control_customers", "Treatment": "treatment_customers"})

    revenue = summary.pivot(index="offer_type", columns="test_group", values="revenue_aed").reset_index()
    revenue.columns.name = None
    revenue = revenue.rename(columns={"Control": "control_revenue_aed", "Treatment": "treatment_revenue_aed"})

    final = pivot.merge(counts, on="offer_type", how="left").merge(revenue, on="offer_type", how="left")
    for col in ["control_response_rate", "treatment_response_rate", "incremental_response_rate"]:
        final[col] = final[col].round(4)
    final["estimated_incremental_responses"] = (
        final["incremental_response_rate"] * final["treatment_customers"]
    ).round(0).astype(int)
    final["notes"] = "Synthetic test setup; replace with actual campaign/control outcomes after launch."
    return final.sort_values("incremental_response_rate", ascending=False)


def extract_feature_importance(best_model: Pipeline, numeric_features: List[str], categorical_features: List[str]) -> pd.DataFrame:
    preprocessor = best_model.named_steps["preprocess"]
    model = best_model.named_steps["model"]
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_names = list(cat_encoder.get_feature_names_out(categorical_features))
    feature_names = numeric_features + cat_names

    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_[0])
    else:
        importance = np.zeros(len(feature_names))

    df = pd.DataFrame({"feature": feature_names, "importance": importance})
    df["importance"] = df["importance"].astype(float)
    return df.sort_values("importance", ascending=False).head(25)


def save_charts(lift: pd.DataFrame, feature_importance: pd.DataFrame, segments: pd.DataFrame) -> None:
    # Chart 1: Lift by score decile.
    plt.figure(figsize=(8, 4.8))
    plt.bar(lift["score_decile"].astype(str), lift["lift_vs_baseline"])
    plt.xlabel("Score decile (1 = highest propensity)")
    plt.ylabel("Lift vs baseline")
    plt.title("Campaign Response Lift by Propensity Decile")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "lift_by_decile.png", dpi=160)
    plt.close()

    # Chart 2: Top feature importance.
    top_features = feature_importance.head(12).sort_values("importance")
    plt.figure(figsize=(8, 5.2))
    plt.barh(top_features["feature"], top_features["importance"])
    plt.xlabel("Importance")
    plt.title("Top Model Drivers")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "feature_importance.png", dpi=160)
    plt.close()

    # Chart 3: Historical response rate by segment.
    segment_rates = segments.sort_values("historical_response_rate")
    plt.figure(figsize=(8, 5.2))
    plt.barh(segment_rates["campaign_segment"], segment_rates["historical_response_rate"])
    plt.xlabel("Historical response rate")
    plt.title("Response Rate by Campaign Segment")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "segment_response_rate.png", dpi=160)
    plt.close()


def main() -> None:
    ensure_dirs()
    write_sql_feature_query()

    raw_csv = RAW_DIR / "synthetic_resort_loyalty_customers.csv"
    db_path = PROCESSED_DIR / "campaign_analytics.sqlite"
    raw = generate_synthetic_loyalty_data()
    raw.to_csv(raw_csv, index=False)

    load_raw_to_sqlite(raw_csv, db_path)
    features = build_features_from_sql(db_path)
    features.to_csv(PROCESSED_DIR / "customer_features.csv", index=False)

    qa_report = make_data_quality_report(raw, features)
    qa_report.to_csv(OUTPUT_DIR / "data_quality_report.csv", index=False)

    best_model, metrics, X_all, y_all, test_scored, numeric_features, categorical_features = train_models(features)
    metrics.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)

    lift = make_lift_table(test_scored)
    lift.to_csv(OUTPUT_DIR / "lift_by_decile.csv", index=False)

    scored = score_all_customers(best_model, features, numeric_features, categorical_features)
    scored_cols = [
        "customer_id",
        "home_market",
        "preferred_channel",
        "loyalty_tier",
        "recency_days",
        "visits_90d",
        "spend_180d_clean",
        "email_open_rate_clean",
        "prior_offer_count_180d",
        "predicted_response_probability",
        "score_decile",
        "campaign_segment",
        "recommended_channel",
        "qa_ready_flag",
    ]
    scored[scored_cols].sort_values(
        ["score_decile", "predicted_response_probability"], ascending=[True, False]
    ).to_csv(OUTPUT_DIR / "scored_campaign_list.csv", index=False)

    segments = make_segment_summary(scored)
    segments.to_csv(OUTPUT_DIR / "campaign_segments.csv", index=False)

    ab_test = make_ab_test_offer_calibration(scored)
    ab_test.to_csv(OUTPUT_DIR / "ab_test_offer_calibration.csv", index=False)

    feature_importance = extract_feature_importance(best_model, numeric_features, categorical_features)
    feature_importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    save_charts(lift, feature_importance, segments)

    print("Project built successfully.")
    print(f"Raw data: {raw_csv}")
    print(f"Features: {PROCESSED_DIR / 'customer_features.csv'}")
    print(f"Outputs: {OUTPUT_DIR}")
    print("Best model:", metrics.iloc[0]["model"], "ROC-AUC:", metrics.iloc[0]["roc_auc"])


if __name__ == "__main__":
    main()
