# Hospitality Loyalty Campaign Propensity & Segmentation Analytics

This is a GitHub-ready portfolio project built for database marketing, CRM analytics, and campaign data science roles.
It uses a synthetic resort loyalty dataset to demonstrate the work a marketing analytics data scientist would do before campaign launch and during performance review.

## What the project shows

- Data cleaning and validation checks for campaign datasets
- SQL feature engineering for recency, frequency, spend, engagement, and offer-fatigue variables
- Python modeling workflow using logistic regression and random forest
- Propensity scoring and lift-by-decile analysis
- Campaign list creation with targeting segments and recommended channels
- A/B test setup and offer calibration summary
- Dashboard-ready CSV outputs and simple charts
- Reproducible folder structure and documented assumptions

## Folder structure

```text
hospitality_loyalty_campaign_project/
  data/
    raw/                         synthetic input data
    processed/                   feature table and SQLite database
  outputs/                       model metrics, segments, scored list, QA reports
    charts/                      lift, feature importance, segment response charts
  sql/                           SQL feature engineering query
  src/                           project build script
  README.md
  requirements.txt
```

## How to run

From the project root:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/build_project.py
```

The script creates all data, features, model outputs, campaign files, and charts.

## Main outputs

- `outputs/model_metrics.csv`: model comparison with ROC-AUC, precision, recall, F1, and baseline response rate
- `outputs/lift_by_decile.csv`: response lift by score decile for campaign targeting
- `outputs/scored_campaign_list.csv`: customer-level campaign list with score, decile, segment, and channel
- `outputs/campaign_segments.csv`: segment sizes, average scores, response rates, and QA-ready rates
- `outputs/ab_test_offer_calibration.csv`: test/control setup and incremental response estimates
- `outputs/data_quality_report.csv`: duplicate, missing value, negative spend, and outlier checks
- `outputs/feature_importance.csv`: top model drivers
- `outputs/charts/`: PNG charts for reporting or Power BI storyboarding

## Resume talking points

Use this project to explain how you would support campaign targeting:

1. Build a clean customer feature table from raw loyalty/transaction data.
2. Validate missing values, duplicates, negative spend, and extreme values before modeling.
3. Train and compare models, then choose the model with stronger validation performance.
4. Convert model output into deciles, campaign segments, and dashboard-ready reporting tables.
5. Set up a control/treatment framework so campaign performance can be measured properly after launch.

## Important note

The dataset is synthetic, so it does not contain real customer data. In a real business environment, the same workflow would be connected to approved CRM, loyalty, transaction, and campaign-response tables.
