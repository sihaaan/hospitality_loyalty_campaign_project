# Hospitality Loyalty Campaign Propensity & Segmentation Analytics

This project analyzes synthetic resort loyalty data to support targeted hospitality campaign planning.
The workflow turns raw loyalty and transaction-style customer records into a clean feature table, propensity scores, audience segments, A/B test cells, and dashboard-ready reporting outputs.

## Project objective

Identify which loyalty customers are most likely to respond to a hospitality campaign, prioritize audiences by predicted response and customer value, and prepare measurement-ready campaign files for launch and post-campaign analysis.

## Approach

- Data cleaning and validation checks for campaign datasets
- SQL feature engineering for recency, frequency, spend, engagement, and offer-fatigue variables
- Python modeling workflow using logistic regression and random forest
- Propensity scoring and lift-by-decile analysis
- Campaign list creation with targeting segments and recommended channels
- A/B test setup and offer calibration summary
- Dashboard-ready CSV outputs and simple charts
- Reproducible folder structure and documented assumptions

## Workflow

1. Build a clean customer feature table from raw loyalty and transaction-style data.
2. Validate missing values, duplicates, negative spend, and extreme values before modeling.
3. Train and compare propensity models, then select the stronger validation performer.
4. Convert model scores into deciles, campaign segments, channel recommendations, and reporting tables.
5. Set up treatment/control measurement outputs so campaign performance can be reviewed after launch.

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

## Important note

The dataset is synthetic, so it does not contain real customer data. In a real business environment, the same workflow would be connected to approved CRM, loyalty, transaction, and campaign-response tables.
