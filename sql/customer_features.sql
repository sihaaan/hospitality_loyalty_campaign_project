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