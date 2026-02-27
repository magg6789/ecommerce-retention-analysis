# E-Commerce Customer Engagement & Retention Analysis

**Tools:** Python, SQL, Tableau | **Dataset:** Olist Brazilian E-Commerce (Kaggle) | **Records:** 100K+ orders, 96K customers, 2016–2018

---

## Overview

This project analyzes customer engagement and retention patterns for a Brazilian e-commerce marketplace. The core question: with a 4.16/5 average review score, why are 58.9% of customers churning within 6 months?

The analysis works through cohort retention, churn segmentation, funnel drop-off, and revenue trends to identify where customers are being lost and what the data suggests about where to focus retention efforts.

---

## Key Findings

| Metric | Value |
|--------|-------|
| Unique customers | 93,358 |
| Delivered orders | 96,478 |
| Total revenue | R$13.2M |
| Avg order value | R$137 |
| Repeat purchase rate | 3.0% |
| Churn rate (180-day) | 58.9% |
| Avg review score | 4.16 / 5.0 |

**The short version:** Acquisition is working. Customers are satisfied. Almost none of them come back.

Cohort retention falls to ~0.5% by month 1 across all 2017 cohorts, and churn rates are nearly identical for customers who gave 5-star reviews (57.5%) versus 1-star reviews (60.2%). The problem is not product quality or delivery experience. There is no mechanism pulling customers back after their first purchase.

---

## Visualizations

### KPI Dashboard
![KPI Dashboard](images/fig1_kpi_dashboard.png)

### Cohort Retention Heatmap
![Cohort Retention](images/fig2_cohort_heatmap.png)

### Churn Analysis
![Churn Analysis](images/fig3_churn_analysis.png)

### Monthly Revenue Trend
![Revenue Trend](images/fig4_revenue_trend.png)

### Engagement Funnel & Top Categories
![Funnel and Categories](images/fig5_funnel_categories.png)

### Satisfaction vs. Churn
![Satisfaction and Churn](images/fig6_satisfaction_churn.png)

---

## Tableau Dashboard

Interactive dashboard with retention curves, cohort heatmap, and engagement funnel:
[View on Tableau Public](https://public.tableau.com/app/profile/miriamgarcia/vizzes)

---

## Repo Structure

```
ecommerce-retention-analysis/
│
├── Customer_Engagement_Retention_Analytics.ipynb   # Main analysis notebook
├── customer_engagement_retention.py                # Analysis script
├── visualizations.py                               # Chart generation
├── images/                                         # Output charts
│   ├── fig1_kpi_dashboard.png
│   ├── fig2_cohort_heatmap.png
│   ├── fig3_churn_analysis.png
│   ├── fig4_revenue_trend.png
│   ├── fig5_funnel_categories.png
│   └── fig6_satisfaction_churn.png
└── README.md
```

> **Note:** Raw data files are not included due to size. Download from [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) and place CSVs in the root directory before running.

---

## Setup

```bash
git clone https://github.com/magg6789/ecommerce-retention-analysis.git
cd ecommerce-retention-analysis
pip install pandas numpy matplotlib seaborn jupyter
jupyter notebook Customer_Engagement_Retention_Analytics.ipynb
```

---

## Analysis Sections

**1. KPI Framework** — baseline metrics: revenue, AOV, repeat rate, churn, satisfaction

**2. Cohort Retention Analysis** — monthly cohorts tracked across 12 periods; 2017 cohorts show consistent sub-1% retention by month 1

**3. Churn Analysis** — 180-day inactivity threshold; breakdown by review score shows satisfaction and retention are nearly uncorrelated

**4. Engagement Funnel** — traces orders from placement through delivery, review, satisfaction, and repeat purchase; 96% drop-off from satisfied customers to repeat buyers

**5. Revenue Trends** — monthly revenue and MoM growth Jan 2017 to Aug 2018; November 2017 Black Friday spike at +52% MoM

**6. Findings & Recommendations** — prioritized next steps including post-purchase re-engagement sequencing, loyalty mechanics, and category-level cohort cuts

---

## Recommendations

**Post-purchase re-engagement is the highest-leverage lever.** 83% of customers rate their experience 4 or 5 stars, but only 3% come back. A re-engagement sequence at 30, 60, and 90 days post-purchase targeting that satisfied-but-inactive segment is the most direct path to improving retention.

**Satisfaction does not equal loyalty without a pull mechanism.** The nearly identical churn rates across all review scores suggest that customers are not leaving because of a bad experience. They simply have no reason to return. Loyalty programs, personalized product recommendations, or subscription options would directly address this.

**Reduce seasonal revenue concentration.** November 2017 was a clear outlier driven by Black Friday. Off-peak campaigns in Q1 and Q3 would reduce dependency on a single promotional window and smooth out the annual revenue curve.

---

## Next Steps

- Churn prediction model using customer features: tenure, AOV, review score, category mix
- Category-level cohort analysis to identify which product segments have the highest natural repurchase potential
- A/B test framework for re-engagement email timing

---

*Dataset: [Olist Brazilian E-Commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) via Kaggle*
