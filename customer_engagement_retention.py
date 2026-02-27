"""
Customer Engagement & Retention Analytics
Miriam Garcia | miriamgarcia.org
Dataset: Olist Brazilian E-Commerce (Kaggle)
"""

# ============================================================
# 0. IMPORTS & DATA LOADING
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Style
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f9f9f9'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['font.family'] = 'DejaVu Sans'
PALETTE = ['#2C5F2D', '#97BC62', '#F9E795', '#B85042', '#E7E8D1']
sns.set_palette(PALETTE)

# Load data
orders = pd.read_csv('olist_orders_dataset.csv', parse_dates=[
    'order_purchase_timestamp', 'order_approved_at',
    'order_delivered_customer_date', 'order_estimated_delivery_date'])
customers = pd.read_csv('olist_customers_dataset.csv')
items = pd.read_csv('olist_order_items_dataset.csv')
reviews = pd.read_csv('olist_order_reviews_dataset.csv')
payments = pd.read_csv('olist_order_payments_dataset.csv')
products = pd.read_csv('olist_products_dataset.csv')
translation = pd.read_csv('product_category_name_translation.csv')

print("Data loaded successfully.")
print(f"Orders: {orders.shape[0]:,} | Customers: {customers['customer_unique_id'].nunique():,} | Items: {items.shape[0]:,}")

# ============================================================
# 1. DATA PREPARATION
# ============================================================

# Filter to delivered orders only
orders_clean = orders[orders['order_status'] == 'delivered'].copy()
orders_clean['year_month'] = orders_clean['order_purchase_timestamp'].dt.to_period('M')
orders_clean['year'] = orders_clean['order_purchase_timestamp'].dt.year

# Merge customers to get unique customer ID
orders_clean = orders_clean.merge(
    customers[['customer_id', 'customer_unique_id', 'customer_state']],
    on='customer_id', how='left')

# Order value per order
order_value = items.groupby('order_id').agg(
    revenue=('price', 'sum'),
    items_count=('order_item_id', 'count')
).reset_index()

# Merge order value
orders_clean = orders_clean.merge(order_value, on='order_id', how='left')

# Merge reviews
reviews_clean = reviews[['order_id', 'review_score']].drop_duplicates('order_id')
orders_clean = orders_clean.merge(reviews_clean, on='order_id', how='left')

# Translate product categories
products_en = products.merge(translation, on='product_category_name', how='left')
items_full = items.merge(products_en[['product_id', 'product_category_name_english']], on='product_id', how='left')

print(f"\nClean dataset: {orders_clean.shape[0]:,} delivered orders")
print(f"Date range: {orders_clean['order_purchase_timestamp'].min().date()} to {orders_clean['order_purchase_timestamp'].max().date()}")

# ============================================================
# 2. KPI FRAMEWORK
# ============================================================
print("\n" + "="*60)
print("KPI FRAMEWORK")
print("="*60)

total_customers   = orders_clean['customer_unique_id'].nunique()
total_orders      = orders_clean['order_id'].nunique()
total_revenue     = orders_clean['revenue'].sum()
avg_order_value   = orders_clean['revenue'].mean()
avg_items_order   = orders_clean['items_count'].mean()
repeat_customers  = orders_clean.groupby('customer_unique_id')['order_id'].count()
repeat_rate       = (repeat_customers > 1).sum() / total_customers * 100
avg_review        = orders_clean['review_score'].mean()

kpis = {
    'Total Unique Customers':  f"{total_customers:,}",
    'Total Orders':            f"{total_orders:,}",
    'Total Revenue (BRL)':     f"R${total_revenue:,.0f}",
    'Avg Order Value (BRL)':   f"R${avg_order_value:.2f}",
    'Avg Items per Order':     f"{avg_items_order:.2f}",
    'Repeat Customer Rate':    f"{repeat_rate:.1f}%",
    'Avg Review Score':        f"{avg_review:.2f} / 5.0",
}

for k, v in kpis.items():
    print(f"  {k:<30} {v}")

# ============================================================
# 3. COHORT ANALYSIS
# ============================================================
print("\n" + "="*60)
print("COHORT ANALYSIS")
print("="*60)

# Assign cohort = first purchase month per customer
cohort_data = orders_clean.groupby('customer_unique_id')['order_purchase_timestamp'].min().reset_index()
cohort_data.columns = ['customer_unique_id', 'first_purchase']
cohort_data['cohort'] = cohort_data['first_purchase'].dt.to_period('M')

orders_cohort = orders_clean.merge(cohort_data[['customer_unique_id', 'cohort']], on='customer_unique_id')
orders_cohort['order_period'] = orders_cohort['order_purchase_timestamp'].dt.to_period('M')
orders_cohort['period_number'] = (orders_cohort['order_period'] - orders_cohort['cohort']).apply(lambda x: x.n)

# Build cohort table
cohort_table = orders_cohort.groupby(['cohort', 'period_number'])['customer_unique_id'].nunique().reset_index()
cohort_pivot = cohort_table.pivot(index='cohort', columns='period_number', values='customer_unique_id')
cohort_sizes = cohort_pivot[0]
cohort_pct = cohort_pivot.divide(cohort_sizes, axis=0) * 100

# Focus on 2017 cohorts (full year, most data)
cohort_pct_2017 = cohort_pct[cohort_pct.index.astype(str).str.startswith('2017')]

print(f"\nCohort retention (period 1+) for 2017 cohorts:")
print(cohort_pct_2017[[0,1,2,3,6,12]].round(1).to_string())

# ============================================================
# 4. CHURN ANALYSIS
# ============================================================
print("\n" + "="*60)
print("CHURN ANALYSIS")
print("="*60)

# Define churn: no purchase in 6+ months after last purchase
last_purchase = orders_clean.groupby('customer_unique_id')['order_purchase_timestamp'].max().reset_index()
last_purchase.columns = ['customer_unique_id', 'last_purchase']
dataset_end = orders_clean['order_purchase_timestamp'].max()
last_purchase['days_since_last'] = (dataset_end - last_purchase['last_purchase']).dt.days
last_purchase['churned'] = last_purchase['days_since_last'] > 180

churn_rate = last_purchase['churned'].mean() * 100
active_rate = 100 - churn_rate

print(f"\nChurn definition: no purchase in 180+ days")
print(f"Churned customers:  {last_purchase['churned'].sum():,} ({churn_rate:.1f}%)")
print(f"Active customers:   {(~last_purchase['churned']).sum():,} ({active_rate:.1f}%)")

# Churn by review score
churn_review = orders_clean.merge(last_purchase[['customer_unique_id', 'churned']], on='customer_unique_id')
churn_by_score = churn_review.groupby('review_score')['churned'].mean() * 100
print(f"\nChurn rate by review score:")
for score, rate in churn_by_score.items():
    print(f"  Score {score}: {rate:.1f}%")

# ============================================================
# 5. FUNNEL ANALYSIS
# ============================================================
print("\n" + "="*60)
print("FUNNEL ANALYSIS")
print("="*60)

funnel_stages = {
    'All Orders':         orders['order_id'].nunique(),
    'Delivered':          orders_clean['order_id'].nunique(),
    'With Review':        orders_clean[orders_clean['review_score'].notna()]['order_id'].nunique(),
    'Score 4 or 5':       orders_clean[orders_clean['review_score'] >= 4]['order_id'].nunique(),
    'Repeat Customers':   int((repeat_customers > 1).sum()),
}

prev = None
for stage, count in funnel_stages.items():
    if prev:
        drop = (1 - count/prev) * 100
        print(f"  {stage:<25} {count:>8,}   ({drop:.1f}% drop from previous)")
    else:
        print(f"  {stage:<25} {count:>8,}")
    prev = count

# ============================================================
# 6. ENGAGEMENT BY STATE (LATAM LENS)
# ============================================================
print("\n" + "="*60)
print("ENGAGEMENT BY STATE")
print("="*60)

state_metrics = orders_clean.groupby('customer_state').agg(
    customers=('customer_unique_id', 'nunique'),
    orders=('order_id', 'nunique'),
    avg_revenue=('revenue', 'mean'),
    avg_review=('review_score', 'mean')
).reset_index()
state_metrics['orders_per_customer'] = state_metrics['orders'] / state_metrics['customers']
state_metrics = state_metrics.sort_values('customers', ascending=False)

print(state_metrics.head(10).to_string(index=False))

# ============================================================
# 7. REVENUE TRENDS
# ============================================================
print("\n" + "="*60)
print("MONTHLY REVENUE TREND")
print("="*60)

monthly = orders_clean.groupby('year_month').agg(
    revenue=('revenue', 'sum'),
    orders=('order_id', 'nunique'),
    customers=('customer_unique_id', 'nunique')
).reset_index()
monthly['year_month_str'] = monthly['year_month'].astype(str)
monthly['mom_growth'] = monthly['revenue'].pct_change() * 100

print(monthly[['year_month_str', 'revenue', 'orders', 'customers', 'mom_growth']].tail(12).to_string(index=False))

print("\nAnalysis complete. Ready for visualization.")


print("Oh so you can read my code? Send me a 🐍 on Linkedin!")
