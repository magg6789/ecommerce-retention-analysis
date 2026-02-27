import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── Style ──────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': '#f9f9f9',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.family': 'DejaVu Sans',
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
})
GREEN  = '#2C5F2D'
LIME   = '#97BC62'
GOLD   = '#F5C842'
RED    = '#B85042'
SAND   = '#E7E8D1'
GRAY   = '#6B6B6B'

# ── Load & prep ────────────────────────────────────────────
orders = pd.read_csv('olist_orders_dataset.csv', parse_dates=[
    'order_purchase_timestamp','order_approved_at',
    'order_delivered_customer_date','order_estimated_delivery_date'])
customers  = pd.read_csv('olist_customers_dataset.csv')
items      = pd.read_csv('olist_order_items_dataset.csv')
reviews    = pd.read_csv('olist_order_reviews_dataset.csv')
payments   = pd.read_csv('olist_order_payments_dataset.csv')
products   = pd.read_csv('olist_products_dataset.csv')
translation= pd.read_csv('product_category_name_translation.csv')

orders_clean = orders[orders['order_status'] == 'delivered'].copy()
orders_clean['year_month'] = orders_clean['order_purchase_timestamp'].dt.to_period('M')

orders_clean = orders_clean.merge(
    customers[['customer_id','customer_unique_id','customer_state']],
    on='customer_id', how='left')

order_value = items.groupby('order_id').agg(
    revenue=('price','sum'), items_count=('order_item_id','count')).reset_index()
orders_clean = orders_clean.merge(order_value, on='order_id', how='left')

reviews_clean = reviews[['order_id','review_score']].drop_duplicates('order_id')
orders_clean = orders_clean.merge(reviews_clean, on='order_id', how='left')

products_en = products.merge(translation, on='product_category_name', how='left')
items_full  = items.merge(products_en[['product_id','product_category_name_english']], on='product_id', how='left')

# Cohort prep
cohort_data = orders_clean.groupby('customer_unique_id')['order_purchase_timestamp'].min().reset_index()
cohort_data.columns = ['customer_unique_id','first_purchase']
cohort_data['cohort'] = cohort_data['first_purchase'].dt.to_period('M')
orders_cohort = orders_clean.merge(cohort_data[['customer_unique_id','cohort']], on='customer_unique_id')
orders_cohort['order_period']  = orders_cohort['order_purchase_timestamp'].dt.to_period('M')
orders_cohort['period_number'] = (orders_cohort['order_period'] - orders_cohort['cohort']).apply(lambda x: x.n)

cohort_table  = orders_cohort.groupby(['cohort','period_number'])['customer_unique_id'].nunique().reset_index()
cohort_pivot  = cohort_table.pivot(index='cohort', columns='period_number', values='customer_unique_id')
cohort_sizes  = cohort_pivot[0]
cohort_pct    = cohort_pivot.divide(cohort_sizes, axis=0) * 100
cohort_pct_plot = cohort_pct[cohort_pct.index.year == 2017].iloc[:, :13]

# Churn prep
last_purchase = orders_clean.groupby('customer_unique_id')['order_purchase_timestamp'].max().reset_index()
last_purchase.columns = ['customer_unique_id','last_purchase']
dataset_end = orders_clean['order_purchase_timestamp'].max()
last_purchase['days_since_last'] = (dataset_end - last_purchase['last_purchase']).dt.days
last_purchase['churned'] = last_purchase['days_since_last'] > 180

# Monthly
monthly = orders_clean.groupby('year_month').agg(
    revenue=('revenue','sum'),
    orders=('order_id','nunique'),
    customers=('customer_unique_id','nunique')).reset_index()
monthly = monthly[monthly['year_month'].astype(str) >= '2017-01'].copy()
monthly['year_month_str'] = monthly['year_month'].astype(str)
monthly['mom_growth'] = monthly['revenue'].pct_change() * 100

# Category
top_cats = items_full.groupby('product_category_name_english')['order_id'].nunique().nlargest(10).reset_index()
top_cats.columns = ['category','orders']

# ══════════════════════════════════════════════════════════════
# FIGURE 1 – KPI Dashboard (summary cards)
# ══════════════════════════════════════════════════════════════
fig1, axes = plt.subplots(2, 4, figsize=(16, 6))
fig1.suptitle('Customer Engagement & Retention — KPI Dashboard\nOlist Brazilian E-Commerce · 2016–2018',
              fontsize=15, fontweight='bold', y=1.01)

kpis = [
    ('93,358',        'Unique Customers',      GREEN),
    ('96,478',        'Delivered Orders',       GREEN),
    ('R$13.2M',       'Total Revenue',          LIME),
    ('R$137',         'Avg Order Value',         LIME),
    ('3.0%',          'Repeat Purchase Rate',    GOLD),
    ('58.9%',         'Churn Rate (180d)',        RED),
    ('4.16 / 5.0',    'Avg Review Score',        GREEN),
    ('97.0%',         'Delivery Success Rate',   LIME),
]

for ax, (val, label, color) in zip(axes.flat, kpis):
    ax.set_facecolor(color + '22')
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.axis('off')
    ax.text(0.5, 0.62, val, ha='center', va='center',
            fontsize=22, fontweight='bold', color=color)
    ax.text(0.5, 0.28, label, ha='center', va='center',
            fontsize=10, color=GRAY, wrap=True)
    for spine in ax.spines.values():
        spine.set_visible(False)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/fig1_kpi_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 1 saved")

# ══════════════════════════════════════════════════════════════
# FIGURE 2 – Cohort Retention Heatmap
# ══════════════════════════════════════════════════════════════
fig2, ax = plt.subplots(figsize=(14, 7))
cohort_plot = cohort_pct_plot.copy()
cohort_plot.index = cohort_plot.index.astype(str)

mask = cohort_plot.isna()
sns.heatmap(cohort_plot, annot=True, fmt='.1f', mask=mask,
            cmap='YlGn', linewidths=0.5, linecolor='white',
            ax=ax, cbar_kws={'label': 'Retention %'},
            annot_kws={'size': 8})

ax.set_title('Cohort Retention Analysis — 2017 Customer Cohorts\nRetention % by Months Since First Purchase',
             fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Months Since First Purchase', fontsize=11)
ax.set_ylabel('Acquisition Cohort', fontsize=11)
ax.tick_params(axis='x', rotation=0)
ax.tick_params(axis='y', rotation=0)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/fig2_cohort_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 2 saved")

# ══════════════════════════════════════════════════════════════
# FIGURE 3 – Churn Analysis
# ══════════════════════════════════════════════════════════════
fig3, axes = plt.subplots(1, 2, figsize=(14, 5))

# 3a: Churn donut
churned   = last_purchase['churned'].sum()
active    = (~last_purchase['churned']).sum()
sizes     = [active, churned]
labels    = [f'Active\n{active:,}\n(41.1%)', f'Churned\n{churned:,}\n(58.9%)']
colors    = [GREEN, RED]
wedges, texts = axes[0].pie(sizes, labels=labels, colors=colors,
                              startangle=90, wedgeprops=dict(width=0.5),
                              textprops={'fontsize':11})
axes[0].set_title('Customer Churn Status\n(180-day inactivity threshold)', fontweight='bold')

# 3b: Days since last purchase distribution
axes[1].hist(last_purchase['days_since_last'], bins=40,
             color=GREEN, alpha=0.8, edgecolor='white')
axes[1].axvline(180, color=RED, linestyle='--', linewidth=2, label='Churn threshold (180 days)')
axes[1].set_xlabel('Days Since Last Purchase')
axes[1].set_ylabel('Number of Customers')
axes[1].set_title('Distribution of Days Since Last Purchase', fontweight='bold')
axes[1].legend(fontsize=9)

fig3.suptitle('Churn Analysis — Olist E-Commerce', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/fig3_churn_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 3 saved")

# ══════════════════════════════════════════════════════════════
# FIGURE 4 – Revenue Trend + MoM Growth
# ══════════════════════════════════════════════════════════════
fig4, ax1 = plt.subplots(figsize=(14, 5))

x = range(len(monthly))
bars = ax1.bar(x, monthly['revenue']/1000, color=[
    LIME if g >= 0 else RED for g in monthly['mom_growth'].fillna(0)],
    alpha=0.85, width=0.6)
ax1.set_ylabel('Revenue (BRL thousands)', fontsize=11)
ax1.set_xticks(x)
ax1.set_xticklabels(monthly['year_month_str'], rotation=45, ha='right', fontsize=8)
ax1.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f'R${v:.0f}K'))

ax2 = ax1.twinx()
ax2.plot(x, monthly['mom_growth'], color=GREEN, linewidth=2,
         marker='o', markersize=5, label='MoM Growth %')
ax2.axhline(0, color=GRAY, linestyle='--', linewidth=0.8)
ax2.set_ylabel('Month-over-Month Growth %', fontsize=11)
ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
ax2.spines['top'].set_visible(False)

# Annotate Nov 2017 spike
nov_idx = list(monthly['year_month_str']).index('2017-11')
ax1.annotate('Black Friday\nspike +52%',
             xy=(nov_idx, monthly.iloc[nov_idx]['revenue']/1000),
             xytext=(nov_idx+1.5, monthly.iloc[nov_idx]['revenue']/1000 + 50),
             arrowprops=dict(arrowstyle='->', color=RED),
             fontsize=8, color=RED)

ax1.set_title('Monthly Revenue Trend & MoM Growth\nJan 2017 – Aug 2018',
              fontsize=13, fontweight='bold')
green_patch = mpatches.Patch(color=LIME, label='Revenue (growth month)')
red_patch   = mpatches.Patch(color=RED,  label='Revenue (decline month)')
ax1.legend(handles=[green_patch, red_patch], loc='upper left', fontsize=9)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/fig4_revenue_trend.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 4 saved")

# ══════════════════════════════════════════════════════════════
# FIGURE 5 – Funnel + Top Categories
# ══════════════════════════════════════════════════════════════
fig5, axes = plt.subplots(1, 2, figsize=(15, 6))

# 5a: Funnel
funnel_labels = ['All Orders\n99,441', 'Delivered\n96,478', 'With Review\n95,832',
                 'Score 4-5\n75,657', 'Repeat\nCustomers\n2,801']
funnel_values = [99441, 96478, 95832, 75657, 2801]
funnel_colors = [GREEN, LIME, GOLD, '#E8A838', RED]

y_pos = range(len(funnel_values))
bars5 = axes[0].barh(list(y_pos), funnel_values, color=funnel_colors, alpha=0.85, height=0.6)
axes[0].set_yticks(list(y_pos))
axes[0].set_yticklabels(funnel_labels, fontsize=9)
axes[0].set_xlabel('Number of Orders / Customers')
axes[0].set_title('Engagement Funnel', fontweight='bold')
axes[0].xaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f'{v:,.0f}'))
axes[0].invert_yaxis()

for bar, val in zip(bars5, funnel_values):
    axes[0].text(bar.get_width() + 500, bar.get_y() + bar.get_height()/2,
                 f'{val:,}', va='center', fontsize=8)

# 5b: Top categories
top_cats_sorted = top_cats.sort_values('orders')
axes[1].barh(top_cats_sorted['category'], top_cats_sorted['orders'],
             color=GREEN, alpha=0.8)
axes[1].set_xlabel('Number of Orders')
axes[1].set_title('Top 10 Product Categories by Orders', fontweight='bold')
axes[1].xaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f'{v:,.0f}'))

fig5.suptitle('Engagement Funnel & Category Analysis', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/fig5_funnel_categories.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 5 saved")

# ══════════════════════════════════════════════════════════════
# FIGURE 6 – Review Score & Satisfaction
# ══════════════════════════════════════════════════════════════
fig6, axes = plt.subplots(1, 2, figsize=(14, 5))

# 6a: Review distribution
review_counts = orders_clean['review_score'].value_counts().sort_index()
colors6 = [RED, '#E8763A', GOLD, LIME, GREEN]
axes[0].bar(review_counts.index, review_counts.values, color=colors6, alpha=0.85, edgecolor='white')
axes[0].set_xlabel('Review Score')
axes[0].set_ylabel('Number of Reviews')
axes[0].set_title('Review Score Distribution\n83% of customers score 4 or 5', fontweight='bold')
axes[0].yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f'{v:,.0f}'))
for i, (score, count) in enumerate(review_counts.items()):
    pct = count / review_counts.sum() * 100
    axes[0].text(score, count + 300, f'{pct:.1f}%', ha='center', fontsize=9)

# 6b: Churn rate by review score
churn_review = orders_clean.merge(last_purchase[['customer_unique_id','churned']], on='customer_unique_id')
churn_by_score = churn_review.groupby('review_score')['churned'].mean() * 100

axes[1].bar(churn_by_score.index, churn_by_score.values, color=colors6, alpha=0.85, edgecolor='white')
axes[1].set_xlabel('Review Score')
axes[1].set_ylabel('Churn Rate %')
axes[1].set_title('Churn Rate by Review Score\nScore 5 customers churn 2.7% less than Score 1', fontweight='bold')
axes[1].yaxis.set_major_formatter(mtick.PercentFormatter())
axes[1].set_ylim(50, 70)
for score, rate in churn_by_score.items():
    axes[1].text(score, rate + 0.3, f'{rate:.1f}%', ha='center', fontsize=9)

fig6.suptitle('Customer Satisfaction & Retention Relationship', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/fig6_satisfaction_churn.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 6 saved")

print("\nAll visualizations saved to /mnt/user-data/outputs/")


print("Oh so you REALLY like visuals? Me too! Send me a 'visualize my data' on Linkedin to connect more on visual building in Python and my fav BI tools Power BI and Tableau!)
