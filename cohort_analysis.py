import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

data = pd.read_csv('data.csv', encoding='ISO-8859-1')

data.dropna(subset=['CustomerID', 'InvoiceDate', 'UnitPrice', 'Quantity'], inplace=True)
data = data[data['Quantity'] > 0]  # Remove negative quantities (returns)
data = data[data['UnitPrice'] > 0]  # Remove non-positive unit prices

data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])


data['Revenue'] = data['Quantity'] * data['UnitPrice']  # Revenue per transaction
data['GrossProfit'] = data['Revenue'] * 0.3  # Assuming a fixed 30% profit margin
data['TransactionMonth'] = data['InvoiceDate'].dt.to_period('M')
data['Weekday'] = data['InvoiceDate'].dt.day_name()
data['Hour'] = data['InvoiceDate'].dt.hour
data['CohortMonth'] = data.groupby('CustomerID')['InvoiceDate'].transform('min').dt.to_period('M')
data['CohortIndex'] = (
    (data['TransactionMonth'].dt.year - data['CohortMonth'].dt.year) * 12 +
    (data['TransactionMonth'].dt.month - data['CohortMonth'].dt.month) + 1
)

cohort_data = data.groupby(['CohortMonth', 'CohortIndex']).agg({
    'CustomerID': 'nunique',
    'Revenue': 'sum',
    'GrossProfit': 'sum',
    'InvoiceNo': 'count'
}).reset_index()
cohort_data['AverageOrderValue'] = cohort_data['Revenue'] / cohort_data['InvoiceNo']

cohort_size = cohort_data.groupby('CohortMonth')['CustomerID'].first().reset_index()
cohort_data = pd.merge(cohort_data, cohort_size, on='CohortMonth', suffixes=('', '_CohortSize'))
cohort_data['ChurnRate'] = 100 - (cohort_data['CustomerID'] / cohort_data['CustomerID_CohortSize']) * 100

user_pivot = cohort_data.pivot(index='CohortMonth', columns='CohortIndex', values='CustomerID')
revenue_pivot = cohort_data.pivot(index='CohortMonth', columns='CohortIndex', values='Revenue')
gross_profit_pivot = cohort_data.pivot(index='CohortMonth', columns='CohortIndex', values='GrossProfit')
aov_pivot = cohort_data.pivot(index='CohortMonth', columns='CohortIndex', values='AverageOrderValue')
churn_pivot = cohort_data.pivot(index='CohortMonth', columns='CohortIndex', values='ChurnRate')

plt.figure(figsize=(14, 8))
sns.heatmap(user_pivot, annot=True, fmt=".0f", cmap="Blues")
plt.title('Cohort Analysis - User Count')
plt.show()

plt.figure(figsize=(14, 8))
sns.heatmap(revenue_pivot, annot=True, fmt=".1f", cmap="YlGnBu")
plt.title('Cohort Analysis - Monthly Revenue')
plt.show()

plt.figure(figsize=(14, 8))
sns.heatmap(aov_pivot, annot=True, fmt=".2f", cmap="Oranges")
plt.title('Cohort Analysis - Average Order Value (AOV)')
plt.show()

plt.figure(figsize=(14, 8))
sns.heatmap(churn_pivot, annot=True, fmt=".1f", cmap="Reds")
plt.title('Cohort Analysis - Churn Rates (%)')
plt.show()

purchase_patterns = data.groupby(['Weekday', 'Hour'])['Revenue'].sum().unstack().fillna(0)

plt.figure(figsize=(14, 8))
sns.heatmap(purchase_patterns, cmap='coolwarm', annot=True, fmt=".0f")
plt.title("Purchasing Trends by Weekday and Hour")
plt.show()

rfm_data = data.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (data['InvoiceDate'].max() - x.max()).days,
    'InvoiceNo': 'count',
    'Revenue': 'sum',
    'GrossProfit': 'sum'
}).rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'Revenue': 'Monetary'})

kmeans = KMeans(n_clusters=4, random_state=42)
rfm_data['Segment'] = kmeans.fit_predict(rfm_data[['Recency', 'Frequency', 'Monetary']])

plt.figure(figsize=(10, 6))
sns.scatterplot(data=rfm_data, x='Recency', y='Monetary', hue='Segment', palette='viridis')
plt.title('Customer Segmentation')
plt.xlabel('Recency (days since last purchase)')
plt.ylabel('Monetary (total spending)')
plt.legend(title='Segment')
plt.show()

