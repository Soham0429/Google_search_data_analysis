# Google Search Data Analysis

## Project Overview
This project performs Exploratory Data Analysis (EDA) on Google Search trend data to identify patterns, seasonality, and changes in public interest over time. The analysis focuses on understanding how search volume varies across dates and extracting meaningful insights from trends.

---

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## Dataset
The dataset is expected to contain Google search trend data.

**Typical Columns:**
- `date` or `Date` – date of the search data
- `search_trend` / `interest` – search popularity metric

(Adjust column names in the notebook if your dataset differs.)

---

## Installation & Setup

```bash
pip install pandas numpy matplotlib seaborn jupyter
```

---

## Usage
1. Place the dataset CSV file in the project directory.
2. Open the notebook:
```bash
jupyter notebook "Google_Search_Data_Analysis.ipynb"
```
3. Run all cells sequentially.

---

## Analysis Steps

### Import Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

### Load Dataset
```python
df = pd.read_csv("google_search_data.csv")
df.head()
```

### Basic Data Exploration
```python
df.info()
df.describe()
df.isnull().sum()
```

### Date-Time Processing
```python
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
```

### Search Trend Over Time
```python
plt.figure(figsize=(12,5))
sns.lineplot(data=df, x='Date', y='Search_Trend')
plt.title("Google Search Trend Over Time")
plt.xlabel("Date")
plt.ylabel("Search Interest")
plt.show()
```

### Monthly Average Trend
```python
monthly_avg = df.groupby('Month')['Search_Trend'].mean()

plt.figure(figsize=(10,5))
sns.barplot(x=monthly_avg.index, y=monthly_avg.values)
plt.title("Average Monthly Search Trend")
plt.xlabel("Month")
plt.ylabel("Average Interest")
plt.show()
```

---

## Key Insights
- Clear temporal patterns can be observed in search activity.
- Certain months show peak interest, indicating possible seasonality.
- Trend visualization helps understand changing public behavior.

---

## Conclusion
Google Search trend analysis provides valuable insight into public interest and behavior. This analysis can be extended using forecasting models, keyword comparison, and region-wise trend analysis.

---

## Future Enhancements
- Forecasting using ARIMA or Prophet
- Comparison of multiple keywords
- Interactive dashboards using Plotly or Streamlit

---

## License
This project is open-source and available under the MIT License.
