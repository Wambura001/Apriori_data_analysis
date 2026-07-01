# Apriori_data_analysis
# Market Basket Optimization using the Apriori Algorithm

## Executive Summary

In retail and e-commerce, understanding customer purchasing behavior is critical to maximizing revenue and improving customer experience. This project implements the **Apriori Algorithm** for **Market Basket Analysis (MBA)** to discover hidden associations between products purchased at a retail store. By uncovering these patterns, businesses can transform raw transactional data into actionable marketing, merchandising, and inventory strategies.

---

## Business Problem

Modern retail environments generate massive amounts of transactional data, yet most of this data remains underutilized. Retailers face several operational and marketing inefficiencies:

* **Suboptimal Product Placement:** Placing complementary products far apart decreases the likelihood of impulse buys.
* **Ineffective Promotional Bundling:** Creating product bundles based on "gut feeling" rather than data often leads to low campaign conversion rates.
* **Missed Cross-Selling Opportunities:** Cashiers and digital recommendation engines fail to suggest the right next item, leaving potential revenue on the table.
* **Inventory Misalignment:** Failing to anticipate demand spikes for paired items can lead to stockouts of complementary products.

### Objective

The primary business objective is to identify directional **"If-Then" rules** (e.g., *If a customer buys Item A, they are highly likely to buy Item B*) to drive real-world marketing actions, increase average basket size, and boost overall retail revenue.

---

## Technical Solution

### Why Apriori?

The Apriori algorithm is uniquely suited for Market Basket Analysis because it uses a bottom-up approach to find **frequent itemsets** based on mathematical thresholds. Instead of checking every possible combination of items (which is computationally impossible for large inventories), Apriori leverages the **Apriori Property**: *all non-empty subsets of a frequent itemset must also be frequent.*

### Core Metrics Explained

The algorithm filters and ranks association rules ($A \implies B$) using three critical metrics:

1. **Support:** The baseline popularity of an itemset. It measures the proportion of total transactions that contain both items.

$$\text{Support}(A \implies B) = \frac{\text{Transactions containing both } A \text{ and } B}{\text{Total Transactions}}$$


2. **Confidence:** The reliability of the rule. It measures how often item $B$ appears in transactions that *already* contain item $A$.

$$\text{Confidence}(A \implies B) = \frac{\text{Support}(A \implies B)}{\text{Support}(A)}$$


3. **Lift:** The strength of the rule. A lift value greater than 1 indicates that the purchase of item $A$ significantly increases the likelihood of purchasing item $B$ compared to random chance.

$$\text{Lift}(A \implies B) = \frac{\text{Confidence}(A \implies B)}{\text{Support}(B)}$$



---

## Implementation Workflow

### 1. Environment & Library Setup

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Install and import apyori for Association Rule Mining
!pip install apyori
from apyori import apriori

```

### 2. Data Preprocessing

The dataset contains 7,501 historical transactions. Because market basket datasets vary in transaction length (number of items per basket), rows are parsed into a structured list of string lists to make them compatible with the Apriori model.

```python
# Load dataset without headers
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)

# Structure transactions into a list of lists (strings)
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])

```

### 3. Model Training

We define strict business thresholds to filter out noise and extract highly relevant, actionable rules:

* `min_support = 0.003`: Items must appear at least ~3 times a day (assuming a weekly cycle: $3 \times 7 / 7501 \approx 0.0028$).
* `min_confidence = 0.2`: The rule must be correct at least 20% of the time.
* `min_lift = 3`: The items must be at least 3 times more likely to be bought together than by chance.

```python
rules = apriori(
    transactions = transactions, 
    min_support = 0.003, 
    min_confidence = 0.2, 
    min_lift = 3, 
    min_length = 2, 
    max_length = 2
)
results = list(rules)

```

---

## Results & Business Insights

After processing, the generated rules were parsed into a structured Pandas DataFrame and sorted by **Lift** in descending order:

| Left Hand Side (Antecedent) | Right Hand Side (Consequent) | Support | Confidence | Lift |
| --- | --- | --- | --- | --- |
| fromage blanc | honey | 0.003333 | 0.245098 | **5.164271** |
| light cream | chicken | 0.004533 | 0.290598 | **4.843951** |
| pasta | escalope | 0.005866 | 0.372881 | **4.700812** |
| pasta | shrimp | 0.005066 | 0.322034 | **4.506672** |
| whole wheat pasta | olive oil | 0.007999 | 0.271493 | **4.122410** |

### Strategic Business Actions

* **Cross-Merchandising Layouts:** Place high-lift items near each other. For example, position *honey* directly adjacent to *fromage blanc*, or display *olive oil* prominently in the *pasta* aisle to capture high-probability impulse buys.
* **Targeted Product Bundling:** Create a meal-kit bundle or discount package combining *light cream* and *chicken*. Since customers buying light cream are nearly **5 times more likely** to buy chicken, a minor promotional bundle can dramatically drive volume.
* **Smart Recommendation Engines:** For e-commerce interfaces, configure the recommendation engine so that when a user adds *pasta* to their digital cart, the system automatically triggers a cross-sell banner for *escalope* or *shrimp* (boasting **37%** and **32%** confidence respectively).
* **Dynamic Inventory Optimization:** Sync inventory management for paired items. Ensuring that a stock promotion on *pasta* is backed by increased inventory levels for *shrimp* and *escalope* prevents unexpected stockouts and customer dissatisfaction.
