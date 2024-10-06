# Credit Scoring Model - Bati Bank

## Overview
You are an Analytics Engineer at **Bati Bank**, a leading financial service provider with over 10 years of experience. Bati Bank is partnering with an eCommerce company to enable a **Buy-Now-Pay-Later (BNPL)** service. Your task is to build a **Credit Scoring Model** using the data provided by the eCommerce platform.

Credit scoring refers to assigning a quantitative measure to a potential borrower to estimate the likelihood of a future default. The goal is to develop a model that evaluates creditworthiness, assigns a risk score, and predicts an optimal loan amount and duration.

### Objectives:
- Define a **proxy variable** to categorize users as high risk (bad) or low risk (good).
- Select observable features that predict default with high correlation.
- Develop models for:
  - Assigning risk probability.
  - Assigning credit score from risk probability estimates.
  - Predicting the optimal amount and duration of the loan.

## Data and Features
The data is provided via the [Xente Challenge | Kaggle](https://www.kaggle.com). 

### Data Fields:
- **TransactionId**: Unique transaction identifier.
- **BatchId**: Unique number assigned to a batch of transactions.
- **AccountId**: Unique identifier for the customer on the platform.
- **SubscriptionId**: Unique identifier for customer subscription.
- **CustomerId**: Unique identifier attached to the account.
- **CurrencyCode**: Country currency.
- **CountryCode**: Numerical geographical code of country.
- **ProviderId**: Source provider of item bought.
- **ProductId**: Item name being bought.
- **ProductCategory**: Broader product categories for ProductId.
- **ChannelId**: Identifies if the customer used web, Android, iOS, etc.
- **Amount**: Value of the transaction (positive for debits, negative for credits).
- **Value**: Absolute value of the transaction amount.
- **TransactionStartTime**: Time when the transaction began.
- **PricingStrategy**: Xente’s pricing structure for merchants.
- **FraudResult**: Fraud status of the transaction (1 for Yes, 0 for No).

---

## Tasks

### Task 1 - Understanding Credit Risk
Focus on the **concept of Credit Risk**. Key references:
- [Statistica Paper](https://www3.stat.sinica.edu.tw/statistica/oldpdf/A28n535.pdf)
- [Alternative Credit Scoring](https://www.hkma.gov.hk/media/eng/doc/key-functions/financial-infrastructure/alternative_credit_scoring.pdf)
- [World Bank Credit Scoring Guidelines](https://thedocs.worldbank.org/en/doc/935891585869698451-0130022020/original/CREDITSCORINGAPPROACHESGUIDELINESFINALWEB.pdf)
- [Towards Data Science Article](https://towardsdatascience.com/how-to-develop-a-credit-risk-model-and-scorecard-91335fc01f03)

### Task 2 - Exploratory Data Analysis (EDA)
- **Overview of the Data**: Structure, number of rows, columns, and data types.
- **Summary Statistics**: Central tendency, dispersion, and shape of the dataset.
- **Numerical Features Distribution**: Identify patterns, skewness, and outliers.
- **Categorical Features Distribution**: Analyze frequency and variability.
- **Correlation Analysis**: Identify relationships between numerical features.
- **Missing Values**: Identify and impute missing values.
- **Outlier Detection**: Use box plots to detect outliers.

### Task 3 - Feature Engineering
- **Aggregate Features**:
  - Total Transaction Amount.
  - Average Transaction Amount.
  - Transaction Count.
  - Standard Deviation of Transaction Amounts.
- **Extract Features**:
  - Transaction Hour, Day, Month, Year.
- **Encode Categorical Variables**:
  - One-Hot Encoding or Label Encoding.
- **Handle Missing Values**: Use imputation or remove rows/columns with missing values.
- **Normalize/Standardize Numerical Features**: Use normalization or standardization.

Key references:
- [Xverse](https://pypi.org/project/xverse/)
- [Weight of Evidence (WoE)](https://pypi.org/project/woe/)

### Task 4 - Default Estimator and WoE Binning
- **Construct a Proxy**: Classify users based on RFMS (Recency, Frequency, Monetary, and Sensitivity).
- **Assign Labels**: Classify users as good (high RFMS score) or bad (low RFMS score).
- **WoE Binning**: Use WoE binning to transform features for logistic regression models.

### Task 5 - Model Selection and Training
- **Split the Data**: Into training and testing sets.
- **Choose Models**: Select at least two models:
  - Logistic Regression
  - Decision Trees
  - Random Forest
  - Gradient Boosting Machines (GBM)
- **Train the Models**: Use training data.
- **Hyperparameter Tuning**: Apply grid search or random search.
- **Model Evaluation**: Evaluate using:
  - Accuracy
  - Precision
  - Recall (Sensitivity)
  - F1 Score
  - ROC-AUC

### Task 6 - Model Serving API
- **Framework Selection**: Choose a framework (Flask, FastAPI, or Django).
- **Load Model**: Load the trained ML model.
- **Define API Endpoints**: Create endpoints to accept data and return predictions.
- **Deployment**: Deploy the API to a web server or cloud platform.

---

## Competency Mapping
This project will contribute to the following competencies:

- **Professionalism for a global-level job**
- **Collaboration and Communication**
- **Software Development Frameworks**
- **Python Programming**
- **SQL Programming**
- **Data & Analytics Engineering**
- **MLOps & AutoML**
- **Deep Learning and Machine Learning**
