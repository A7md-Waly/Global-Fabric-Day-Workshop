# Customer Churn Prediction Documentation

## Introduction

This notebook, `Customer Churn Prediction.ipynb`, is part of a Fabric Data Science workshop focused on banking analytics. It builds a binary classification model to predict customer churn using Microsoft Fabric’s PySpark and AutoML capabilities. The model leverages a banking dataset stored in Delta tables within a Fabric Lakehouse, processes data with PySpark, and uses FLAML (Fast and Lightweight AutoML) to train and optimize machine learning models. The notebook tracks experiments with MLflow and saves predictions for visualization in Power BI.

### Objectives

- Predict which customers are likely to churn based on their banking behavior.
- Demonstrate end-to-end ML workflow in Fabric: data loading, preprocessing, modeling, and evaluation.
- Enable participants to visualize churn predictions and understand customer retention strategies.

### Dataset

The dataset includes six Delta tables:

- **Customers**: CustomerID, Address, JoinDate.
- **Accounts**: AccountID, CustomerID, AccountType, Balance, CreatedDate.
- **Transactions**: TransactionID, AccountID, TransactionType, Amount, TransactionDate.
- **Loans**: LoanID, CustomerID, LoanType, LoanAmount, InterestRate, LoanStartDate, LoanEndDate.
- **Cards**: CardID, CustomerID, CardType, ExpirationDate.
- **SupportCalls**: CallID, CustomerID, IssueType, Resolved, CallDate.

## Setup

### Prerequisites

- **Fabric Workspace**: Access to a Fabric environment with a Lakehouse containing the Delta tables.

- **Spark Cluster**: Fabric Runtime 1.2+ or Spark 3.4+ for AutoML compatibility.

### Environment Configuration

1. Create a notebook in Fabric and attach it to a Spark cluster.
2. Install dependencies using the above command or Fabric’s environment management.
3. Ensure Delta tables are accessible at `Banking_Data.<table_name>`.

### Notebook Initialization

The notebook starts by initializing a Spark session for data processing.

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, sum, avg, max, when, current_date, datediff, substring, length, expr

spark = SparkSession.builder.appName("CustomerChurn").getOrCreate()
```

## Step 1: Load the Data

The notebook loads data from Delta tables in the Fabric Lakehouse using Spark SQL.

```python
customers_df = spark.sql("SELECT * FROM Banking_Data.Customers")
accounts_df = spark.sql("SELECT * FROM Banking_Data.Accounts")
transactions_df = spark.sql("SELECT * FROM Banking_Data.Transactions")
loans_df = spark.sql("SELECT * FROM Banking_Data.Loans")
cards_df = spark.sql("SELECT * FROM Banking_Data.Cards")
support_calls_df = spark.sql("SELECT * FROM Banking_Data.SupportCalls")
```

## Step 2: Feature Engineering

The notebook aggregates and transforms data to create features for churn prediction. Features are derived from all tables and joined by `CustomerID`.

### Customer Features

- **state_code**: Extracted from the last two characters of `Address`.
- **tenure_days**: Days from `JoinDate` to the current date (May 29, 2025).

```python
customers_df = customers_df.withColumn(
    "state_code",
    substring(col("Address"), -8, 2)
).withColumn(
    "tenure_days",
    datediff(current_date(), col("JoinDate"))
)
customers_features = customers_df.select("CustomerID", "state_code", "tenure_days")
```

### Account Features

Aggregates account data by `CustomerID`:

- Number of accounts, total/average balance.
- Counts of account types (Savings, Checking, Business).
- Maximum account age and days since last balance update.

```python
account_agg = accounts_df.groupBy("CustomerID").agg(
    count("AccountID").alias("num_accounts"),
    sum("Balance").alias("total_balance"),
    count(when(col("AccountType") == "Savings", 1)).alias("num_savings_accounts"),
    count(when(col("AccountType") == "Checking", 1)).alias("num_checking_accounts"),
    count(when(col("AccountType") == "Business", 1)).alias("num_business_accounts"),
    avg("Balance").alias("avg_balance"),
    max(datediff(current_date(), col("CreatedDate"))).alias("max_account_age_days"),
    max("CreatedDate").alias("last_balance_update"),
    datediff(current_date(), max("CreatedDate")).alias("days_since_last_balance_update")
)
```

### Transaction Features

Aggregates transactions by `AccountID`, then by `CustomerID`:

- Total transactions, amount, and counts by type (Deposit, Withdrawal, Transfer, Payment).
- Days since last transaction.

```python
transaction_agg = transactions_df.groupBy("AccountID").agg(
    count("TransactionID").alias("num_transactions"),
    sum("Amount").alias("total_transaction_amount"),
    count(when(col("TransactionType") == "Deposit", 1)).alias("num_deposits"),
    count(when(col("TransactionType") == "Withdrawal", 1)).alias("num_withdrawals"),
    count(when(col("TransactionType") == "Transfer", 1)).alias("num_transfers"),
    count(when(col("TransactionType") == "Payment", 1)).alias("num_payments"),
    max("TransactionDate").alias("last_transaction_date"),
    datediff(current_date(), max("TransactionDate")).alias("days_since_last_transaction")
)

trans_acc = transaction_agg.join(
    accounts_df.select("AccountID", "CustomerID"),
    "AccountID",
    "inner"
).groupBy("CustomerID").agg(
    sum("num_transactions").alias("total_num_transactions"),
    sum("total_transaction_amount").alias("total_transaction_amount"),
    sum("num_deposits").alias("total_deposits"),
    sum("num_withdrawals").alias("total_withdrawals"),
    sum("num_transfers").alias("total_transfers"),
    sum("num_payments").alias("total_payments"),
    max("last_transaction_date").alias("last_transaction_date"),
    max("days_since_last_transaction").alias("days_since_last_transaction")
)
```

### Loan Features

Aggregates loan data by `CustomerID`:

- Number of loans, total amount, average interest rate.
- Counts by loan type (Car, Personal, Home, Education).
- Maximum loan duration and days to loan end.

```python
loan_agg = loans_df.groupBy("CustomerID").agg(
    count("LoanID").alias("num_loans"),
    sum("LoanAmount").alias("total_loan_amount"),
    avg("InterestRate").alias("avg_interest_rate"),
    count(when(col("LoanType") == "Car", 1)).alias("num_car_loans"),
    count(when(col("LoanType") == "Personal", 1)).alias("num_personal_loans"),
    count(when(col("LoanType") == "Home", 1)).alias("num_home_loans"),
    count(when(col("LoanType") == "Education", 1)).alias("num_education_loans"),
    max(datediff(col("LoanEndDate"), col("LoanStartDate"))).alias("max_loan_duration_days"),
    max(when(col("LoanEndDate") <= current_date(), 0).otherwise(datediff(col("LoanEndDate"), current_date()))).alias("days_to_loan_end")
)
```

### Card Features

Aggregates card data by `CustomerID`:

- Number of cards and counts by type (Credit, Debit, Prepaid).
- Maximum days to card expiry.

```python
card_agg = cards_df.groupBy("CustomerID").agg(
    count("CardID").alias("num_cards"),
    count(when(col("CardType") == "Credit", 1)).alias("num_credit_cards"),
    count(when(col("CardType") == "Debit", 1)).alias("num_debit_cards"),
    count(when(col("CardType") == "Prepaid", 1)).alias("num_prepaid_cards"),
    max(datediff(col("ExpirationDate"), current_date())).alias("max_days_to_card_expiry")
)
```

### Support Call Features

Aggregates support call data by `CustomerID`:

- Number of calls and counts by issue type (Transaction Dispute, Loan Query, Card Issue, Account Access).
- Number of resolved calls and days since last resolved call.

```python
support_agg = support_calls_df.groupBy("CustomerID").agg(
    count("CallID").alias("num_support_calls"),
    count(when(col("IssueType") == "Transaction Dispute", 1)).alias("num_transaction_disputes"),
    count(when(col("IssueType") == "Loan Query", 1)).alias("num_loan_queries"),
    count(when(col("IssueType") == "Card Issue", 1)).alias("num_card_issues"),
    count(when(col("IssueType") == "Account Access", 1)).alias("num_account_access_issues"),
    count(when(col("Resolved") == "Yes", 1)).alias("num_resolved_calls"),
    max(when(col("Resolved") == "Yes", datediff(current_date(), col("CallDate")))).alias("days_since_last_resolved_call")
)
```

### Feature Aggregation

Joins all features and computes the `loan_to_balance_ratio`.

```python
features_df = customers_features.join(account_agg, "CustomerID", "left") \
    .join(trans_acc, "CustomerID", "left") \
    .join(loan_agg, "CustomerID", "left") \
    .join(card_agg, "CustomerID", "left") \
    .join(support_agg, "CustomerID", "left")

features_df = features_df.withColumn(
    "loan_to_balance_ratio",
    when(
        (col("total_balance").isNotNull()) & (col("total_balance") > 0),
        col("total_loan_amount") / col("total_balance")
    ).otherwise(0.0)
)
```

## Step 3: Data Preprocessing

The notebook cleans the data using a Data Wrangler-generated function, removing unnecessary columns, handling duplicates, and filling missing values.

```python
def clean_data(features_df):
    features_df = features_df.drop('last_balance_update', 'last_transaction_date')
    features_df = features_df.dropDuplicates()
    features_df = features_df.dropna(subset=['num_accounts'])
    features_df = features_df.fillna(value=0, subset=[
        'num_accounts', 'total_balance', 'num_savings_accounts', 'num_checking_accounts',
        'num_business_accounts', 'avg_balance', 'max_account_age_days', 'days_since_last_balance_update',
        'total_num_transactions', 'total_transaction_amount', 'total_deposits', 'total_withdrawals',
        'total_transfers', 'total_payments', 'days_since_last_transaction', 'num_loans',
        'total_loan_amount', 'avg_interest_rate', 'num_car_loans', 'num_personal_loans',
        'num_home_loans', 'num_education_loans', 'max_loan_duration_days', 'days_to_loan_end',
        'num_cards', 'num_credit_cards', 'num_debit_cards', 'num_prepaid_cards',
        'max_days_to_card_expiry', 'num_support_calls', 'num_transaction_disputes',
        'num_loan_queries', 'num_card_issues', 'num_account_access_issues', 'num_resolved_calls',
        'days_since_last_resolved_call', 'loan_to_balance_ratio'
    ])
    return features_df

features_df = clean_data(features_df)
```

## Step 4: Define Churn Label

The churn label is defined based on customer inactivity and account metrics:

- Days since last transaction &gt; 365.
- Total transactions ≤ 5 and days since last transaction ≥ 180.
- No cards.
- Days since last resolved call &gt; 180.
- One account with total balance &lt; 3000.

```python
df = features_df.withColumn(
    "churn",
    when(
        (col("days_since_last_transaction") > 365) |
        ((col("total_num_transactions") <= 5) & (col("days_since_last_transaction") >= 180)) |
        (col("num_cards") == 0) |
        (col("days_since_last_resolved_call") > 180) |
        (col("num_accounts") == 1) & (col("total_balance") < 3000),
        1
    ).otherwise(0)
)
```

## Step 5: Feature Scaling

The notebook normalizes features using `StringIndexer` for `state_code` and `MinMaxScaler` for all other columns (except `CustomerID`).

```python
from pyspark.ml import Pipeline
from pyspark.ml.feature import MinMaxScaler, VectorAssembler, StringIndexer
from pyspark.ml.functions import vector_to_array

def clean_data(df):
    indexer = StringIndexer(inputCol='state_code', outputCol='state_code_index')
    df = indexer.fit(df).transform(df).drop('state_code').withColumnRenamed('state_code_index', 'state_code')
    columns_to_normalize = [col for col in df.columns if col != 'CustomerID']
    assembler = VectorAssembler(inputCols=columns_to_normalize, outputCol="features")
    scaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures")
    pipeline = Pipeline(stages=[assembler, scaler])
    scaled_model = pipeline.fit(df)
    scaled_df = scaled_model.transform(df)
    scaled_df = scaled_df.withColumn("scaled_array", vector_to_array("scaledFeatures"))
    for i, col_name in enumerate(columns_to_normalize):
        scaled_df = scaled_df.withColumn(col_name, scaled_df["scaled_array"][i])
    df = scaled_df.drop("features", "scaledFeatures", "scaled_array")
    return df

df_clean = clean_data(df)
```

## Step 6: Train-Test Split

The data is split into training (95%) and test (5%) sets, with the test set saved as a Delta table.

```python
train_df, test_df = df_clean.randomSplit([0.95, 0.05], seed=42)
test_df.write.format("delta").saveAsTable("customer_churn_test")
```

## Step 7: AutoML with FLAML

The notebook uses FLAML for automated model selection and hyperparameter tuning, integrated with MLflow for experiment tracking.

### Install Dependencies

Ensures `scikit-learn` version 1.5.1 is installed.

```python
%pip install scikit-learn==1.5.1
```

### Configure Logging

Suppresses unnecessary warnings and logs for a cleaner output.

```python
import logging
import warnings

logging.getLogger('synapse.ml').setLevel(logging.CRITICAL)
logging.getLogger('mlflow.utils').setLevel(logging.CRITICAL)
warnings.simplefilter('ignore', category=FutureWarning)
warnings.simplefilter('ignore', category=UserWarning)
```

### Prepare Data for AutoML

Converts the training DataFrame to Pandas, renames columns to remove special characters, and limits to 100,000 rows.

```python
import re
import pandas as pd
import numpy as np

X = train_df.drop("CustomerID").limit(100000).toPandas()
X = X.rename(columns=lambda c: re.sub('[^A-Za-z0-9_]+', '_', c))
target_col = re.sub('[^A-Za-z0-9_]+', '_', "churn")
```

### Check Class Imbalance

Visualizes the class distribution and warns if the dominant class exceeds 80%.

```python
import matplotlib.pyplot as plt

distribution = X[target_col].value_counts(normalize=True)
dominant_class_proportion = distribution.max()

distribution.plot(kind='bar')
plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Proportion")
plt.show()

if dominant_class_proportion > 0.8:
    print(f"The dataset is imbalanced. The dominant class has {dominant_class_proportion * 100:.2f}% of the samples.")
    print("You may need to handle class imbalance before training the model.")
    print("For more information, see https://aka.ms/smote-example")
else:
    print("The dataset is balanced.")
```

### Featurization

Handles missing values using a `ColumnTransformer` with mean, median, or mode imputation based on feature skewness.

```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

X = X.convert_dtypes()
X = X.dropna(axis=1, how='all')
X = X.select_dtypes(include=['number', 'datetime', 'category'])

from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(X, test_size=0.2, random_state=41)

mean_features, median_features, mode_features = [], [], []
preprocessor, all_features, datetime_features = create_fillna_processor(X_train, mean_features, median_features, mode_features)
X_train = fillna(X_train, preprocessor, all_features, datetime_features)
X_test = fillna(X_test, preprocessor, all_features, datetime_features)

y_train = X_train.pop(target_col)
y_test = X_test.pop(target_col)
```

### Configure AutoML

Sets up FLAML with a 120-second time budget, binary classification task, and Spark for distributed training.

```python
import mlflow
from flaml import AutoML

mlflow.autolog(exclusive=False)
mlflow.set_experiment("Customer-Churn-Prediction")

settings = {
    "time_budget": 120,
    "task": "binary",
    "log_file_name": "flaml_experiment.log",
    "eval_method": "cv",
    "n_splits": 3,
    "max_iter": 10,
    "force_cancel": True,
    "seed": 41,
    "mlflow_exp_name": "Customer-Churn-Prediction",
    "use_spark": True,
    "n_concurrent_trials": 3,
    "verbose": 1,
    "featurization": "auto",
}
if flaml.__version__ > "2.3.3":
    settings["entrypoint"] = "low-code"

automl = AutoML(**settings)
```

### Run AutoML

Trains the model and logs results to MLflow.

```python
with mlflow.start_run(nested=True, run_name="Customer-Churn-Prediction-Model"):
    automl.fit(X_train=X_train, y_train=y_train)
```

### Save Model

Registers the best model in MLflow.

```python
model_path = f"runs:/{automl.best_run_id}/model"
registered_model = mlflow.register_model(model_uri=model_path, name="Customer-Churn-Prediction-Model")
print(f"Model '{registered_model.name}' version {registered_model.version} registered successfully.")
```

## Step 8: Generate Predictions

The notebook is truncated before generating predictions, but the test set (`customer_churn_test`) can be used to predict churn. Example code to complete this step:

```python
# Load registered model
model = mlflow.pyfunc.load_model(f"models:/Customer-Churn-Prediction-Model/{registered_model.version}")

# Predict on test set
test_X = test_df.drop("CustomerID", "churn").toPandas()
test_X = test_X.rename(columns=lambda c: re.sub('[^A-Za-z0-9_]+', '_', c))
predictions = model.predict(test_X)

# Save predictions
predictions_df = spark.createDataFrame(
    pd.DataFrame({
        "CustomerID": test_df.select("CustomerID").toPandas()["CustomerID"],
        "churn_prediction": predictions
    })
)
predictions_df.write.format("delta").saveAsTable("customer_churn_predictions")
```

## Evaluation and Visualization

### Evaluation

- **Metrics**: AutoML logs metrics (e.g., accuracy, precision, recall) to MLflow. Access them via the MLflow UI in Fabric.
- **Class Imbalance**: If the dataset is imbalanced, consider techniques like SMOTE (see notebook’s link).

### Power BI Visualization

1. Import the `customer_churn_predictions` Delta table into Power BI.
2. Create visualizations:
   - **Bar Chart**: Churn rate by `state_code`.
   - **Pie Chart**: Proportion of churned vs. non-churned customers.
   - **Scatter Plot**: `total_balance` vs. `total_loan_amount` colored by churn prediction.