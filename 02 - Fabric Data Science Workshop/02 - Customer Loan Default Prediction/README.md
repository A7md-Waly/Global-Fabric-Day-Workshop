# Customer Loan Default Prediction Documentation

## Introduction

The `Customer Loan Default Prediction.ipynb` notebook is designed for a Fabric Data Science workshop focused on banking analytics. It builds a binary classification model to predict loan defaults using Microsoft Fabric’s PySpark for data processing and FLAML (Fast and Lightweight AutoML) for model training. The notebook leverages a banking dataset stored in Delta tables within a Fabric Lakehouse, tracks experiments with MLflow, generates batch predictions using Fabric’s `PREDICT` function, and saves results for visualization in Power BI.

### Objectives

- Predict which loans are likely to default based on customer and loan characteristics.
- Demonstrate an end-to-end ML workflow in Fabric: data loading, preprocessing, modeling, and prediction.
- Enable participants to visualize default predictions and inform credit risk strategies.

### Dataset

The dataset includes five Delta tables:

- **Customers**: CustomerID, Address, JoinDate.
- **Accounts**: AccountID, CustomerID, AccountType, Balance, CreatedDate.
- **Transactions**: TransactionID, AccountID, TransactionType, Amount, TransactionDate.
- **Loans**: LoanID, CustomerID, LoanType, LoanAmount, InterestRate, LoanStartDate, LoanEndDate.
- **Cards**: CardID, CustomerID, CardType, ExpirationDate.

## Setup

### Prerequisites

- **Fabric Workspace**: Access to a Fabric environment with a Lakehouse containing the Delta tables.

- **Spark Cluster**: Fabric Runtime 1.2+ or Spark 3.4+ for AutoML compatibility.

### Environment Configuration

1. Create a notebook in Fabric and attach it to a Spark cluster.
2. Install dependencies using the above command or Fabric’s environment management.
3. Ensure Delta tables are accessible at `Banking_Data.<table_name>`.

### Notebook Initialization

The notebook initializes a Spark session for data processing.

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, sum, avg, max, when, current_date, datediff, substring

spark = SparkSession.builder.appName("CustomerLoanDefault").getOrCreate()
```

## Step 1: Load the Data

The notebook loads data from Delta tables in the Fabric Lakehouse using Spark SQL.

```python
customers_df = spark.sql("SELECT * FROM Banking_Data.Customers")
accounts_df = spark.sql("SELECT * FROM Banking_Data.Accounts")
transactions_df = spark.sql("SELECT * FROM Banking_Data.Transactions")
loans_df = spark.sql("SELECT * FROM Banking_Data.Loans")
cards_df = spark.sql("SELECT * FROM Banking_Data.Cards")
```

## Step 2: Feature Engineering

The notebook aggregates and transforms data to create features for loan default prediction, joined by `CustomerID` or `LoanID`.

### Customer Features

- **state_code**: Last two characters of `Address`.
- **tenure_days**: Days from `JoinDate` to May 29, 2025.

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
- Days since last balance update.

```python
account_agg = accounts_df.groupBy("CustomerID").agg(
    count("AccountID").alias("num_accounts"),
    sum("Balance").alias("total_balance"),
    count(when(col("AccountType") == "Savings", 1)).alias("num_savings_accounts"),
    count(when(col("AccountType") == "Checking", 1)).alias("num_checking_accounts"),
    count(when(col("AccountType") == "Business", 1)).alias("num_business_accounts"),
    avg("Balance").alias("avg_balance"),
    datediff(current_date(), max("CreatedDate")).alias("days_since_last_balance_update")
)
```

### Transaction Features

Aggregates transactions by `AccountID`, then by `CustomerID`:

- Total transactions, amount, and days since last transaction.

```python
transaction_agg = transactions_df.groupBy("AccountID").agg(
    count("TransactionID").alias("num_transactions"),
    sum("Amount").alias("total_transaction_amount"),
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
    max("last_transaction_date").alias("last_transaction_date"),
    max("days_since_last_transaction").alias("days_since_last_transaction")
)
```

### Loan Features

Extracts per-loan features and aggregates by `CustomerID`:

- Per-loan: `LoanID`, `LoanType`, `LoanAmount`, `InterestRate`, loan duration, days to loan end.
- Aggregated: Number of loans, total amount, counts by type (Car, Personal, Home, Education).

```python
loan_features = loans_df.select(
    "CustomerID",
    "LoanID",
    "LoanType",
    "LoanAmount",
    "InterestRate",
    datediff(col("LoanEndDate"), col("LoanStartDate")).alias("loan_duration_days"),
    when(col("LoanEndDate") <= current_date(), 0)
        .otherwise(datediff(col("LoanEndDate"), current_date()))
        .alias("days_to_loan_end")
)

loan_agg = loans_df.groupBy("CustomerID").agg(
    count("LoanID").alias("num_loans"),
    sum("LoanAmount").alias("total_loan_amount"),
    count(when(col("LoanType") == "Car", 1)).alias("total_car_loans"),
    count(when(col("LoanType") == "Personal", 1)).alias("total_personal_loans"),
    count(when(col("LoanType") == "Home", 1)).alias("total_home_loans"),
    count(when(col("LoanType") == "Education", 1)).alias("total_education_loans")
)
```

### Card Features

Aggregates card data by `CustomerID`:

- Number of cards and counts by type (Credit, Debit, Prepaid).

```python
card_agg = cards_df.groupBy("CustomerID").agg(
    count("CardID").alias("num_cards"),
    count(when(col("CardType") == "Credit", 1)).alias("num_credit_cards"),
    count(when(col("CardType") == "Debit", 1)).alias("num_debit_cards"),
    count(when(col("CardType") == "Prepaid", 1)).alias("num_prepaid_cards")
)
```

### Feature Aggregation

Joins features and computes derived metrics:

- **loan_to_balance_ratio**: `total_loan_amount` / `total_balance` (0 if undefined).
- **transaction_recency_score**: `1 / (1 + days_since_last_transaction / 30)`.

```python
features_df = loan_features.join(customers_features, "CustomerID", "left") \
    .join(account_agg, "CustomerID", "left") \
    .join(trans_acc, "CustomerID", "left") \
    .join(loan_agg, "CustomerID", "left") \
    .join(card_agg, "CustomerID", "left")

features_df = features_df.withColumn(
    "loan_to_balance_ratio",
    when(
        (col("total_balance").isNotNull()) & (col("total_balance") > 0),
        col("total_loan_amount") / col("total_balance")
    ).otherwise(0.0)
).withColumn(
    "transaction_recency_score",
    when(
        col("days_since_last_transaction").isNotNull(),
        1.0 / (1.0 + col("days_since_last_transaction") / 30.0)
    ).otherwise(0.0)
)
```

## Step 3: Data Preprocessing

The notebook cleans the data using a Data Wrangler-generated function, removing columns, handling duplicates, filtering outliers, and filling missing values.

```python
def clean_data(features_df):
    features_df = features_df.drop('CustomerID', 'last_transaction_date')
    features_df = features_df.fillna(value=0, subset=[
        'num_accounts', 'total_balance', 'num_savings_accounts', 'num_checking_accounts',
        'num_business_accounts', 'avg_balance', 'days_since_last_balance_update',
        'total_num_transactions', 'total_transaction_amount', 'days_since_last_transaction',
        'total_loan_amount', 'num_cards', 'num_credit_cards', 'num_debit_cards',
        'num_prepaid_cards', 'loan_to_balance_ratio', 'transaction_recency_score'
    ])
    features_df = features_df.dropDuplicates()
    features_df = features_df.filter((features_df['loan_to_balance_ratio'] > -1) & (features_df['loan_to_balance_ratio'] < 50))
    return features_df

df_clean = clean_data(features_df)
```

## Step 4: Define Default Label

The default label is defined based on loan and customer metrics:

- Interest rate &gt; 10%.
- Total balance &lt; $10,000.
- Loan-to-balance ratio &gt; 10.
- Transaction recency score &lt; 0.08.

```python
df_clean = df_clean.withColumn(
    "default",
    when(
        (col("InterestRate") > 10) |
        (col("total_balance") < 10000) |
        (col("loan_to_balance_ratio") > 10) |
        (col("transaction_recency_score") < 0.08),
        1
    ).otherwise(0)
)
```

## Step 5: Feature Scaling

The notebook encodes `LoanType` with one-hot encoding and normalizes numeric features using `MinMaxScaler`.

```python
from pyspark.ml.feature import MinMaxScaler, VectorAssembler
from pyspark.sql.functions import when
from pyspark.sql.types import FloatType, IntegerType
from pyspark.ml.functions import vector_to_array

def clean_data(df_clean):
    loan_types = df_clean.select('LoanType').distinct().rdd.flatMap(lambda x: x).collect()
    for loan_type in loan_types:
        df_clean = df_clean.withColumn(f'LoanType_{loan_type}', when(df_clean['LoanType'] == loan_type, 1).otherwise(0))
    numeric_columns = [field.name for field in df_clean.schema.fields if field.dataType in [IntegerType(), FloatType()] and field.name != 'LoanID']
    assembler = VectorAssembler(inputCols=numeric_columns, outputCol='features')
    df_vector = assembler.transform(df_clean)
    scaler = MinMaxScaler(inputCol='features', outputCol='scaled_features')
    scaler_model = scaler.fit(df_vector)
    df_scaled = scaler_model.transform(df_vector)
    df_scaled = df_scaled.withColumn("scaled_array", vector_to_array("scaled_features"))
    for i, col_name in enumerate(numeric_columns):
        df_scaled = df_scaled.withColumn(col_name, df_scaled["scaled_array"][i])
    df_clean = df_scaled.drop("LoanType", "features", "scaled_features", "scaled_array")
    return df_clean

df_clean = clean_data(df_clean)
```

## Step 6: Train-Test Split

The data is split into training (95%) and test (5%) sets, with the test set saved as a Delta table.

```python
train_df, test_df = df_clean.randomSplit([0.95, 0.05], seed=42)
test_df.write.format("delta").saveAsTable("customer_default_test")
```

## Step 7: AutoML with FLAML

The notebook uses FLAML for automated model selection and hyperparameter tuning, integrated with MLflow for experiment tracking.

### Install Dependencies

Ensures `scikit-learn` version 1.5.1 is installed.

```python
%pip install scikit-learn==1.5.1
```

### Configure Logging

Suppresses unnecessary warnings and logs.

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

X = train_df.drop("LoanID").limit(100000).toPandas()
X = X.rename(columns=lambda c: re.sub('[^A-Za-z0-9_]+', '_', c))
target_col = re.sub('[^A-Za-z0-9_]+', '_', "default")
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
mlflow.set_experiment("Customer-Loan-Default-Prediction")

settings = {
    "time_budget": 120,
    "task": "binary",
    "log_file_name": "flaml_experiment.log",
    "eval_method": "cv",
    "n_splits": 3,
    "max_iter": 10,
    "force_cancel": True,
    "seed": 41,
    "mlflow_exp_name": "Customer-Loan-Default-Prediction",
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
with mlflow.start_run(nested=True, run_name="Customer-Loan-Default-Prediction-Model"):
    automl.fit(X_train=X_train, y_train=y_train)
```

### Save Model

Registers the best model in MLflow.

```python
model_path = f"runs:/{automl.best_run_id}/model"
registered_model = mlflow.register_model(model_uri=model_path, name="Customer-Loan-Default-Prediction-Model")
print(f"Model '{registered_model.name}' version {registered_model.version} registered successfully.")
```

## Step 8: Generate Predictions

The notebook uses Fabric’s `PREDICT` function via `MLFlowTransformer` to generate batch predictions on the test set, which are saved as a Delta table.

```python
from synapse.ml.predict import MLFlowTransformer

feature_cols = X_train.columns.to_list()
model = MLFlowTransformer(
    inputCols=feature_cols,
    outputCol=target_col,
    modelName="Customer-Loan-Default-Prediction-Model",
    modelVersion=registered_model.version,
)

df_test = spark.createDataFrame(X_test)
batch_predictions = model.transform(df_test)

saved_name = "Tables/customer_default_test_predictions".replace(".", "_")
batch_predictions.write.mode("overwrite").format("delta").option("overwriteSchema", "true").save(saved_name)
```

## Evaluation and Visualization

### Evaluation

- **Metrics**: AutoML logs metrics (e.g., accuracy, precision, recall, AUC) to MLflow. Access them via the MLflow UI in Fabric.
- **Class Imbalance**: If the dataset is imbalanced, consider techniques like SMOTE (see notebook’s link).

### Power BI Visualization

1. Import the `customer_default_test_predictions` Delta table into Power BI.
2. Create visualizations:
   - **Bar Chart**: Default rate by `state_code`.
   - **Pie Chart**: Proportion of defaulted vs. non-defaulted loans.
   - **Scatter Plot**: `LoanAmount` vs. `total_balance` colored by default prediction.