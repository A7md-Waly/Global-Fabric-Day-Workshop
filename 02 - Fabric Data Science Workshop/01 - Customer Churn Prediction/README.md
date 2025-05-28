# Customer Churn Prediction Notebook

This notebook builds a machine learning model to predict customer churn for a banking institution using Microsoft Fabric’s PySpark and FLAML AutoML. Churn prediction helps identify customers at risk of leaving, enabling proactive retention strategies. The process includes loading data from a Lakehouse, engineering features, labeling churn, preprocessing, training an AutoML model, and saving the model for deployment.

> **Note**: This notebook assumes a Fabric environment with Spark 3.4+ and the `Banking_Data` Lakehouse tables. The sample data has mismatched `CustomerID` values across tables, which may result in empty joins. Preprocess the data to align IDs for accurate results.

## Step 1: Initialize Spark Session

### Introduction
A Spark session is created to enable distributed data processing in Fabric’s Lakehouse environment. This sets up the foundation for handling large-scale banking data.

### Actions
- Initialize a Spark session named `CustomerChurn`.
- Import necessary PySpark functions for data manipulation.

## Step 2: Load Data from Lakehouse

### Introduction
Raw data is loaded from the `Banking_Data` Lakehouse, containing customer, account, transaction, loan, card, and support call information. These tables provide a comprehensive view of customer behavior.

### Actions
- Load six Delta tables: `Customers`, `Accounts`, `Transactions`, `Loans`, `Cards`, and `SupportCalls`.
- Use Spark SQL to query the Lakehouse tables.

> **Data Note**: The sample data has non-overlapping `CustomerID` values (e.g., `Customers`: 1-6, `Accounts`: 735-4693). Ensure IDs align across tables before proceeding. For testing, update `CustomerID` in `Accounts`, `Loans`, `Cards`, and `SupportCalls` to match `Customers` (e.g., 1-6).

## Step 3: Engineer Features

### Introduction
Features are created by aggregating and transforming raw data to capture customer behavior, such as account balances, transaction frequency, and support interactions. These features are critical for predicting churn in banking.

### Actions
- **Customers**: Extract `state_code` from `Address` and calculate `tenure_days` (days since `JoinDate`).
- **Accounts**: Aggregate by `CustomerID` to compute number of accounts, total and average balance, and account age.
- **Transactions**: Aggregate by `AccountID`, then by `CustomerID`, to calculate transaction counts, amounts, and recency.
- **Loans**: Aggregate by `CustomerID` for loan counts, amounts, interest rates, and durations.
- **Cards**: Aggregate by `CustomerID` for card counts and days to expiry.
- **Support Calls**: Aggregate by `CustomerID` for call counts, issue types, and resolution status.
- Join all aggregates on `CustomerID` using left joins.
- Compute `loan_to_balance_ratio` to measure debt relative to savings.
- Display the resulting `features_df`.

> **Why It Matters**: Features like `days_since_last_transaction` and `num_support_calls` signal customer engagement, key indicators of churn risk.

## Step 4: Clean and Preprocess Data

### Introduction
Data is cleaned to handle missing values, duplicates, and inconsistencies, ensuring quality input for the model. This step uses Data Wrangler-generated code for reproducibility.

### Actions
- Drop unneeded columns: `last_balance_update`, `last_transaction_date`.
- Remove duplicate rows.
- Drop rows with missing `num_accounts`.
- Fill missing values with 0 for 37 numeric columns (e.g., `total_balance`, `num_loans`).
- Display the cleaned `features_df`.

> **Fabric Tip**: Use Data Wrangler to explore missing values and generate cleaning code interactively.

## Step 5: Label Churn

### Introduction
A binary `churn` label (1 = churned, 0 = not churned) is created based on banking-specific criteria, such as inactivity or low engagement. This label is the target variable for the model.

### Actions
- Define churn as:
  - `days_since_last_transaction` > 365 (inactive for over a year).
  - `total_num_transactions` ≤ 5 and `days_since_last_transaction` ≥ 180 (low activity).
  - `num_cards` = 0 (no cards).
  - `days_since_last_resolved_call` > 180 (unresolved issues).
  - `num_accounts` = 1 and `total_balance` < 3000 (low-value single account).
- Add `churn` column to `features_df`.
- Display the DataFrame with the new label.

> **Why It Matters**: These rules reflect banking churn patterns, such as customers who stop transacting or have unresolved issues.

## Step 6: Normalize Features

### Introduction
Features are normalized to a 0-1 scale to ensure fair comparison across different units (e.g., dollars vs. days). A StringIndexer encodes `state_code` for model compatibility.

### Actions
- Encode `state_code` as numeric using `StringIndexer`.
- Assemble all columns (except `CustomerID`) into a feature vector.
- Apply `MinMaxScaler` to normalize the feature vector.
- Split the scaled vector back into individual columns.
- Drop temporary columns (`features`, `scaledFeatures`, `scaled_array`).
- Display the normalized `df_clean`.

> **Fabric Tip**: Data Wrangler simplifies feature scaling and encoding tasks with a low-code interface.

## Step 7: Split and Save Data

### Introduction
The dataset is split into training and test sets for model training and evaluation. Data is saved as CSV and Delta tables for persistence and downstream use (e.g., Power BI).

### Actions
- Split `df_clean` into `train_df` (90%) and `test_df` (10%) with a random seed of 42.
- Save `train_df` and `test_df` as CSV files in `Files/customer_churn`.
- Save `test_df` as a Delta table named `customer_churn_test`.

> **Note**: The commented-out line for saving `train_df` as a Delta table can be enabled if needed. Ensure Lakehouse permissions allow table creation.

## Step 8: Install Dependencies

### Introduction
Specific library versions are installed to ensure compatibility with FLAML AutoML in the Fabric environment.

### Actions
- Install `scikit-learn==1.5.1` using pip.

## Step 9: Configure Logging

### Introduction
Logging and warning settings are adjusted to reduce clutter and focus on critical outputs, improving notebook readability.

### Actions
- Suppress logs from `synapse.ml` and `mlflow.utils`.
- Ignore `FutureWarning` and `UserWarning` messages.

## Step 10: Load Data for AutoML

### Introduction
The training data is converted to a Pandas DataFrame for FLAML compatibility. Column names are cleaned to avoid issues during model training.

### Actions
- Convert `train_df` (up to 100,000 rows) to Pandas DataFrame `X`.
- Replace special characters in column names with underscores.
- Define `target_col` as `churn`.
- Display the prepared DataFrame.

> **Note**: The `limit(100000)` is for performance. Remove it to use all data if your dataset is small or compute resources allow.

## Step 11: Check Class Imbalance

### Introduction
Class imbalance in the `churn` column is assessed, as it can affect model performance. If one class dominates, techniques like oversampling may be needed.

### Actions
- Compute and plot the class distribution of `churn`.
- If the dominant class exceeds 80%, print a warning and suggest imbalance handling (e.g., SMOTE).
- Otherwise, confirm the dataset is balanced.

> **Why It Matters**: Imbalanced data can bias the model toward the majority class, reducing accuracy for churned customers.

## Step 12: Prepare Features for AutoML

### Introduction
Features are preprocessed to handle missing values and select appropriate data types, ensuring compatibility with FLAML’s AutoML.

### Actions
- Convert object columns to optimal dtypes.
- Drop columns with all missing values.
- Select numeric, datetime, and categorical columns for training.
- Split `X` into `X_train` (80%) and `X_test` (20%) with a random seed of 41.
- Create a `ColumnTransformer` to impute missing values:
  - Mean for low-skew numeric features.
  - Median for high-skew numeric features.
  - Mode for categorical features.
- Apply the imputer to `X_train` and `X_test`.
- Extract `y_train` and `y_test` as the `churn` column.
- Display the first 10 rows of `X_train`.

> **Note**: The sample data shows sparse features (many missing values). Ensure sufficient data after joins to avoid empty DataFrames.

## Step 13: Configure MLflow Experiment

### Introduction
MLflow is set up to track AutoML experiments, logging metrics, parameters, and models for comparison and reproducibility.

### Actions
- Enable MLflow autologging.
- Set the experiment name to `Customer-Churn-Prediction`.

> **Fabric Tip**: Use the MLflow UI in Fabric to compare model runs and select the best performer.

## Step 14: Configure AutoML Trial

### Introduction
FLAML’s AutoML is configured to find the best model for churn prediction. Settings control the trial duration, task type, and evaluation method.

### Actions
- Define AutoML settings:
  - `time_budget`: 3600 seconds (1 hour).
  - `task`: Binary classification.
  - `eval_method`: 3-fold cross-validation.
  - `n_concurrent_trials`: 3 for parallel processing.
  - `use_spark`: True for distributed training in Fabric.
  - `featurization`: Auto for automatic feature engineering.
- Create an `AutoML` instance with these settings.

## Step 15: Run AutoML Trial

### Introduction
The AutoML trial tests multiple models and hyperparameters to identify the best churn prediction model. Results are tracked in MLflow.

### Actions
- Run `automl.fit` with `X_train` and `y_train` within an MLflow run named `Customer-Churn-Prediction-Model`.

> **Why It Matters**: AutoML automates model selection, saving time and improving accuracy for banking churn prediction.

## Step 16: Save the Final Model

### Introduction
The best model from the AutoML trial is registered in MLflow for deployment and future use, ensuring traceability.

### Actions
- Register the model using the `best_run_id` as `Customer-Churn-Prediction-Model`.
- Print the registered model’s name and version.

> **Fabric Tip**: Deploy the registered model in Fabric for real-time churn predictions or integrate with Power BI for reporting.

## Additional Notes for Workshop

### Data Preparation
- **Sample Data Issue**: The provided sample data has mismatched `CustomerID` values, leading to empty joins. To fix:
  - Update `CustomerID` in `Accounts`, `Loans`, `Cards`, `Transactions`, and `SupportCalls` to match `Customers` (e.g., 1-6).
  - Alternatively, use a larger dataset with consistent IDs.
- **Verification**: After loading data, check DataFrame counts (e.g., `customers_df.count()`) to ensure joins are not empty.

### Power BI Visualizations
- **Churn Distribution**: Pie chart of `churn` (0 vs. 1).
- **Feature Insights**: Scatter plot of `total_balance` vs. `days_since_last_transaction`, colored by `churn`.
- **Risk Segments**: Bar chart of average `total_loan_amount` by `churn`.
- **Geographic Trends**: Map of `state_code` with churn rate.
- Import `customer_churn_test` Delta table and model predictions into Power BI.

### MLflow Usage
- Access the MLflow UI in Fabric to compare runs.
- Sort by metrics like accuracy or AUC to select the best model.
- Log additional metrics (e.g., precision, recall) if needed for banking use cases.

### Performance Tips
- If `toPandas()` is slow, reduce the dataset size or use Spark-native preprocessing.
- Adjust `time_budget` in AutoML for faster prototyping (e.g., 600 seconds for demos).
- If class imbalance is detected, apply SMOTE (see `https://aka.ms/smote-example`) before AutoML.

This notebook provides a robust pipeline for churn prediction, leveraging Fabric’s capabilities for scalable data processing and automated machine learning.