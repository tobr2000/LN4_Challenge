#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 16:00:49 2024

@author: tobr2000
"""
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from imblearn.over_sampling import SMOTE

categorical_cols = [f'X{i}' for i in range(1, 21)]
dtype_spec = {col: 'object' for col in categorical_cols}

# Load the datasets
print("Loading datasets...")
data_train = pd.read_csv('data_train.csv', dtype=dtype_spec)
data_test = pd.read_csv('data_test.csv', dtype=dtype_spec)
y_train = pd.read_csv('Y_train.csv')

# Map categorical rankings to numerical values
print("Mapping categorical rankings to numerical values...")
ranking_map = {'level_1': 1, 'level_2': 2, 'level_3': 3}
rank_cols = [f'X{i}' for i in range(1, 21)]

for col in rank_cols:
    data_train[col] = data_train[col].map(ranking_map)
    data_test[col] = data_test[col].map(ranking_map)

# Split data into multiple time series dataframes
print("Splitting data into time series dataframes...")

def create_time_series(df, id_col, time_col, feature_cols):
    time_series_data = {}
    for col in feature_cols:
        df_pivot = df.pivot(index=id_col, columns=time_col, values=col)
        time_series_data[col] = df_pivot
    return time_series_data

# List of all feature columns
feature_cols = data_train.columns.difference(['ID', 'month'])

# Create time series dataframes for training and test sets
train_time_series = create_time_series(data_train, 'ID', 'month', feature_cols)
test_time_series = create_time_series(data_test, 'ID', 'month', feature_cols)

# Impute missing values for numerical features using KNN Imputer
print("Imputing missing values in time series dataframes...")
knn_imputer = KNNImputer(n_neighbors=5)

def impute_time_series(time_series_data):
    for col in time_series_data:
        time_series_data[col] = pd.DataFrame(knn_imputer.fit_transform(time_series_data[col]), 
                                             columns=time_series_data[col].columns, 
                                             index=time_series_data[col].index)
    return time_series_data

train_time_series_imputed = impute_time_series(train_time_series)
test_time_series_imputed = impute_time_series(test_time_series)

# Add expanding window statistics
def add_expanding_window_features(df):
    expanding_mean = df.expanding().mean()
    expanding_max = df.expanding().max()
    expanding_min = df.expanding().min()
    
    expanding_mean.columns = [f"{col}_expanding_mean" for col in df.columns]
    expanding_max.columns = [f"{col}_expanding_max" for col in df.columns]
    expanding_min.columns = [f"{col}_expanding_min" for col in df.columns]
    
    return pd.concat([df, expanding_mean, expanding_max, expanding_min], axis=1)

# Add mean and trend columns
print("Adding mean, trend, and expanding window columns...")

def add_features(time_series_data):
    for col in time_series_data:
        df = time_series_data[col]
        df['mean'] = df.mean(axis=1)
        df['trend'] = df.apply(lambda row: np.polyfit(range(len(row)), row, 1)[0], axis=1)
        df = add_expanding_window_features(df)
        time_series_data[col] = df
    return time_series_data

train_time_series_final = add_features(train_time_series_imputed)
test_time_series_final = add_features(test_time_series_imputed)

# Interaction features for original columns X21 to X40
def add_interaction_features(df, interaction_cols):
    interaction_data = pd.DataFrame(index=df.index)
    for i, col1 in enumerate(interaction_cols):
        for col2 in interaction_cols[i+1:]:
            interaction_data[f"{col1}_x_{col2}"] = df[col1] * df[col2]
    return interaction_data

interaction_cols = [f'X{i}' for i in range(21, 41)]
train_interactions = add_interaction_features(data_train[interaction_cols].copy(), interaction_cols)
test_interactions = add_interaction_features(data_test[interaction_cols].copy(), interaction_cols)

# Display the transformed data for a sample feature
sample_key = list(train_time_series_final.keys())[0]
sample_df = train_time_series_final[sample_key].head()

# Print the sample dataframe
print(sample_df)

# Function to merge time series dataframes
def merge_time_series_data(time_series_data):
    merged_data = pd.concat(time_series_data.values(), axis=1)
    merged_data.reset_index(inplace=True)
    return merged_data

# Merge the time series dataframes
print("Merging time series dataframes...")
data_train_merged = merge_time_series_data(train_time_series_final)
data_test_merged = merge_time_series_data(test_time_series_final)

# Merge interaction features back into the final dataset
data_train_merged = pd.concat([data_train_merged, train_interactions], axis=1)
data_test_merged = pd.concat([data_test_merged, test_interactions], axis=1)

# Align target data with the training data
print("Aligning target data...")
common_ids = set(data_train_merged['ID']).intersection(set(y_train['ID']))
data_train_filtered = data_train_merged[data_train_merged['ID'].isin(common_ids)].reset_index(drop=True)
y_train_filtered = y_train[y_train['ID'].isin(common_ids)].reset_index(drop=True)

# Ensure target data is aligned with the training data
y_train_aligned = y_train_filtered.set_index('ID').loc[data_train_filtered['ID']].reset_index()

# Convert all column names to strings
data_train_filtered.columns = data_train_filtered.columns.astype(str)
data_test_merged.columns = data_test_merged.columns.astype(str)

# Add Gaussian noise to the training data
print("Adding Gaussian noise to the training data...")
noise_factor = 0.1
X_train_noisy = data_train_filtered.drop(columns=['ID']) + noise_factor * np.random.normal(size=data_train_filtered.drop(columns=['ID']).shape)
X_train_noisy = pd.DataFrame(X_train_noisy, columns=data_train_filtered.drop(columns=['ID']).columns)

print(X_train_noisy.head())
# Prepare data for XGBoost
print("Preparing data for XGBoost...")
X = X_train_noisy
y = (y_train_aligned['Y'] == 'rec').astype(int)  # Convert target to binary

# Impute any remaining missing values before applying SMOTE
print("Imputing remaining missing values...")
X = pd.DataFrame(knn_imputer.fit_transform(X), columns=X.columns)

# Apply SMOTE for data augmentation
print("Applying SMOTE for data augmentation...")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Normalize the features
print("Normalizing the features...")
scaler = StandardScaler()
X_resampled = scaler.fit_transform(X_resampled)
data_test_final = scaler.transform(data_test_merged.drop(columns=['ID']))

# Split the data using TimeSeriesSplit
print("Splitting the data using TimeSeriesSplit...")
tscv = TimeSeriesSplit(n_splits=5)
auc_scores_xgb = []

for train_index, val_index in tscv.split(X_resampled):
    X_train_split, X_valid_split = X_resampled[train_index], X_resampled[val_index]
    y_train_split, y_valid_split = y_resampled[train_index], y_resampled[val_index]

    # Check if the validation set contains only one class
    if len(np.unique(y_valid_split)) == 1:
        print("Skipping fold due to only one class present in y_valid_split")
        continue

    # XGBoost model with regularization
    print("Training XGBoost model...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=100, 
        max_depth=3, 
        alpha=0.1, 
        reg_lambda=0.1, 
        random_state=42, 
        use_label_encoder=False, 
        eval_metric='logloss'
    )
    xgb_model.fit(X_train_split, y_train_split)
    y_valid_pred_xgb = xgb_model.predict_proba(X_valid_split)[:, 1]
    auc_xgb = roc_auc_score(y_valid_split, y_valid_pred_xgb)
    auc_scores_xgb.append(auc_xgb)
    print(f"Fold AUC (XGB): {auc_xgb}")

print(f'Average AUC score: {np.mean(auc_scores_xgb)}')

# Train final model on the entire training set
print("Training final XGBoost model on the entire training set...")
xgb_model.fit(X_resampled, y_resampled)

# Calculate and print AUC on the entire training set
y_train_pred_xgb = xgb_model.predict_proba(X_resampled)[:, 1]
train_auc_xgb = roc_auc_score(y_resampled, y_train_pred_xgb)
print(f'AUC on the entire training set: {train_auc_xgb}')

# Generate predictions on the test data
print("Generating predictions on the test data...")
y_test_pred_xgb = xgb_model.predict_proba(data_test_final)[:, 1]


# Aggregate predictions by ID
print("Aggregating predictions by ID...")
submission_xgb = pd.DataFrame({'ID': data_test_merged['ID'], 'rec': y_test_pred_xgb})


# Aggregate predictions by ID and average them
submission_xgb = submission_xgb.groupby('ID').agg({'rec': 'mean'}).reset_index()


# Prepare the submission file
submission_xgb.to_csv('sample_submission_xgb_vNoise02.csv', index=False)
print("Submission file created: sample_submission_xgb_vNoise02.csv")
