#################################################################################
# PREDICTIVE MAINTENANCE - OPTIMIZED MODELS WITH ACCURACY IMPROVEMENT
# Date: November 18, 2025
# Models: Logistic Regression, Naive Bayes, SVM, KNN, Decision Tree, Random Forest, XGBoost
#################################################################################

# Import all required libraries
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from imblearn.over_sampling import SMOTE

# Import all model libraries
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import seaborn as sns

print("="*80)
print("PREDICTIVE MAINTENANCE - MODEL OPTIMIZATION")
print("="*80)

#################################################################################
# STEP 1: LOAD AND PREPARE DATA
#################################################################################

print("\n[1/9] Loading and preparing data...")

# Load dataset
df = pd.read_csv("predictive_maintenance.csv")

print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Check for target column
if 'Target' in df.columns:
    target_col = 'Target'
elif 'Failure Type' in df.columns:
    target_col = 'Failure Type'
elif 'Machine failure' in df.columns:
    target_col = 'Machine failure'
else:
    print("Available columns:", list(df.columns))
    target_col = input("Enter the name of your target column: ")

# Prepare features and target
X = df.drop([target_col], axis=1, errors='ignore')
y = df[target_col]

# Remove ID columns
id_cols = ['UDI', 'Product ID', 'id', 'ID']
for col in id_cols:
    if col in X.columns:
        X = X.drop(col, axis=1)

# Encode categorical variables
categorical_cols = X.select_dtypes(include=['object']).columns
print(f"Encoding {len(categorical_cols)} categorical columns: {list(categorical_cols)}")

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# Encode target if categorical
if y.dtype == 'object' or y.dtype.name == 'category':
    le_target = LabelEncoder()
    y = le_target.fit_transform(y)
    print(f"Target classes: {le_target.classes_}")

print(f"Features shape: {X.shape}")
print(f"Target distribution:\n{pd.Series(y).value_counts()}")

#################################################################################
# NOTE: THIS FILE CONTAINS PART 1 OF THE COMPLETE CODE
# 
# FOR THE COMPLETE CODE WITH ALL 7 MODELS, PLEASE COPY THE FULL CODE FROM:
# The complete code I provided above in the chat
#
# OR: Download the complete code from the workspace after I add it all
#
# The complete code includes:
# - Data preprocessing (DONE ABOVE)
# - Train/Test split
# - Feature scaling  
# - SMOTE for class imbalance
# - 7 Optimized models:
#   1. Logistic Regression
#   2. Naive Bayes
#   3. SVM
#   4. KNN
#   5. Decision Tree
#   6. Random Forest
#   7. XGBoost
# - Model comparison and visualization
#################################################################################

print("\nPlease add the remaining code from the complete version")
print("See the chat for the full code with all 7 models!")

# ==============================================================================
# STEP 2: Train/Test Split
# ==============================================================================
print("\n" + "="*80)
print("STEP 2: Splitting data into train and test sets (80/20)")
print("="*80)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

# ==============================================================================
# STEP 3: Feature Scaling
# ==============================================================================
print("\n" + "="*80)
print("STEP 3: Scaling features using StandardScaler")
print("="*80)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Feature scaling completed successfully")

# ==============================================================================
# STEP 4: Handle Class Imbalance with SMOTE
# ==============================================================================
print("\n" + "="*80)
print("STEP 4: Applying SMOTE to handle class imbalance")
print("="*80)

print(f"Before SMOTE - Training set size: {X_train_scaled.shape[0]}")
print(f"Class distribution: {pd.Series(y_train).value_counts().to_dict()}")

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

print(f"\nAfter SMOTE - Training set size: {X_train_resampled.shape[0]}")
print(f"Class distribution: {pd.Series(y_train_resampled).value_counts().to_dict()}")


# ==============================================================================
# STEP 5-11: Train All 7 Models with Optimized Hyperparameters
# ==============================================================================
print("\n" + "="*80)
print("TRAINING 7 OPTIMIZED MODELS")
print("="*80)

# Dictionary to store all models and their results
models = {}
results = []

# ------------------------------------------------------------------------------
# MODEL 1: Logistic Regression
# ------------------------------------------------------------------------------
print("\n[1/7] Training Logistic Regression...")
log_reg = LogisticRegression(
    C=0.1,
    max_iter=2000,
    class_weight='balanced',
    random_state=42
)
log_reg.fit(X_train_resampled, y_train_resampled)
y_pred_lr = log_reg.predict(X_test_scaled)

accuracy_lr = accuracy_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr, average='weighted')
cv_scores_lr = cross_val_score(log_reg, X_train_resampled, y_train_resampled, cv=5)

models['Logistic Regression'] = log_reg
results.append({
    'Model': 'Logistic Regression',
    'Test Accuracy': accuracy_lr,
    'F1-Score': f1_lr,
    'CV Mean': cv_scores_lr.mean(),
    'CV Std': cv_scores_lr.std()
})

print(f"✓ Logistic Regression - Accuracy: {accuracy_lr:.4f} | F1-Score: {f1_lr:.4f}")

# ------------------------------------------------------------------------------
# MODEL 2: Naive Bayes
# ------------------------------------------------------------------------------
print("\n[2/7] Training Naive Bayes...")
nb_model = GaussianNB(var_smoothing=1e-9)
nb_model.fit(X_train_resampled, y_train_resampled)
y_pred_nb = nb_model.predict(X_test_scaled)

accuracy_nb = accuracy_score(y_test, y_pred_nb)
f1_nb = f1_score(y_test, y_pred_nb, average='weighted')
cv_scores_nb = cross_val_score(nb_model, X_train_resampled, y_train_resampled, cv=5)

models['Naive Bayes'] = nb_model
results.append({
    'Model': 'Naive Bayes',
    'Test Accuracy': accuracy_nb,
    'F1-Score': f1_nb,
    'CV Mean': cv_scores_nb.mean(),
    'CV Std': cv_scores_nb.std()
})

print(f"✓ Naive Bayes - Accuracy: {accuracy_nb:.4f} | F1-Score: {f1_nb:.4f}")


# ------------------------------------------------------------------------------
# MODEL 3: Support Vector Machine (SVM)
# ------------------------------------------------------------------------------
print("\n[3/7] Training Support Vector Machine...")
svm_model = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    class_weight='balanced',
    random_state=42
)
svm_model.fit(X_train_resampled, y_train_resampled)
y_pred_svm = svm_model.predict(X_test_scaled)

accuracy_svm = accuracy_score(y_test, y_pred_svm)
f1_svm = f1_score(y_test, y_pred_svm, average='weighted')
cv_scores_svm = cross_val_score(svm_model, X_train_resampled, y_train_resampled, cv=5)

models['SVM'] = svm_model
results.append({
    'Model': 'SVM',
    'Test Accuracy': accuracy_svm,
    'F1-Score': f1_svm,
    'CV Mean': cv_scores_svm.mean(),
    'CV Std': cv_scores_svm.std()
})

print(f"✓ SVM - Accuracy: {accuracy_svm:.4f} | F1-Score: {f1_svm:.4f}")

# ------------------------------------------------------------------------------
# MODEL 4: K-Nearest Neighbors (KNN)
# ------------------------------------------------------------------------------
print("\n[4/7] Training K-Nearest Neighbors...")
knn_model = KNeighborsClassifier(
    n_neighbors=7,
    weights='distance',
    metric='minkowski',
    p=2
)
knn_model.fit(X_train_resampled, y_train_resampled)
y_pred_knn = knn_model.predict(X_test_scaled)

accuracy_knn = accuracy_score(y_test, y_pred_knn)
f1_knn = f1_score(y_test, y_pred_knn, average='weighted')
cv_scores_knn = cross_val_score(knn_model, X_train_resampled, y_train_resampled, cv=5)

models['KNN'] = knn_model
results.append({
    'Model': 'KNN',
    'Test Accuracy': accuracy_knn,
    'F1-Score': f1_knn,
    'CV Mean': cv_scores_knn.mean(),
    'CV Std': cv_scores_knn.std()
})

print(f"✓ KNN - Accuracy: {accuracy_knn:.4f} | F1-Score: {f1_knn:.4f}")


# ------------------------------------------------------------------------------
# MODEL 5: Decision Tree Classifier
# ------------------------------------------------------------------------------
print("\n[5/7] Training Decision Tree Classifier...")
dt_model = DecisionTreeClassifier(
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42
)
dt_model.fit(X_train_resampled, y_train_resampled)
y_pred_dt = dt_model.predict(X_test_scaled)

accuracy_dt = accuracy_score(y_test, y_pred_dt)
f1_dt = f1_score(y_test, y_pred_dt, average='weighted')
cv_scores_dt = cross_val_score(dt_model, X_train_resampled, y_train_resampled, cv=5)

models['Decision Tree'] = dt_model
results.append({
    'Model': 'Decision Tree',
    'Test Accuracy': accuracy_dt,
    'F1-Score': f1_dt,
    'CV Mean': cv_scores_dt.mean(),
    'CV Std': cv_scores_dt.std()
})

print(f"✓ Decision Tree - Accuracy: {accuracy_dt:.4f} | F1-Score: {f1_dt:.4f}")

# ------------------------------------------------------------------------------
# MODEL 6: Random Forest Classifier
# ------------------------------------------------------------------------------
print("\n[6/7] Training Random Forest Classifier...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_resampled, y_train_resampled)
y_pred_rf = rf_model.predict(X_test_scaled)

accuracy_rf = accuracy_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf, average='weighted')
cv_scores_rf = cross_val_score(rf_model, X_train_resampled, y_train_resampled, cv=5)

models['Random Forest'] = rf_model
results.append({
    'Model': 'Random Forest',
    'Test Accuracy': accuracy_rf,
    'F1-Score': f1_rf,
    'CV Mean': cv_scores_rf.mean(),
    'CV Std': cv_scores_rf.std()
})

print(f"✓ Random Forest - Accuracy: {accuracy_rf:.4f} | F1-Score: {f1_rf:.4f}")

# ------------------------------------------------------------------------------
# MODEL 7: XGBoost Classifier
# ------------------------------------------------------------------------------
print("\n[7/7] Training XGBoost Classifier...")
xgb_model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
xgb_model.fit(X_train_resampled, y_train_resampled)
y_pred_xgb = xgb_model.predict(X_test_scaled)

accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
f1_xgb = f1_score(y_test, y_pred_xgb, average='weighted')
cv_scores_xgb = cross_val_score(xgb_model, X_train_resampled, y_train_resampled, cv=5)

models['XGBoost'] = xgb_model
results.append({
    'Model': 'XGBoost',
    'Test Accuracy': accuracy_xgb,
    'F1-Score': f1_xgb,
    'CV Mean': cv_scores_xgb.mean(),
    'CV Std': cv_scores_xgb.std()
})

print(f"✓ XGBoost - Accuracy: {accuracy_xgb:.4f} | F1-Score: {f1_xgb:.4f}")


# ==============================================================================
# STEP 12: Create Results Comparison DataFrame
# ==============================================================================
print("\n" + "="*80)
print("MODEL COMPARISON RESULTS")
print("="*80)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Test Accuracy', ascending=False)

print("\n" + results_df.to_string(index=False))
print("\n" + "="*80)

# ==============================================================================
# STEP 13: Detailed Classification Reports
# ==============================================================================
print("\n" + "="*80)
print("DETAILED CLASSIFICATION REPORTS FOR ALL MODELS")
print("="*80)

print("\n" + "-"*80)
print("1. LOGISTIC REGRESSION")
print("-"*80)
print(classification_report(y_test, y_pred_lr))

print("\n" + "-"*80)
print("2. NAIVE BAYES")
print("-"*80)
print(classification_report(y_test, y_pred_nb))

print("\n" + "-"*80)
print("3. SUPPORT VECTOR MACHINE (SVM)")
print("-"*80)
print(classification_report(y_test, y_pred_svm))

print("\n" + "-"*80)
print("4. K-NEAREST NEIGHBORS (KNN)")
print("-"*80)
print(classification_report(y_test, y_pred_knn))

print("\n" + "-"*80)
print("5. DECISION TREE")
print("-"*80)
print(classification_report(y_test, y_pred_dt))

print("\n" + "-"*80)
print("6. RANDOM FOREST")
print("-"*80)
print(classification_report(y_test, y_pred_rf))

print("\n" + "-"*80)
print("7. XGBOOST")
print("-"*80)
print(classification_report(y_test, y_pred_xgb))


# ==============================================================================
# STEP 14: Visualize Model Comparison
# ==============================================================================
print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

plt.figure(figsize=(14, 6))

# Subplot 1: Test Accuracy Comparison
plt.subplot(1, 2, 1)
plt.barh(results_df['Model'], results_df['Test Accuracy'], color='skyblue', edgecolor='navy')
plt.xlabel('Test Accuracy', fontsize=12, fontweight='bold')
plt.title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
plt.xlim([0, 1])
plt.grid(axis='x', alpha=0.3)
for i, v in enumerate(results_df['Test Accuracy']):
    plt.text(v + 0.01, i, f'{v:.4f}', va='center', fontsize=10)

# Subplot 2: F1-Score Comparison
plt.subplot(1, 2, 2)
plt.barh(results_df['Model'], results_df['F1-Score'], color='lightcoral', edgecolor='darkred')
plt.xlabel('F1-Score', fontsize=12, fontweight='bold')
plt.title('Model F1-Score Comparison', fontsize=14, fontweight='bold')
plt.xlim([0, 1])
plt.grid(axis='x', alpha=0.3)
for i, v in enumerate(results_df['F1-Score']):
    plt.text(v + 0.01, i, f'{v:.4f}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualization saved as 'model_comparison.png'")
plt.show()

# Additional visualization: Cross-validation scores
plt.figure(figsize=(12, 6))
plt.barh(results_df['Model'], results_df['CV Mean'], 
         xerr=results_df['CV Std'], 
         color='mediumseagreen', 
         edgecolor='darkgreen',
         capsize=5)
plt.xlabel('Cross-Validation Mean Accuracy', fontsize=12, fontweight='bold')
plt.title('5-Fold Cross-Validation Results', fontsize=14, fontweight='bold')
plt.xlim([0, 1])
plt.grid(axis='x', alpha=0.3)
for i, (mean, std) in enumerate(zip(results_df['CV Mean'], results_df['CV Std'])):
    plt.text(mean + 0.01, i, f'{mean:.4f} ± {std:.4f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('cross_validation_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Cross-validation visualization saved as 'cross_validation_comparison.png'")
plt.show()


# ==============================================================================
# STEP 15: Final Summary and Best Model
# ==============================================================================
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

best_model_name = results_df.iloc[0]['Model']
best_accuracy = results_df.iloc[0]['Test Accuracy']
best_f1 = results_df.iloc[0]['F1-Score']
best_cv_mean = results_df.iloc[0]['CV Mean']
best_cv_std = results_df.iloc[0]['CV Std']

print(f"\n🏆 BEST PERFORMING MODEL: {best_model_name}")
print("-" * 80)
print(f"   Test Accuracy:        {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
print(f"   F1-Score:             {best_f1:.4f}")
print(f"   CV Mean Accuracy:     {best_cv_mean:.4f} ± {best_cv_std:.4f}")

print("\n" + "="*80)
print("MODEL RANKING BY ACCURACY:")
print("="*80)
for idx, row in results_df.iterrows():
    print(f"{idx+1}. {row['Model']:20s} - Accuracy: {row['Test Accuracy']:.4f} | F1: {row['F1-Score']:.4f}")

print("\n" + "="*80)
print("OPTIMIZATION SUMMARY")
print("="*80)
print("\nOptimizations Applied:")
print("  ✓ Feature Scaling (StandardScaler)")
print("  ✓ Class Imbalance Handling (SMOTE)")
print("  ✓ Optimized Hyperparameters for Each Model")
print("  ✓ Cross-Validation (5-fold)")
print("  ✓ Balanced Class Weights (where applicable)")

print("\nHyperparameters Used:")
print("  - Logistic Regression: C=0.1, max_iter=2000")
print("  - Naive Bayes: var_smoothing=1e-9")
print("  - SVM: kernel='rbf', C=1.0, gamma='scale'")
print("  - KNN: n_neighbors=7, weights='distance'")
print("  - Decision Tree: max_depth=10, min_samples_split=10")
print("  - Random Forest: n_estimators=200, max_depth=15")
print("  - XGBoost: n_estimators=200, learning_rate=0.1, max_depth=6")

print("\n" + "="*80)
print("✓ ALL MODELS TRAINED AND EVALUATED SUCCESSFULLY!")
print("="*80)
print(f"\nTotal execution completed at: {pd.Timestamp.now()}")
print("\nOutput Files Generated:")
print("  1. model_comparison.png")
print("  2. cross_validation_comparison.png")
print("\nYou can now use the best model for predictions!")
print("="*80)
