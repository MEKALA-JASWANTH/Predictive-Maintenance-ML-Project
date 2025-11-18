# 🔧 Predictive Maintenance ML Project

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/ML-7%20Models-green)
![Status](https://img.shields.io/badge/Status-Active-success)

---

<div align="center">
  <img src="https://raw.githubusercontent.com/MEKALA-JASWANTH/Predictive-Maintenance-ML-Project/main/assets/banner.png" alt="Project Banner" width="800"/>
</div>

---

## 📋 Project Overview

This project implements a comprehensive **Predictive Maintenance System** using Machine Learning to predict equipment failures before they occur. The system analyzes sensor data and operational parameters to classify different types of machine failures, enabling proactive maintenance and reducing downtime.

## 🎯 Key Features

- ✅ **7 Optimized ML Models** with hyperparameter tuning
- ✅ **Interactive Streamlit Web Application** for real-time predictions
- ✅ **Complete Jupyter Notebook** with exploratory data analysis
- ✅ **Class Imbalance Handling** using SMOTE
- ✅ **Cross-Validation** for robust model evaluation
- ✅ **Feature Scaling** with StandardScaler
- ✅ **Model Persistence** for deployment

---

## 📁 Project Structure

```
Predictive-Maintenance-ML-Project/
│
├── app.py                          # Streamlit web application
├── Predictive maintainance.ipynb   # Complete analysis notebook
├── optimized_models_code.py        # Optimized ML model implementations
├── predictive_maintenance.csv      # Training dataset
├── requirements.txt                # Python dependencies
├── xgb_model_fold_4.joblib        # Trained XGBoost model
├── XG_boost.pkl                   # Alternative model file
├── .gitignore                     # Git ignore file
└── README.md                      # Project documentation
```

---

## 🤖 Machine Learning Models

### 1. **Logistic Regression**
- **Purpose**: Binary/multiclass classification baseline
- **Hyperparameters**: 
  - `max_iter=1000`
  - `class_weight='balanced'`
  - `solver='lbfgs'`
- **Use Case**: Simple, interpretable model for linear decision boundaries

### 2. **Naive Bayes (Gaussian)**
- **Purpose**: Probabilistic classifier based on Bayes' theorem
- **Hyperparameters**: 
  - `var_smoothing=1e-9`
- **Use Case**: Fast predictions, works well with high-dimensional data

### 3. **Support Vector Machine (SVM)**
- **Purpose**: Classification with optimal hyperplane separation
- **Hyperparameters**: 
  - `kernel='rbf'`
  - `C=10`
  - `gamma='scale'`
  - `class_weight='balanced'`
- **Use Case**: Non-linear classification with kernel trick

### 4. **K-Nearest Neighbors (KNN)**
- **Purpose**: Instance-based learning
- **Hyperparameters**: 
  - `n_neighbors=7`
  - `weights='distance'`
  - `metric='minkowski'`
  - `p=2`
- **Use Case**: Local pattern recognition

### 5. **Decision Tree Classifier**
- **Purpose**: Tree-based classification with interpretable rules
- **Hyperparameters**: 
  - `max_depth=10`
  - `min_samples_split=10`
  - `min_samples_leaf=5`
  - `class_weight='balanced'`
- **Use Case**: Non-linear relationships, feature importance

### 6. **Random Forest Classifier** ⭐
- **Purpose**: Ensemble of decision trees
- **Hyperparameters**: 
  - `n_estimators=200`
  - `max_depth=15`
  - `min_samples_split=5`
  - `min_samples_leaf=2`
  - `class_weight='balanced'`
  - `n_jobs=-1`
- **Use Case**: High accuracy, handles overfitting, feature importance

### 7. **XGBoost Classifier** ⭐ (Best Performance)
- **Purpose**: Gradient boosting for optimal predictions
- **Hyperparameters**: 
  - `n_estimators=200`
  - `learning_rate=0.1`
  - `max_depth=6`
  - `min_child_weight=3`
  - `subsample=0.8`
  - `colsample_bytree=0.8`
  - `eval_metric='logloss'`
- **Use Case**: Best overall performance, production deployment

---

## 📦 Modules and Libraries Used

### Core Data Science
- **pandas** (`2.2.3`): Data manipulation and analysis
- **numpy** (`2.2.5`): Numerical computing

### Machine Learning
- **scikit-learn** (`1.6.1`): ML models and utilities
  - `LogisticRegression`
  - `GaussianNB`
  - `SVC`
  - `KNeighborsClassifier`
  - `DecisionTreeClassifier`
  - `RandomForestClassifier`
  - `StandardScaler`
  - `train_test_split`
  - `cross_val_score`
- **xgboost** (`1.7.6`): Gradient boosting framework
- **imbalanced-learn** (`0.13.0`): SMOTE for class imbalance

### Visualization
- **matplotlib** (`3.10.0`): Plotting library
- **seaborn** (`0.13.2`): Statistical visualizations

### Model Persistence
- **joblib** (`1.4.2`): Save/load trained models

### Web Application
- **streamlit** (`1.45.0`): Interactive web app framework

### Jupyter Environment
- **jupyter** (`1.1.1`): Interactive notebooks
- **notebook** (`7.4.2`): Jupyter notebook server

---

## 🚀 Installation Guide

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (for cloning)

### Step 1: Clone the Repository

```bash
git clone https://github.com/MEKALA-JASWANTH/Predictive-Maintenance-ML-Project.git
cd Predictive-Maintenance-ML-Project
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all 22 required packages including:
- pandas, numpy
- scikit-learn, xgboost, imbalanced-learn
- matplotlib, seaborn
- streamlit, jupyter
- joblib

---

## 💻 How to Run the Project

### Option 1: Run Streamlit Web Application ⭐

The easiest way to use the predictive maintenance system:

```bash
python -m streamlit run app.py
```

**What happens:**
1. Streamlit server starts
2. Web browser opens automatically at `http://localhost:8501`
3. Interactive interface loads

**How to use the app:**
1. Enter machine parameters:
   - Air temperature [K]: 200-400
   - Process temperature [K]: 200-400
   - Rotational speed [rpm]: 0-5000
   - Torque [Nm]: 0-100
   - Tool wear [min]: 0-300
   - Type L: 0 or 1
   - Type M: 0 or 1

2. Click **"Predict"** button

3. View results:
   - Model prediction (Failure type or No failure)
   - Custom logic prediction
   - Confidence level

**To stop the app:** Press `Ctrl + C` in terminal

---

### Option 2: Run Jupyter Notebook

For complete analysis, training, and experimentation:

```bash
python -m notebook
```

**What happens:**
1. Jupyter server starts
2. Browser opens with file explorer
3. Click on `Predictive maintainance.ipynb`

**Notebook Contents:**
- Data loading and exploration
- Feature engineering
- Model training for all 7 models
- Hyperparameter optimization
- Performance comparison
- Visualizations
- Model saving

**How to execute:**
- Click **Cell → Run All** to execute entire notebook
- Or run cells individually with `Shift + Enter`

---

### Option 3: Run Python Script Directly

Execute the optimized models script:

```bash
python optimized_models_code.py
```

**What it does:**
- Loads the dataset
- Preprocesses features
- Trains all 7 models
- Applies SMOTE resampling
- Performs cross-validation
- Prints accuracy metrics
- Saves models

---

## 📊 Dataset Information

**File**: `predictive_maintenance.csv`

**Features:**
- `UDI`: Unique identifier
- `Product ID`: Product identification
- `Type`: Product quality variant (L, M, H)
- `Air temperature [K]`: Air temperature in Kelvin
- `Process temperature [K]`: Process temperature in Kelvin
- `Rotational speed [rpm]`: Rotational speed
- `Torque [Nm]`: Torque in Newton-meters
- `Tool wear [min]`: Tool wear in minutes

**Target Variables:**
- `Machine failure`: Binary (0=No failure, 1=Failure)
- `Failure Type`: Categories (No Failure, Heat Dissipation, Power Failure, Overstrain Failure, Tool Wear Failure)

**Dataset Size:** 10,000 samples

---

## 🔬 Model Training Pipeline

```
1. Data Loading
   ↓
2. Exploratory Data Analysis
   ↓
3. Feature Engineering
   - One-hot encoding for categorical variables
   - Feature selection
   ↓
4. Train-Test Split (80-20)
   ↓
5. Feature Scaling (StandardScaler)
   ↓
6. Handle Class Imbalance (SMOTE)
   ↓
7. Model Training
   - 7 different algorithms
   - Optimized hyperparameters
   ↓
8. Cross-Validation (5-fold)
   ↓
9. Model Evaluation
   - Accuracy
   - Precision
   - Recall
   - F1-Score
   ↓
10. Model Persistence (joblib)
```

---

## 📈 Model Performance

All models were trained with:
- ✅ StandardScaler for feature normalization
- ✅ SMOTE for handling class imbalance
- ✅ 5-fold cross-validation
- ✅ Optimized hyperparameters

**Top Performers:**
1. **XGBoost** - Best overall accuracy and robustness
2. **Random Forest** - Excellent feature importance insights
3. **SVM** - Strong with balanced dataset

---

## 🛠️ Troubleshooting

### Common Issues and Solutions

**Issue 1: `ModuleNotFoundError`**
```bash
# Solution: Install missing packages
pip install -r requirements.txt
```

**Issue 2: `streamlit: command not found`**
```bash
# Solution: Use python -m
python -m streamlit run app.py
```

**Issue 3: `jupyter: command not found`**
```bash
# Solution: Use python -m
python -m notebook
```

**Issue 4: XGBoost version compatibility**
```bash
# Solution: Install specific version
pip install xgboost==1.7.6
```

**Issue 5: Port already in use (Streamlit)**
```bash
# Solution: Use different port
streamlit run app.py --server.port 8502
```

**Issue 6: Large file size errors**
```bash
# Solution: Model files are tracked, data file is excluded
# Check .gitignore file
```

---

## 🎓 Learning Outcomes

This project demonstrates:
- ✅ Complete ML pipeline from data to deployment
- ✅ Multiple algorithm comparison
- ✅ Hyperparameter optimization
- ✅ Class imbalance handling
- ✅ Model persistence and deployment
- ✅ Interactive web application development
- ✅ Production-ready code structure

---

## 🔮 Future Enhancements

- [ ] Add more ensemble methods (Stacking, Voting)
- [ ] Implement deep learning models (LSTM, CNN)
- [ ] Add real-time monitoring dashboard
- [ ] Integrate with IoT sensors
- [ ] Deploy to cloud platform (AWS, Azure, Heroku)
- [ ] Add API endpoints (FastAPI/Flask)
- [ ] Implement A/B testing framework
- [ ] Add explainability (SHAP, LIME)

---

## 👤 Author

**MEKALA-JASWANTH**

- GitHub: [@MEKALA-JASWANTH](https://github.com/MEKALA-JASWANTH)
- Project: [Predictive-Maintenance-ML-Project](https://github.com/MEKALA-JASWANTH/Predictive-Maintenance-ML-Project)

---

## 📝 License

This project is open source and available under the MIT License.

---

## 🙏 Acknowledgments

- Dataset source: Industrial predictive maintenance data
- Inspired by real-world manufacturing challenges
- Built with industry-standard ML practices

---

## 📞 Support

If you encounter any issues or have questions:
1. Check the [Troubleshooting](#-troubleshooting) section
2. Review the [Issues](https://github.com/MEKALA-JASWANTH/Predictive-Maintenance-ML-Project/issues) page
3. Create a new issue with detailed description

---

## 📸 Screenshots


### Streamlit Web Application

<div align="center">
  <img width="2880" height="1800" alt="image" src="https://github.com/user-attachments/assets/9dc3dd48-107f-43a7-9610-0e0e62959b97" />
  <p><i>Interactive Streamlit web application for real-time predictions</i></p>
</div>

### Project Workflow

<div align="center">
  <img src="https://raw.githubusercontent.com/MEKALA-JASWANTH/Predictive-Maintenance-ML-Project/main/assets/workflow-diagram.png" alt="ML Pipeline Workflow" width="700"/>
  <p><i>Complete machine learning pipeline from data to deployment</i></p>
</div>

---

## ⭐ Show Your Support

If you find this project helpful, please give it a ⭐ star on GitHub!

---

**Last Updated**: November 2025
**Version**: 1.0.0
