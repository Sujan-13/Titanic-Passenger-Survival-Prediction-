# Titanic Dataset Analysis & Predictive Modeling

A comprehensive machine learning project analyzing survival patterns in the Titanic disaster through exploratory data analysis, feature engineering, and XGBoost classification with SHAP-based interpretability.

## Overview

This notebook performs an end-to-end analysis of the Titanic dataset, investigating how factors like age, gender, class, and fare influenced passenger survival. The project combines statistical analysis with predictive modeling to uncover survival patterns and provide model interpretability.

## Key Sections

### 1. **Data Exploration & Cleaning**
- Dataset overview (shape, data types, descriptive statistics)
- Missing value analysis and handling strategy
- Duplicate detection
- Outlier identification and removal

### 2. **Univariate & Bivariate Analysis**
Detailed investigation of survival relationships:
- **Age**: Distribution analysis with KDE; age-survival correlations
- **Sex**: Gender-based survival rates
- **Passenger Class**: Class stratification and survival outcomes
- **Fare**: Economic analysis and fare-survival patterns
- **Cabin & Deck**: Feature extraction from cabin codes
- **Embarkation Point**: Port-based survival variations
- **Family Metrics**: Siblings/spouses, parents/children, and aggregate family size

### 3. **Feature Engineering**
- **Deck Extraction**: Parsed cabin letters to determine passenger deck location
- **Family Size**: Aggregated relatives into meaningful family size categories
- **Age Imputation**: Filled missing age values using distribution patterns
- **Encoding**: Converted categorical variables for model consumption
- **Transformations**: Applied log transformations for skewed distributions

### 4. **Data Preprocessing**
- **Outlier Detection**: Identified and handled extreme values
- **Scaling**: Applied StandardScaler for normalization
- **Log Transformation**: Addressed skewness in fare and other features

### 5. **Predictive Modeling**
- **Model**: XGBoost Classifier
- **Metrics**: Classification reports, confusion matrices
- **Hyperparameter tuning**: Optimized for balanced precision-recall

### 6. **Model Interpretability**
- **SHAP Analysis**: Explainable AI to understand feature contributions
- **Feature Importance**: Ranked features by impact on predictions
- **Individual Predictions**: SHAP values for per-sample interpretability

## Technologies & Libraries

| Category | Tools |
|----------|-------|
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **ML & Preprocessing** | Scikit-learn, XGBoost |
| **Interpretability** | SHAP |

## Key Findings

1. **Gender Bias**: Female passengers had significantly higher survival rates ("women and children first")
2. **Class Advantage**: 1st class passengers experienced vastly better survival odds than 3rd class
3. **Economic Factor**: Higher fares correlated with improved survival rates
4. **Age Dependency**: Younger passengers, especially children, had higher survival likelihood
5. **Family Dynamics**: Traveling with family had mixed effects on individual survival

## Project Structure

```
.
├── main.ipynb           # Complete analysis notebook
└── Dataset/
    └── Titanic-Dataset.csv  # Input dataset
```

## Getting Started

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn xgboost shap
```

### Running the Analysis
```bash
jupyter notebook main.ipynb
```

## Usage

The notebook is structured to be read sequentially. Each section builds on previous analyses:
1. Start with data exploration to understand the dataset
2. Review univariate/bivariate analyses for feature relationships
3. Examine feature engineering decisions
4. Study the XGBoost model and SHAP interpretability results

## Interpretation Guide

- **SHAP Summary Plot**: Shows feature importance and their directional impact
- **SHAP Force Plot**: Explains individual predictions through additive feature contributions
- **Confusion Matrix**: Evaluates classification performance
- **Classification Report**: Precision, recall, and F1-scores per class

## Insights & Implications

This analysis demonstrates how historical biases and systemic inequalities manifested in real survival outcomes. The dataset serves as a powerful case study for:
- Understanding how protected characteristics (gender, class) influenced life-or-death outcomes
- The importance of ethical AI and model transparency
- The necessity of explainable models (SHAP) in high-stakes prediction tasks

## Author
**Sujan Gupta** | [GitHub](https://github.com/Sujan-13) | [LinkedIn](https://linkedin.com/in/sujan-g-545aa7187)

## License
This project is open source and available under the MIT License.

## Acknowledgments
- Titanic dataset from Kaggle
- SHAP library for model interpretability
- XGBoost team for the robust gradient boosting implementation

---

**Note**: This analysis is for educational and research purposes. All results reflect patterns in historical data and should not be generalized beyond the Titanic context without careful consideration.
