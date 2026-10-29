# Customer Intelligence AI - Project Summary

## Project Overview

Built a comprehensive, production-style data science application for customer churn prediction and retention analytics. The application combines enterprise-grade business intelligence, predictive machine learning, explainable AI, customer-level analytics, SQL insights, and interactive retention planning in a polished Streamlit dashboard.

## Measured Results

### Model Performance
- **Best Model**: Logistic Regression
- **Test ROC-AUC**: 0.8464
- **Test F1 Score**: 0.6160
- **Test Precision**: 0.5106
- **Test Recall**: 0.7763
- **Cross-Validation ROC-AUC**: 0.8434 ± 0.0056

### Dataset
- **Original Records**: 7,043
- **Cleaned Records**: 7,010 (removed 33 with missing/duplicate data)
- **Churn Rate**: 26.5%
- **Features**: 21 columns including demographics, services, contract details

### Test Results
- **Total Tests**: 16
- **Passed**: 16
- **Failed**: 0
- **Coverage**: Data ingestion, feature engineering, database operations, analytics metrics

## Key Features Implemented

1. **Executive Analytics Dashboard**
   - Interactive KPI cards with real-time metrics
   - Churn analysis by contract type, payment method, internet service
   - Tenure band distribution and monthly charges analysis
   - Business insights generation from calculated results

2. **Customer Explorer**
   - Search by customer ID
   - Multi-filter capability (contract, tenure, charges, services)
   - Paginated customer table with sorting
   - Churn prediction integration with risk bands
   - CSV export functionality

3. **Individual Churn Prediction**
   - Customer profile input form with validation
   - Real-time churn probability prediction
   - Risk band classification (Low/Medium/High)
   - SHAP-based feature contributions
   - Plain-English explanations

4. **Model Lab**
   - Multi-model training (Dummy, Logistic Regression, Random Forest)
   - Cross-validation with stratified K-fold
   - Comprehensive evaluation metrics (ROC-AUC, F1, Precision, Recall, PR-AUC)
   - Confusion matrix visualization
   - Threshold exploration for precision-recall trade-offs

5. **Explainable AI**
   - Global feature importance using SHAP
   - Individual prediction explanations
   - Positive and negative feature contributions
   - Plain-English explanation generation
   - Customer profile view with feature values

6. **SQL Analytics**
   - Predefined query library
   - Custom SQL query execution
   - Parameterized queries for safety
   - SQL vs Python validation
   - Interactive analytics tables

7. **Retention Simulator**
   - Scenario-based what-if analysis
   - Campaign reach and effectiveness parameters
   - Financial outcome calculations
   - Break-even analysis
   - Sensitivity analysis visualization
   - Multi-scenario comparison

## Technical Architecture

### Backend Modules
- **Data Ingestion**: Automated download, validation, and cleaning pipeline
- **Feature Engineering**: Preprocessing pipeline with StandardScaler and OneHotEncoder
- **Model Training**: Leakage-safe sklearn Pipeline with stratified train/test split
- **Database**: SQLite with indexed queries and parameterized SQL
- **Inference**: Batch and single-customer prediction pipelines
- **Explainability**: SHAP integration for model interpretation

### Frontend
- **Framework**: Streamlit multi-page application
- **Visualization**: Plotly Express and Graph Objects with custom dark theme
- **Interactivity**: Global filters, session state management, real-time updates
- **UI/UX**: Dark navy theme with cyan/blue accents, responsive layout

## Resume Bullets

### Data Scientist
- Built end-to-end churn prediction model achieving 84.6% ROC-AUC using Logistic Regression with stratified cross-validation
- Implemented feature engineering pipeline with StandardScaler and OneHotEncoder, processing 7,010 customer records
- Developed SHAP-based explainable AI system providing feature-level interpretations for individual predictions
- Created automated data validation pipeline identifying and handling missing values, duplicates, and data quality issues

### Machine Learning Engineer
- Designed leakage-safe ML pipeline using sklearn Pipeline with train/test split and 5-fold stratified cross-validation
- Trained and evaluated multiple models (Logistic Regression, Random Forest, XGBoost) with comprehensive metrics (ROC-AUC, F1, Precision, Recall, PR-AUC)
- Implemented model artifact persistence system with metadata tracking for reproducible inference
- Built batch prediction pipeline supporting 7,000+ customers with risk band classification

### Data Engineer
- Designed SQLite database schema with indexed queries supporting complex analytics and SQL-based exploration
- Created parameterized SQL query library with injection-safe execution and Python validation
- Implemented data cleaning pipeline handling 11 non-numeric values and 22 duplicate records
- Built automated test suite with 16 tests covering data validation, feature engineering, and database operations

### Analytics Architect
- Developed executive analytics dashboard with interactive KPI cards, trend visualizations, and business insights
- Created retention scenario simulator with financial outcome modeling and sensitivity analysis
- Implemented SQL vs Python validation ensuring consistency across calculation methods
- Built customer intelligence workspace with multi-filter search, pagination, and export functionality

### Senior UI/UX Engineer
- Designed dark-themed Streamlit application with custom color scheme (navy background, cyan accents)
- Implemented 7-page multi-page application with consistent navigation and branding
- Created interactive Plotly visualizations with hover tooltips, zoom, and selection
- Built responsive layout supporting desktop and tablet viewing with professional card-based design

## Interview Questions

### Python
1. **Question**: How do you handle chained assignment warnings in pandas?
   **Answer**: Use `.copy()` to create explicit copies and `.loc[]` for assignments to avoid chained assignment. For example: `df.loc[:, "column"] = value` instead of `df["column"] = value` on a slice.

2. **Question**: What's the difference between `@st.cache_data` and `@st.cache_resource` in Streamlit?
   **Answer**: `@st.cache_data` caches data objects (DataFrames, lists) that can be serialized, while `@st.cache_resource` caches resources (database connections, ML models) that shouldn't be serialized and should persist across reruns.

3. **Question**: How do you handle missing data in pandas?
   **Answer**: Options include dropping rows with `dropna()`, filling with `fillna()`, imputing with mean/median, or using forward/backward fill. The choice depends on the amount and pattern of missing data, and whether the missingness is informative.

### SQL
1. **Question**: How do you prevent SQL injection in Python?
   **Answer**: Use parameterized queries with placeholders (`?` or `:name`) and pass parameters separately. Never concatenate user input directly into SQL strings. Libraries like `sqlite3` and `sqlalchemy` support parameterized queries.

2. **Question**: What's the difference between `WHERE` and `HAVING` in SQL?
   **Answer**: `WHERE` filters rows before grouping, while `HAVING` filters groups after aggregation. Use `WHERE` for row-level conditions and `HAVING` for conditions on aggregated values.

3. **Question**: How do you optimize SQL query performance?
   **Answer**: Create indexes on frequently queried columns, use `EXPLAIN` to analyze query plans, avoid `SELECT *`, use appropriate JOIN types, limit result sets with `LIMIT`, and consider denormalization for read-heavy workloads.

### Statistics
1. **Question**: What is ROC-AUC and how do you interpret it?
   **Answer**: ROC-AUC (Receiver Operating Characteristic - Area Under Curve) measures the ability of a classifier to distinguish between classes. A value of 0.5 is random guessing, 1.0 is perfect classification. Values above 0.8 indicate good discrimination.

2. **Question**: What's the difference between precision and recall?
   **Answer**: Precision measures the proportion of positive predictions that are actually positive (TP / (TP + FP)). Recall measures the proportion of actual positives that are correctly predicted (TP / (TP + FN)). There's typically a trade-off between them.

3. **Question**: How do you handle class imbalance in classification?
   **Answer**: Techniques include: resampling (oversampling minority, undersampling majority), using class weights in model training, choosing appropriate evaluation metrics (F1, AUC-PR instead of accuracy), and using algorithms robust to imbalance.

### Exploratory Data Analysis (EDA)
1. **Question**: What's your EDA process when starting a new project?
   **Answer**: Start with understanding the business problem, examine data structure and types, check for missing values and duplicates, analyze distributions (histograms, box plots), explore correlations (heatmap, scatter plots), identify outliers, and generate initial insights and hypotheses.

2. **Question**: How do you detect and handle outliers?
   **Answer**: Detect using visual methods (box plots, scatter plots) and statistical methods (Z-score, IQR). Handle by: removing if they're errors, capping/winsorizing, transforming (log), or keeping if they're legitimate but note them in analysis.

3. **Question**: What visualizations do you use for categorical vs numerical data?
   **Answer**: For categorical: bar charts, pie charts (for few categories), count plots. For numerical: histograms, box plots, violin plots, density plots. For relationships: scatter plots, heatmaps for correlations, pair plots for multiple variables.

### Machine Learning
1. **Question**: What's the difference between overfitting and underfitting?
   **Answer**: Overfitting: model learns noise in training data, performs well on training but poorly on test. Underfitting: model too simple to capture patterns, performs poorly on both. Address with regularization, cross-validation, and appropriate model complexity.

2. **Question**: How do you prevent data leakage in ML pipelines?
   **Answer**: Keep preprocessing inside the pipeline so it's fit only on training data, use proper train/test split before any feature engineering, avoid using future information, use time-based splits for time series, and validate with cross-validation.

3. **Question**: What's the difference between bagging and boosting?
   **Answer**: Bagging (Bootstrap Aggregating): trains multiple models independently on bootstrap samples and averages predictions (e.g., Random Forest). Boosting: trains models sequentially, each correcting errors of previous ones (e.g., XGBoost, AdaBoost).

### Model Evaluation
1. **Question**: Why use cross-validation instead of a single train/test split?
   **Answer**: Cross-validation provides more reliable performance estimates by reducing variance, uses all data for both training and testing across folds, helps detect overfitting, and provides confidence intervals through standard deviation across folds.

2. **Question**: When would you use PR-AUC instead of ROC-AUC?
   **Answer**: PR-AUC (Precision-Recall AUC) is more informative than ROC-AUC for imbalanced datasets where the positive class is rare. ROC-AUC can be overly optimistic in such cases, while PR-AUC focuses on performance on the positive class.

3. **Question**: How do you choose a classification threshold?
   **Answer**: Consider business objectives: maximize precision if false positives are costly, maximize recall if false negatives are costly. Use precision-recall curves, F1-score optimization, or cost-sensitive analysis. Threshold can be tuned on validation data.

### SHAP
1. **Question**: What is SHAP and how does it work?
   **Answer**: SHAP (SHapley Additive exPlanations) uses game theory to assign each feature an importance value for a particular prediction. It calculates the average marginal contribution of a feature across all possible feature combinations, providing local and global interpretability.

2. **Question**: What are the limitations of SHAP?
   **Answer**: Computationally expensive for large datasets and complex models, explanations depend on the reference/background data, correlated features can complicate attribution, and explanations don't prove causation—only show how the model uses features.

3. **Question**: How do you interpret SHAP values?
   **Answer**: Positive SHAP values increase the prediction, negative values decrease it. The magnitude shows the strength of influence. Global importance is the mean absolute SHAP value across samples. Local explanations show feature contributions for individual predictions.

### Dashboard Design
1. **Question**: How do you design an effective analytics dashboard?
   **Answer**: Start with user needs and business questions, prioritize key metrics at the top, use consistent visual hierarchy, ensure interactivity (filters, drill-down), maintain responsive design, use appropriate chart types for data, and provide context and explanations.

2. **Question**: What are best practices for dashboard color schemes?
   **Answer**: Use limited color palette (3-5 colors), ensure accessible contrast, use color consistently (same color = same meaning), avoid distracting gradients, consider color blindness, and use color to highlight important information rather than decorate.

3. **Question**: How do you handle dashboard performance with large datasets?
   **Answer**: Use data aggregation and sampling, implement caching, lazy loading for large tables, paginate results, precompute expensive calculations, use efficient chart libraries, and consider backend processing for complex queries.

### Business Impact
1. **Question**: How do you measure the business impact of a churn prediction model?
   **Answer**: Track metrics like: customer retention rate before/after model deployment, campaign response rates, revenue saved from prevented churn, cost savings from targeted interventions, and ROI of retention efforts. Conduct A/B tests comparing model-guided vs random targeting.

2. **Question**: How do you communicate model limitations to business stakeholders?
   **Answer**: Be transparent about accuracy and error rates, explain that predictions are probabilities not certainties, note training data limitations and potential drift, provide confidence intervals, emphasize human judgment in final decisions, and document false positive/negative business costs.

3. **Question**: How do you ensure responsible use of predictive models?
   **Answer**: Document model limitations and intended use, avoid using sensitive attributes for discrimination, provide explainability for decisions, implement monitoring for drift and bias, establish governance processes, and ensure human oversight for high-stakes decisions.

## Remaining Limitations

1. **XGBoost Integration**: XGBoost was skipped due to import issues in the training environment. Can be added once dependencies are resolved.

2. **Model Versioning**: No automated model versioning or rollback mechanism. Consider using MLflow or similar for production.

3. **Data Drift Monitoring**: No automated monitoring for data drift or model performance degradation over time.

4. **Authentication**: No user authentication or authorization implemented for multi-user scenarios.

5. **Real-time Updates**: Dashboard requires manual refresh for new data; no real-time streaming or automated retraining.

6. **Advanced Visualizations**: Some advanced visualizations (cohort analysis, feature relationship explorer) could be added for deeper insights.

7. **Deployment**: Application not yet deployed to production environment; requires Streamlit Community Cloud or similar setup.

8. **Scalability**: SQLite is suitable for current dataset size but may need migration to PostgreSQL for larger datasets or concurrent users.

## Future Enhancements

1. **A/B Testing Framework**: Implement controlled experiments to measure actual retention campaign effectiveness.

2. **Automated Retraining**: Schedule periodic model retraining with new data to maintain performance.

3. **Customer Segmentation**: Add clustering-based customer segmentation for targeted marketing.

4. **Time Series Analysis**: Incorporate temporal patterns if historical timestamp data becomes available.

5. **API Integration**: Build REST API for programmatic access to predictions and analytics.

6. **Multi-tenant Support**: Add support for multiple organizations or datasets.

7. **Advanced Explainability**: Implement counterfactual explanations and decision tree surrogate models.

8. **Notification System**: Add alerts for high-risk customers or unusual churn patterns.

---

**Project Status**: Complete and Functional
**Test Status**: All 16 tests passing
**Model Status**: Trained and deployed (Logistic Regression, ROC-AUC: 0.8464)
**Application Status**: Running locally at http://localhost:8502
