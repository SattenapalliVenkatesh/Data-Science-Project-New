
# Predicting Divorce Likelihood Using Socio-Behavioral Data

This project aims to predict the likelihood of divorce based on socio-behavioral factors, such as communication patterns, conflict resolution strategies, and relationship dynamics. The analysis uses machine learning techniques to predict divorce outcomes, compares them with traditional statistical approaches, and incorporates explainable AI methods like SHAP for model interpretability. Additionally, the project addresses the issue of data imbalance, which affects the trade-off between false positives and false negatives.

## **Research Questions**
1. How accurately can machine learning models predict divorce likelihood, and how do they compare to traditional statistical methods?
2. Which socio-behavioral attributes (e.g., communication patterns, conflict resolution) are most influential in predicting divorce, and how can SHAP analysis enhance model interpretability?
3. How can data imbalance be handled effectively in predictive models to balance false positives and false negatives, especially in divorce prediction?

## **Dataset Details**
- **Name:** Divorce Predictors Dataset
- **Source:** UCI Machine Learning Repository - Divorce Predictors Dataset
- **Contributors:** The dataset was created and published by Dr. S. Yöntem in 2020, based on a socio-behavioral questionnaire inspired by the Gottman Couples Therapy Model. It contains responses from married couples, each answering 54 questions related to communication, conflict resolution, and relationship dynamics. The data is labeled with divorce outcomes: "divorced" or "not divorced."

## **Libraries Used**
- **pandas** for data manipulation
- **matplotlib** and **seaborn** for data visualization
- **scikit-learn** for machine learning models and evaluation metrics
- **SHAP** for model interpretability and explainability

## **Steps Performed in the Notebook**
1. **Import Libraries:** Import necessary libraries for data manipulation, model building, and evaluation.
2. **Load Dataset:** Load the Divorce Predictors dataset and preview its structure.
3. **Data Preprocessing:**
   - Clean the data by renaming columns for clarity and adjusting for consistency.
   - Handle missing values if any, and perform feature engineering.
4. **Exploratory Data Analysis (EDA):** Analyze and visualize the dataset to uncover relationships between features and the target variable.
5. **Model Training and Evaluation:**
   - Train multiple machine learning models: Logistic Regression, Decision Tree, and KNN.
   - Evaluate each model’s performance using accuracy, precision, recall, and F1-score metrics.
6. **Hyperparameter Optimization:** Optimize model parameters for better performance using techniques like **Random Search** and **Stratified K-Fold cross-validation**.
7. **Explainable AI (SHAP Analysis):** Use SHAP values to understand model decisions and highlight key socio-behavioral factors contributing to the prediction of divorce.
8. **Model Evaluation Summary:** Compare the models based on performance metrics like **ROC AUC**, **accuracy**, **precision**, and **recall**.

## **How to Run the Notebook**
1. Clone the repository to your local machine.
2. Install the required libraries:
   ```bash
   pip install pandas seaborn matplotlib scikit-learn shap
   ```
3. Download the dataset from the **UCI Machine Learning Repository** or upload your own dataset structured similarly.
4. Open the notebook in **Jupyter Notebook** or **Google Colab**.
5. Execute all the cells to perform the analysis and model training.

## **Results**
The project provides a prediction model for divorce likelihood based on socio-behavioral data, offering insights into the factors influencing divorce outcomes. The models are evaluated using various metrics, and SHAP analysis enhances interpretability by identifying key features.

## **Contributions**
Feel free to contribute by:
- Improving model performance.
- Adding new machine learning algorithms.
- Enhancing data preprocessing and feature engineering steps.

## **License**
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
