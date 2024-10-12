# Predicting Life Expectancy

Welcome to **Predicting Life Expectancy**, a data science project focused on predicting life expectancy based on various socio-economic, health, and demographic factors. This repository contains all the materials and instructions needed to understand, recreate, and expand upon the analysis conducted as part of this project. The project leverages Python, Jupyter notebooks, and machine learning techniques to build a robust predictive model.

## Project Overview

This project aims to predict the life expectancy of individuals in different countries based on various features such as health expenditure, education levels, vaccination coverage, and economic status. The dataset provided contains 24 features for over 2000 samples, representing both training and test datasets, where the target variable to predict is "TARGET_LifeExpectancy." We have implemented data cleaning, exploration, modeling, and prediction techniques to achieve reliable results.

## Repository Structure

- **`s3970066.ipynb`**: Jupyter notebook containing all the data exploration, model training, evaluation, and prediction steps.
- **`train.csv`**: Training dataset with 24 columns (23 features + target variable).
- **`test.csv`**: Test dataset with 23 columns, lacking the target variable "TARGET_LifeExpectancy".
- **`s3970066.csv`**: The prediction file containing life expectancy values for the test dataset.
- **`s3970066.pdf`**: Report detailing the approach, methodology, model selection, evaluation, and results.

## Dataset Description

The dataset consists of various socio-economic, health, and demographic indicators for different countries over several years. Key features include:

- **Country**: Identifier for the country.
- **Year**: Year of data collection.
- **Status**: Developed or developing status (binary indicator).
- **AdultMortality**: Adult mortality rates for both sexes.
- **IncomeCompositionOfResources**: Human Development Index in terms of income composition of resources.
- **Schooling**: Average number of years of schooling.

The target variable is **TARGET_LifeExpectancy**, which represents the predicted life expectancy in years. The dataset includes 24 features for over 2000 samples.

## Key Steps and Methods

### 1. Data Preprocessing

- **Missing Value Imputation**: Missing values in important features like "PercentageExpenditure," "IncomeCompositionOfResources," "GDP," and "Schooling" were replaced using linear regression. This imputation was performed based on trends over time for each country. For example, missing values in schooling were predicted based on known values for that country using linear regression, assuming a steady trend. Linear interpolation was also employed for features with consistent yearly trends, ensuring the continuity of data points where possible.
  
- **Outlier Handling**: Outliers were categorized into three types based on their z-scores: mild (z-score between 3 and 3.5), moderate (z-score between 3.5 and 4), and extreme (z-score > 4). Extreme outliers were handled using Winsorization, where values were capped at a specific threshold to mitigate their impact on the model without distorting the underlying data distribution. Outliers in variables like "AdultMortality" and "Alcohol" were analyzed closely, as they often represented real-world anomalies rather than errors.

- **Feature Scaling**: All numerical features were standardized using z-score normalization to ensure consistent scaling, which is crucial for improving model performance and handling bias, especially for linear models. Features such as "GDP" and "Population" had particularly wide ranges, which required normalization to prevent any feature from dominating the model.

### 2. Exploratory Data Analysis (EDA)

- **Histograms and Distribution Analysis**: Generated histograms for all numerical features to visualize their distributions in both training and test datasets. The analysis revealed that most variables had similar distributions across the two datasets, which indicated a balanced dataset, helping ensure that the model trained effectively. The histograms also helped identify skewness in features like "HIV-AIDS" and "Measles," indicating potential areas for data transformation.

- **Correlation Analysis**: Conducted correlation analysis to identify relationships between features:
  ![image](https://github.com/user-attachments/assets/f7cce919-ea7b-4226-a127-4153ff1ea1c8)

  - **Strong Positive Correlations**: Life expectancy showed strong positive correlations with "IncomeCompositionOfResources" (0.798) and "Schooling" (0.716). This suggests that higher income levels and better education are significantly associated with longer life expectancy. The correlation matrix heatmap visualized these relationships, helping to identify which features could be most impactful for predicting life expectancy.
  - **Strong Negative Correlations**: "AdultMortality" was negatively correlated with life expectancy (-0.66), indicating that higher adult mortality rates were associated with lower life expectancy. Similarly, "HIV-AIDS" prevalence negatively impacted life expectancy (-0.522). Scatter plots between these features and life expectancy were used to visually confirm these relationships.

- **Pair Plot Analysis**: A pair plot was generated for a subset of the most relevant features, including "Schooling," "IncomeCompositionOfResources," and "AdultMortality." This visual analysis provided insights into the pairwise relationships between features, indicating potential multicollinearity that could affect model performance.

- **Box Plots for Categorical Features**: Box plots were used to analyze the distribution of life expectancy across the "Status" feature (developed vs. developing countries). It was observed that developed countries generally had higher life expectancy, with fewer outliers compared to developing countries, which had a wider range of life expectancy values.

### 3. Model Development and Evaluation

- **Model Selection**: Evaluated several machine learning models including **Linear Regression**, **Ridge Regression**, **Lasso Regression**, and **Random Forest Regression**.
  - **Initial Evaluation**: All models were evaluated using metrics such as **RMSE** (Root Mean Square Error), **MAE** (Mean Absolute Error), **R²** (Coefficient of Determination), and a custom accuracy metric that measured predictions within 10% of the actual values. The models were compared using bar plots to visualize their performance across these metrics.
  - **Regularization Techniques**: **Ridge Regression** and **Lasso Regression** were selected for further analysis due to their ability to handle multicollinearity and overfitting. Ridge Regression was particularly effective in balancing complexity and model performance. **Lasso Regression** was also analyzed for its feature selection capabilities, as it can shrink some feature coefficients to zero, effectively removing them from the model.

- **Cross-Validation**: A 5-fold cross-validation was used to ensure robust evaluation and to avoid overfitting. This approach provided a reliable assessment of model performance on unseen data. Cross-validation results were plotted to compare model stability, showing that Ridge Regression had the least variance in performance across folds.

- **Hyperparameter Tuning**: The regularization parameter **'alpha'** for Ridge and Lasso Regression was tuned within the range [0.5, 1.0]. Ridge Regression with an alpha value of 1.0 provided the best results, offering a balance between regularization strength and predictive accuracy. Grid search was used to systematically explore different alpha values, and the results were plotted to visualize the impact of alpha on model performance metrics.

- **Feature Importance Analysis**: For Random Forest Regression, feature importances were calculated and plotted. Features like "IncomeCompositionOfResources" and "Schooling" were found to have the highest importance, reinforcing the findings from correlation analysis. This helped in understanding the contribution of each feature to the model's predictions.

### 4. Prediction and Results
![image](https://github.com/user-attachments/assets/05bd7a73-fa04-42a4-82de-b970f6422629)
![image](https://github.com/user-attachments/assets/ce9e727e-7a61-461a-b134-08ae536afbab)
![image](https://github.com/user-attachments/assets/947107c5-d728-4b4b-88ff-53d7fff60788)
![image](https://github.com/user-attachments/assets/4ffd781d-1b01-40cd-9f48-5de7db63f738)
![image](https://github.com/user-attachments/assets/2087f798-ad19-4bc7-bcf5-27ab6f523a1e)



- **Final Model**: The **Ridge Regression** model with alpha = 1.0 was selected as the final model. It was trained on the entire training dataset and used to generate predictions for the test dataset. The predictions were saved in `s3970066.csv`.

- **Performance Metrics**:
  - **RMSE** (Root Mean Square Error): Provided insight into the average prediction error, with lower values indicating better performance. The RMSE for the final model was 3.45 years.
  - **MAE** (Mean Absolute Error): Measured the average magnitude of the errors in predictions, with an MAE of 2.67 years, indicating the average difference between predicted and actual life expectancy.
  - **R² Score**: 0.82, indicating that the model explained 82% of the variance in life expectancy.

- **Residual Analysis**: Residual plots were used to assess normality and homoscedasticity, confirming that the model assumptions were reasonably met. The residuals were randomly scattered around zero, indicating that the model did not suffer from systematic bias.

### 5. Feature Engineering and Discussion

- **Feature Interaction Terms**: Interaction terms between "Schooling" and "GDP" were created to explore whether the combined effect of education and economic status had a significant impact on life expectancy. However, these interaction terms did not significantly improve model performance and were ultimately excluded from the final model.

- **Feature Importance**: The importance of features such as "IncomeCompositionOfResources" and "Schooling" was highlighted, as they showed strong correlations with the target variable. The Ridge model coefficients were analyzed to determine which features had the greatest influence on life expectancy predictions.

- **Model Limitations**: The assumption of linearity in certain features may not always hold, and biases in the dataset toward economically developed countries could impact generalizability. The dataset's imbalance, with more samples from developing countries, could lead to skewed predictions for countries with fewer data points.

## How to Recreate the Analysis

Follow these steps to recreate the analysis:

1. **Clone the Repository**
   ```bash
   git clone https://github.com/reyiyama/predicting_life_expectancy.git
   cd predicting_life_expectancy
   ```

2. **Install the Dependencies**
   Make sure you have Python 3.7+ installed. Then install the required libraries by running:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Jupyter Notebook**
   - Start Jupyter Notebook:
     ```bash
     jupyter notebook
     ```
   - Open `s3970066.ipynb` and run all the cells to perform data exploration, preprocessing, model training, and predictions. The notebook is well-documented with explanations for each step, including data cleaning, feature engineering, model evaluation, and interpretation of results.

4. **Generate Predictions**
   - Ensure both `train.csv` and `test.csv` are in the root directory.
   - Running the notebook will generate predictions that will be saved in `s3970066.csv`.

## Dependencies

- **Python 3.7+**
- **Jupyter Notebook**
- **Pandas**: For data manipulation
- **NumPy**: For numerical operations
- **Scikit-learn**: For machine learning models and evaluation metrics
- **Matplotlib & Seaborn**: For visualizations

## Results Summary

- The final model used was **Ridge Regression** with an alpha value of 1.0.
- **Key Metrics**:
  - **RMSE**: 3.45 years, indicating the average prediction error.
  - **MAE**: 2.67 years, measuring the average magnitude of errors in predictions.
  - **R² Score**: 0.82, showing that the model explains 82% of the variance in the data.

## Insights & Discussion

- **Key Correlations**: Life expectancy is highly correlated with income levels and education, reaffirming the importance of socioeconomic factors in determining health outcomes.
- **Limitations**: Assumptions of linearity might not hold for all features, and the dataset has biases toward economically developed countries.
- **Further Improvements**: Including non-linear models such as Gradient Boosting might capture more complex relationships between features.

## Future Work

- **Model Expansion**: Experiment with more advanced models like Gradient Boosting or Neural Networks to see if they improve accuracy.
- **Feature Engineering**: Create new features like interaction terms between health expenditure and schooling or non-linear transformations to capture complex relationships.
- **Addressing Dataset Imbalance**: Use techniques such as SMOTE (Synthetic Minority Over-sampling Technique) to address the imbalance between developed and developing countries.

## Contributing

Feel free to submit issues or pull requests. All kinds of contributions are welcome!

## License

This project is licensed under the MIT License.

---
Thank you for visiting this project. If you found it interesting, consider giving it a star!

## Author

**Amay Viswanathan Iyer**
