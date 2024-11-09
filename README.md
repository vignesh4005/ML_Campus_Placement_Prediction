# Campus Placement Prediction - Machine Learning Classifier Models

This project aims to predict a student's placement status based on academic and extracurricular factors such as CGPA, internship experience, and project involvement. Three machine learning classifiers were built and evaluated: Logistic Regression, Decision Tree, and K-Nearest Neighbors (KNN). Each model was tuned and assessed for accuracy to determine which best predicts placement outcomes.

## Dataset
The dataset used in this analysis can be found [here](https://raw.githubusercontent.com/ArchanaInsights/Datasets/main/campus_placement.csv).

### Dataset Attributes
- CGPA
- Internship Experience
- Project Involvement
- Other relevant academic and extracurricular features
- Placement Status (target variable)

## Project Steps

### 1. Data Preprocessing
- **Load Dataset**: Imported and explored the dataset to understand its structure.
- **Handle Missing Values**: Checked and handled any missing data.
- **Encode Categorical Features**: Categorical data was converted into numerical format using appropriate encoding.
- **Feature Selection**: Selected relevant features for modeling.
- **Data Splitting**: Split the dataset into training (80%) and testing sets (20%).
- **Feature Scaling**: Standardized numerical features to optimize model performance.

### 2. Model Building and Evaluation

#### Logistic Regression
- **Parameter Tuning**: Tested different values of `max_iter` (100, 200, 300, 400, 500).
- **Best Accuracy Score**: 79.8% for all max_iter values, indicating stable performance.

#### Decision Tree
- **Parameter Tuning**: Tested different values of `max_depth` (1-10).
- **Best Accuracy Score**: 77.6% with `max_depth = 3`.

#### K-Nearest Neighbors (KNN)
- **Parameter Tuning**: Tested different values of `k` (5-20).
- **Best Accuracy Score**: 79.1% at `k = 15`, which balanced accuracy and stability.

### 3. Model Comparison and Analysis
| Model               | Best Parameter   | Accuracy Score |
|---------------------|------------------|----------------|
| Logistic Regression | max_iter = 100   | 79.8%         |
| Decision Tree       | max_depth = 3    | 77.6%         |
| KNN                 | k = 15           | 79.1%         |

### 4. Evaluation Summary
- **Logistic Regression**: Best-performing model with an accuracy of 79.8%. It offers stability and interpretability, although it may miss non-linear relationships.
- **Decision Tree**: Scored 77.6% and shows signs of overfitting with increasing depth beyond `max_depth = 4`.
- **KNN**: Competitive with an accuracy of 79.1%, though computationally intensive and sensitive to k-values. Recommended `k = 15` for the best balance between accuracy and stability.

### 5. Conclusion
Logistic Regression proved to be the most effective model, achieving the highest accuracy and offering a consistent performance across different iterations. KNN also showed strong performance, though it required careful tuning of `k`. Decision Tree, despite capturing non-linear relationships, performed the least effectively and exhibited risks of overfitting.

## Visualization
Line charts for model accuracy scores across parameter values are provided in the notebook to visually assess the performance of each model.

## Dependencies
- Python
- pandas
- seaborn
- matplotlib
- scikit-learn
