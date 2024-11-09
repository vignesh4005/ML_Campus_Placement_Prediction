# Campus Placement Prediction - Machine Learning Classifier Models

This project aims to predict a student's placement status based on academic and extracurricular factors such as CGPA, internship experience, and project involvement. Three machine learning classifiers were built and evaluated: Logistic Regression, Decision Tree, and K-Nearest Neighbors (KNN). Each model was tuned and assessed for accuracy to determine which best predicts placement outcomes.

## Dataset
The dataset used in this analysis can be found [here](https://raw.githubusercontent.com/ArchanaInsights/Datasets/main/campus_placement.csv).

### Dataset Attributes
- **Academic Performance**: CGPA, SSC Marks, HSC Marks
- **Skills Ratings**: Aptitude Test Score, Soft Skills Rating
- **Extracurricular Factors**: Internship Experience, Project Involvement, Placement Training
- **Placement Status**: Target variable indicating whether a student is placed or not

## Exploratory Data Analysis (EDA)
Exploratory Data Analysis was conducted using Python libraries such as pandas, seaborn, and matplotlib. Key insights include:

- **Placement Analysis**: Out of 10,000 students, 40% are placed while 60% are not, showing a significant proportion of unplaced students.
  
- **Box Plot Analysis**: Box plots for marks-related attributes (CGPA, Aptitude Test Score, Soft Skills Rating, SSC Marks, HSC Marks) indicate no significant outliers, suggesting a well-distributed and reliable dataset for prediction.

- **Scatter Plot Analysis**: Scatter plots reveal that high CGPA, SSC, and HSC marks alone do not guarantee placement, implying that placement outcomes are influenced by factors beyond academic scores.

- **Chi-Square Hypothesis Testing**: The Chi-Square test results indicate a statistically significant difference in placement rates between students who received placement training and those who did not.

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
  ![image](https://github.com/user-attachments/assets/6ae8b416-cff4-426a-ade2-f65ff383b218)


#### Decision Tree
- **Parameter Tuning**: Tested different values of `max_depth` (1-10).
- **Best Accuracy Score**: 77.6% with `max_depth = 3`.

  ![image](https://github.com/user-attachments/assets/ddf8057b-b227-4cef-8ba7-76577f3a01a7)


#### K-Nearest Neighbors (KNN)
- **Parameter Tuning**: Tested different values of `k` (5-20).
- **Best Accuracy Score**: 79.1% at `k = 15`, which balanced accuracy and stability.
  ![image](https://github.com/user-attachments/assets/31394883-018c-4e8e-b208-1c3b5eab934c)


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

## Dependencies
- Python
- Pandas
- Seaborn
- Matplotlib
- SciPy
- Scikit-Learn
