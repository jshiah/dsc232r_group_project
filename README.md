# dsc232r_group_project
UCSD Master's of Data Science DSC232R Course - PySpark Data Analysis Group Project


The FEMA National Flood Insurance Policy database includes 45 feature columns and 50,406,943 observations of policy transactions collected from all states in the U.S. that are participating in the National Flood Insurance Program (NFIP). There are many features in the NFIP dataset with large quantities of missing values in its columns, such as base flood elevation measurements to flood obstruction types. Such features have approximately 30 to 50 million and greater missing values in their respective columns. Features with the least amount of missing values consist of surveying county codes and federal policy fees, each of which have 50,000 or lower missing values. 
Overall, while the dataset shows a strong presence of policy and basic property information, it suffers from a significant absence of detailed geographic and structural data. This gap in the data can hamper effective risk assessment and pricing of flood insurance policies, especially in areas prone to flooding where such data is most critical. Addressing these missing values, either by data imputation where appropriate or by collecting missing data, could significantly enhance the robustness of any analysis or predictive modeling based on this dataset.

The summary statistics for this dataset include measures of central tendency, dispersion, and range, all of which are critical for understanding the distribution and potential data quality issues within the dataset. The information provides a comprehensive overview of various insurance policy and property-related analyses and patterns.

-- Please refer to the [FEMA Data Exploration Jupyter Notebook](pyspark.ipynb) to review data exploration on summary statistics and data distribution. --

As for data visualization, the preprocessing of the data introduced a few challenges. When attempting to drop all null values from the FEMA dataset, the dropna() method removed every observation in the entire data frame, rendering the dataset completely empty after the method was executed. In order to properly handle the null values, the decision was made to impute each null element with the mean of all values from the respective column. Imputing null values with the column values’ means resolved the issue of skewing data visualization due to null values. The second challenge of the data preprocessing step involved dealing with the dataset size being too large to perform data visualizations on. Using the entire dataset to visualize a general relationship between the variables was not possible, so the methods .sample() and .toPandas() were used to successfully visualize only a sample of the dataset. After testing out various percentage values in the Fraction parameter, 0.1 proved to be the only parameter value that would work properly on our dataset. Using the sample dataset, scatterplots were produced to show strong positive correlations. Examples are: between policy cost and total insurance premium of the policy, with a correlation of 0.9939. County code vs. census tract, with a strong positive correlation of 0.9861. Finally, the policy count and total building insurance coverage have a correlation of 0.9347, all of which may be found in the FEMA Jupyter Notebook’s Scatterplots section. 


----------------------------
To download the FEMA dataset, please refer to the link:
[FEMA National Flood Insurance Policy database - Kaggle Link](https://www.kaggle.com/datasets/lynma01/femas-national-flood-insurance-policy-database/data)

Environment Setup Instructions
1. Download the FEMA dataset (from link above) from the folder with the cloned GitHub repository.
2. Unzip the downloaded dataset.
3. Load the dataset into [pyspark.ipynb](pyspark.ipynb) and run the notebook.


## Building The First Linear Regression Model:
Our goal was to identify the top 10 features that can accurately represent the dataset as a whole. We chose 'totalinsurancepremium' as our target variable.

Next, we found linear regression coefficients for all 19 numerical features, and ranked them by their absolute value. We then collected the 10 features that had the coefficients with the highest absolute values.

## Building The Second Linear Regression Model Using Top 10 Features Identified By The First Model:

The top ten features that we ultimately chose were:
'Basementenclosurecrawlspacetype', 'crsdiscount', 'federalpolicyfee', 'hfiaasurcharge', 'latitude', 'numberoffloorsininsuredbuilding', 'occupancytype', 'policycost', 'policycount' and 'policytermindicator'. We used the RegressionEvaluator library to find RMSE and R² values for both the training and test set.

Next, we created bar graphs to compare RMSE and R² values for the training and test set. 

## Where does The Model Fit in the Fitting Graph?
Based on the comparison from above, we determined that our linear regression model is IDEAL (e.g., not overfitting or underfitting).

## RMSE and R² Values:
-RMSE for Training set: 118.133
-R² squared value for Training set: 0.995
-RMSE for Testing set: 117.118
-R² squared value for Testing set: 0.995

## What Are The Next Models You Are Thinking Of And Why?
Based on our initial findings, we are considering using the following models:

### 1) Logistic Regression
Reason: Logistic Regression is a straightforward and easily interpretable model, making it an ideal choice for binary classification. In our use case, this model will be used to predict whether an area is highly susceptible to flooding or not.
Application: The simplicity of Logistic Regression will allow us to investigate the relationship between the predictor variables and the likelihood of flooding, providing insights into key risk factors.

### 2) Decision Tree Classifier
Reason: The Decision Tree Classifier is well-suited for complex classification tasks. It excels at handling large datasets with numerous features and provides valuable insights into feature importance.
Application: This model will help us understand the complex interactions between various predictors and their impact on flood susceptibility. By identifying the most significant features, we can better target our risk mitigation efforts.

### 3) K-Means Clustering
Reason: K-Means Clustering can effectively group areas with similar flood risk profiles into distinct clusters. This method is particularly useful for exploratory data analysis and segmentation.
Application: By clustering areas based on their flood risk profiles, we can identify regions that share similar characteristics and tailor our risk management strategies accordingly. This clustering will also help in visualizing and understanding the geographic distribution of flood risk.

### 4) DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
Reason: DBSCAN is ideal for identifying dense regions of flood incidents and handling noise in the data. Unlike K-Means, DBSCAN does not require the number of clusters to be specified beforehand, and can find arbitrarily shaped clusters.
Application: This model is particularly useful for spatial data analysis, as it can reveal high-density flood zones and outliers.

## Conclusions:
In summary, the proposed models have been chosen for their specific strengths and suitability to different aspects of our flood risk analysis:
Logistic Regression: For its straightforward and interpretable approach to binary classification.
Decision Tree Classifier: For handling complex classification tasks and providing insights into feature importance.
K-Means Clustering: For grouping areas with similar flood risk profiles into distinct clusters.
DBSCAN: For identifying dense regions of flood incidents and handling noise in spatial data.

By leveraging these models, we aim to develop a comprehensive understanding of flood risks. 

## What is the conclusion of your 1st model? What can be done to possibly improve it?

Our second linear regression model built using the top 10 features identified from the initial model performs well, with very high R² values of 0.995 for the training and test sets. This means that the model explains 99.5% of the variance in 'totalinsurancepremium'. The RMSE values for both the training and test sets are very close (118.133 and 117.118), indicating that the model generalizes well and is not overfitting or underfitting.

A few strategies that could potentially be used to improve our model's performance include:

1) Using Cross-Validation to ensure that the model's performance is consistent across different subsets of the data. 

2) Checking for outliers that may be disproportionately influencing the model. Removing or transforming outliers (if they exist) could potentially improve our model's performance.

3) Ensuring that all selected features are properly scaled. Standardization of features might lead to improvement in our model's performance.

4) Rather than selecting the top 10 features based solely on magnitude of coefficients, we could potentially try different methods like forward selection, backward elimination, or recursive feature elimination.
