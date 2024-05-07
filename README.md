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
