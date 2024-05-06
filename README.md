# dsc232r_group_project
UCSD Master's of Data Science DSC232R Course - PySpark Data Analysis Group Project


The FEMA National Flood Insurance Policy database includes 45 feature columns and 50,406,943 observations of policy transactions collected from all states in the U.S. that are participating in the National Flood Insurance Program (NFIP). There are many features in the NFIP dataset with large quantities of missing values in its columns, such as base flood elevation measurements to flood obstruction types. Such features have approximately 30 to 50 million and greater missing values in their respective columns. Features with the least amount of missing values consist of surveying county codes and federal policy fees, each of which have 50,000 or lower missing values. 
Overall, while the dataset shows a strong presence of policy and basic property information, it suffers from a significant absence of detailed geographic and structural data. This gap in the data can hamper effective risk assessment and pricing of flood insurance policies, especially in areas prone to flooding where such data is most critical. Addressing these missing values, either by data imputation where appropriate or by collecting missing data, could significantly enhance the robustness of any analysis or predictive modeling based on this dataset.

The summary statistics for this dataset include measures of central tendency, dispersion, and range, all of which are critical for understanding the distribution and potential data quality issues within the dataset. The information provides a comprehensive overview of various insurance policy and property-related analyses and patterns.

-- Please refer to the [FEMA Data Exploration Jupyter Notebook](pyspark.ipynb) file to review data exploration on summary statistics and data distribution. --
