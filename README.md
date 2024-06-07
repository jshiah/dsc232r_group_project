# dsc232r_group_project
UCSD Master's of Data Science DSC232R Course - PySpark Data Analysis Group Project

## Introduction
Floods are a common threat to structures in and outside of the United States. Historically, flooding has occurred in all 50 states, and multiple factors such as inclement weather patterns and geography can influence flood risk. (reference 1).
Physical damage to structures and properties can be covered by insurance policies issued by the National Flood Insurance Program (NFIP) in the event of severe flooding. (reference 2).
Flood insurance prices estimated by the Federal Emergency Management Agency (FEMA) depend on many factors such as levels of rainfall, costs of rebuilding structures, and distance from the ocean. (reference 3). For example, premiums are expected to increase by 342% in Palm Beach County, FL- specifically in the Tequesta and Jupiter areas. This neighborhood is already the most expensive in South Florida with flood insurance premiums in excess of $7,000 dollars per year. There is an ever-increasing amount of flood risk to regions like Florida as climate change causes rising sea levels and rain bomb situations which are currently overwhelming the Fort Lauderdale area. In addition, new structures being built in flood-prone areas are driving increases in insurance premiums (reference 4).   
The FEMA National Flood Insurance Policy database includes 45 features and 50,406,943 observations of flood policy transactions collected from all states in the U.S. that are participating in the National Flood Insurance Program (NFIP). This dataset was chosen due to its ample size, and it contains many features and observations that can be utilized for predictive modeling. While this dataset contains considerable null values, data imputation was implemented to reduce bias, enabling maximal utilization as well as creation of accurate and robust models. The source of the data is FEMA, a reputable government agency. This provides transparency of the conditions in the United States with respect to flooding and allows users of the data to make informed decisions. The dataset was relatively easy to interpret, enabling creation of accurate feature descriptions to be used for predictive modeling. Finally, this dataset was chosen because our group was interested in investigating ways in which floods can be predicted and ultimately prevented as they are prevalent across the United States, as outlined above. 

-- Please refer to the [FEMA Data Exploration Jupyter Notebook](pyspark.ipynb) to review data exploration on summary statistics and data distribution. --

### Methods
#### Data Exploration
•	Printed schema and created descriptions of all the columns  
•	Displayed the Number of Variables (Columns), Number of Observations(first few rows)  
•	Identified the number of missing values for each column within the dataframe  
•	Constructed a summary of the statistics and distribution of numerical data in the dataset  
•	Ran correlations among all the numerical variables to create scatterplots of variables that were highly correlated with each other (shown from Figures 1, 2, and 3).   

#### Preprocessing
•	Created separate data frame for numerical columns  
•	Imputed null elements with mean of all values from respective column from this data frame which resolves the issue of skewing data visualization due to null values  
•	Utilized methods like .sample() and .toPandas() to create and visualize a sample dataset that is 0.1 of the original dataset (since the dataset  is too large to create scatterplots using all of the data).  

#### Model 1 – Linear Regression
•	Feature importance scores are displayed for the 19 independent variables of the Linear Regression model out of the 20 numerical columns where the dependent variable is the total insurance premium of the policy  
•	Various Linear Regression models were constructed with 1 Feature being the column that had the highest feature importance score, 5 features, 10 features, 15 features and finally 19 features  
•	Dataset is split into 70% train data and 30% test data  
•	Models are evaluated based on train and test root mean squared error (RMSE) and R-squared values using the RegressionEvaluator library  
•	A fitting graph is constructed such that x = number of features in model and y = error.   
```
lr_f19 = LinearRegression(
    featuresCol='predict_features_19', 
    labelCol='totalinsurancepremiumofthepolicy', 
    predictionCol='predicted_premium_f19', 
    regParam=0.3
)
lr_model_f19 = lr_f19.fit(train_data_f19)
predictions_rmse_train_f19 = lr_model_f19.transform(train_data_f19)
evaluator_f19 = RegressionEvaluator(labelCol = 'totalinsurancepremiumofthepolicy', predictionCol = 'predicted_premium_f19', metricName = 'rmse')
rmse_train_f19 = evaluator_f19.evaluate(predictions_rmse_train_f19)
print("Root Mean Squared Error (RMSE) on train data for Linear Regression using 19 features: {:.7f}".format(rmse_train_f19))
```
```
evaluator_r2_f19 = RegressionEvaluator(labelCol = 'totalinsurancepremiumofthepolicy', predictionCol = 'predicted_premium_f19', metricName = 'r2')
r2_train_f19 = evaluator_r2_f19.evaluate(predictions_rmse_train_f19)
print("R-squared (R2) on train data for Linear Regression using 19 features: {:.7f}".format(r2_train_f19))
```
```
predictions_rmse_test_f19 = lr_model_f19.transform(test_data_f19)
evaluator_f19 = RegressionEvaluator(labelCol = 'totalinsurancepremiumofthepolicy', predictionCol = 'predicted_premium_f19', metricName = 'rmse')
rmse_test_f19 = evaluator_f19.evaluate(predictions_rmse_test_f19)
print("Root Mean Squared Error (RMSE) on test data for Linear Regression using 19 features: {:.7f}".format(rmse_test_f19))
```
```
evaluator_r2_f19 = RegressionEvaluator(labelCol = 'totalinsurancepremiumofthepolicy', predictionCol = 'predicted_premium_f19', metricName = 'r2')
r2_test_f19 = evaluator_r2_f19.evaluate(predictions_rmse_test_f19)
print("R-squared (R2) on test data for Linear Regression using 19 features: {:.7f}".format(r2_test_f19))
```
#### Model 2 – Decision Trees
•	Feature importance scores are found for the 19 independent variables of the Decision Tree model out of the 20 numerical columns where the dependent variable is the total insurance premium of the policy  
•	Decision Tree models were constructed with 1 Feature being the column that had the highest feature importance score, 5 features, 10 features, 15 features and finally 19 features  
•	Dataset is split into 70% train data and 30% test data  
•	Models are evaluated based on train and test root mean squared error (RMSE) and R-squared values using the RegressionEvaluator library  
•	A fitting graph is constructed such that x = number of features in model and y = error    
```
dt_f19 = DecisionTreeRegressor(featuresCol='predict_features_19', labelCol='totalinsurancepremiumofthepolicy', predictionCol='predicted_premium_f19')
dt_model_f19 = dt_f19.fit(train_data_f19)
# RMSE for Train data
predictions_rmse_train_f19 = dt_model_f19.transform(train_data_f19)

evaluator_f19 = RegressionEvaluator(labelCol='totalinsurancepremiumofthepolicy', predictionCol='predicted_premium_f19', metricName='rmse')
rmse_train_f19 = evaluator_f19.evaluate(predictions_rmse_train_f19)
print("Root Mean Squared Error (RMSE) on train data for Decision Tree using 19 features: {:.7f}".format(rmse_train_f19))

# R-squared for Train data
evaluator_r2_f19 = RegressionEvaluator(labelCol='totalinsurancepremiumofthepolicy', predictionCol='predicted_premium_f19', metricName='r2')
r2_train_f19 = evaluator_r2_f19.evaluate(predictions_rmse_train_f19)
print("R-squared (R2) on train data for Decision Tree using 19 features: {:.7f}".format(r2_train_f19))

# RMSE for Test data
predictions_rmse_test_f19 = dt_model_f19.transform(test_data_f19)

evaluator_f19 = RegressionEvaluator(labelCol='totalinsurancepremiumofthepolicy', predictionCol='predicted_premium_f19', metricName='rmse')
rmse_test_f19 = evaluator_f19.evaluate(predictions_rmse_test_f19)
print("Root Mean Squared Error (RMSE) on test data for Decision Tree using 19 features: {:.7f}".format(rmse_test_f19))

# R-squared for Test data
evaluator_r2_f19 = RegressionEvaluator(labelCol='totalinsurancepremiumofthepolicy', predictionCol='predicted_premium_f19', metricName='r2')
r2_test_f19 = evaluator_r2_f19.evaluate(predictions_rmse_test_f19)
print("R-squared (R2) on test data for Decision Tree using 19 features: {:.7f}".format(r2_test_f19))
```
#### Results
#### Data Exploration
#### Column Descriptions
Descriptions of the columns from the FEMA's National Flood Insurance Policy Database, grouped by their data types and purposes:

##### Geographic and Location Details  
censustract (long): Census tract number indicating the specific area where the property is located, used for demographic analysis.  
countycode (integer): Numeric code representing the county in which the property is insured.  
floodzone (string): Designation of the flood zone according to FEMA's mapping, crucial for assessing the property's flood risk.  
latitude (double), longitude (double): Geographic coordinates specifying the precise location of the property.  
propertystate (string): The U.S. state where the property is located.  
reportedcity (string): The city reported for the insured property.  
reportedzipcode (integer): Zip code where the property is situated, used for localizing insurance coverage and risk.  
##### Property and Construction Details  
agriculturestructureindicator (string): Indicates whether the property is used for agricultural purposes.  
basementenclosurecrawlspacetype (integer): Type of basement or crawlspace present at the property, affecting flood risk assessment.  
construction (string): Describes the type of construction materials and methods used, which can affect the property's vulnerability to flood damage.  
numberoffloorsininsuredbuilding (integer): Total floors in the insured building, important for determining potential flood damage and insurance coverage needs.  
elevatedbuildingindicator (string): Indicates whether the building is elevated, a key factor in reducing flood risk.  

##### Policy Details
policycost (integer): The total cost of the flood insurance policy.  
policycount (integer): The number of policies associated with a single property or account.  
policyeffectivedate (date), policyterminationdate (date): Start and end dates of the flood insurance coverage.  
totalbuildinginsurancecoverage (integer), totalcontentsinsurancecoverage (integer): The amount of insurance coverage for the building and its contents, respectively.  
totalinsurancepremiumofthepolicy (integer): Total premium amount for the flood insurance policy.  

##### Flood Risk Assessment Specifics
basefloodelevation (double): The base flood elevation expected for a particular area, critical for understanding flood risk levels.  
elevationcertificateindicator (string), elevationdifference (integer): Presence of an elevation certificate and the difference in elevation, respectively, both crucial for assessing compliance with floodplain management.  
lowestadjacentgrade (double), lowestfloorelevation (double): Measures of elevation that help determine the property's flood exposure.  

##### Insurance Policy Features
crsdiscount (double): Community Rating System discount applied to the policy, which can reduce insurance premiums based on community flood preparedness.  
deductibleamountinbuildingcoverage (integer), deductibleamountincontentscoverage (integer): Deductible amounts for building and contents coverage, influencing out-of-pocket costs after a flood.  
hfiaasurcharge (integer): Surcharge applied under the Homeowner Flood Insurance Affordability Act.  
federalpolicyfee (integer): A fee associated with the federal policy governing flood insurance.  

##### Special Indicators
condominiumindicator (string), primaryresidenceindicator (string): Indicate whether the insured property is a condominium or the primary residence of the owner.  
houseofworshipindicator (string), nonprofitindicator (string): Indicators of whether the property is used as a house of worship or is owned by a nonprofit organization, affecting policy terms and possibly qualifying for special considerations.  
postfirmconstructionindicator (string): Indicates if the building was constructed after the community's first Flood Insurance Rate Map was issued, which can affect insurance rates.  
smallbusinessindicatorbuilding (string): Indicates whether the insured building is used for small business purposes.  
##### Additional Policy and Coverage Information
originalconstructiondate (date), originalnbdate (date): Dates of original construction and the building's initial notebook entry, important for historical property assessments.  
cancellationdateoffloodpolicy (date): Date when the flood policy was cancelled, if applicable.  
regularemergencyprogramindicator (string): Indicates the type of FEMA program under which the policy is covered, distinguishing between regular and emergency management programs.  
ratemethod (integer): Describes the method used to calculate the insurance rate, impacting how premiums are determined.  
locationofcontents (string): Specifies where within the property the insured contents are located, relevant for claims and risk assessments.  

**** MISSING DATA TABLE GOES HERE ****

#### High Missing Values:
Cancellation Date of Policy, Obstruction Type, Agriculture Structure Indicator, Lowest Adjacent Grade, Non-Profit Indicator, House of Worship Indicator, Base Flood Elevation, Small Business Indicator Building, Lowest Floor Elevation, Elevation Certificate Indicator: These features have approximately 43-32 million missing values each.   

#### Moderate Missing Values:
Location of Contents, Deductible Amount in Contents, Rate Method: These features range between approximately 15.3 million and 902 thousand million missing values.  

#### Low Missing Values:
Deductible Amount in Building Coverage, Census Tract, Latitude, Longitude, Original Construction Date, Post Firm Construction Indicator, Flood Zone, Number of Floors in Insured Building, County Code, Primary Residence Indicator: These fields range from approximately 661 thousand to 21 thousand missing values.  

#### Minimal to No Missing Values:
The remaining features in the dataframe have between approximately 800 and 0 missing values. A full breakdown of missing values can be observed in Table 1 above. 

#### Measures of Central Tendency and Dispersion
Base Flood Elevation:  
Average (Mean): 119.47 ft.    
Standard Deviation: 522.49 ft.  
Range: -9999 - 85,640 ft.    
Lowest Adjacent Grade:  
Average (Mean): 129.20 ft.    
Standard Deviation: 609.92 ft.    
Range: -9,999 - 99,990.9 ft.    
Lowest Floor Elevation:  
Average (Mean): 385.62 ft.  
Standard Deviation: 1,676.42 ft.  
Range: -9,997.9 - 99,989 ft.  
Basement Enclosure Crawl Space Type:  
Average (Mean): 0.37, indicating a slight bias towards lower classifications.  
Standard Deviation: 0.86, showing moderate variability within the data.  
Range: Min 0 to Max 4, spanning several classification levels.  
Census Tract:  
Average (Mean): Approximately 2.6 x 10¹⁰.  
Standard Deviation: About 1.58 x 10¹⁰, suggesting a wide spread across census tracts.  
CRS Discount:  
Average: 0.064, typically low across the dataset.  
Standard Deviation: 0.091, with most data points close to zero but some higher values.  
Deductible Amount in Building and Contents Coverage:  
Building Coverage Average: 1.66 with a deviation of 1.46.  
Contents Coverage Average: 0.98 with a deviation of 1.05.  
Both show low average deductible amounts but with notable variation.  
Elevation Difference:  
Average: 1.69, indicating minor differences in elevation on average.  
Standard Deviation: 3.39, suggesting significant outliers affecting the elevation difference.  
Policy Related Figures (Policy Cost, Policy Count, Total Insurance Coverage, etc.):  
These values have a high mean and standard deviation, indicating a significant spread in the policy costs and coverages, reflecting diverse insurance policies and property valuations.  

#### Extremes (Minimum and Maximum Values)
Notable minimums include negative values in Federal Policy Fee and HFIAA Surcharge, indicating refunds or adjustments.  
The maximum values in Total Building Insurance Coverage and Total Insurance Premium of the Policy reach into the hundreds of millions, highlighting cases with exceptionally high insurance coverage.  
Implications The substantial missing data in critical geographical and elevation columns could significantly hinder risk assessment accuracy. The wide variability in policy costs and coverage levels underscores the diverse nature of the insured properties. Accurate and complete data in these fields are crucial for effective risk management and policy pricing in flood insurance.  
The initial exploration provided a basis for further data cleaning, particularly in addressing missing values and outliers, which were essential for improving data quality and reliability.  

#### Preprocessing
A numerical dataframe was created by dropping categorical and otherwise other features that were not numeric, leaving only the 20 features that were ultimately used for modeling. Column means were imputed for numerical data, eliminating all missing data for the 20 features that were used for modeling.  
Outliers were examined, but ultimately retained. Initially, negative values in columns were scrutinized but found to represent refunds on insurance policies and premiums.  

#### Model 1 – Linear Regression
Five different Linear regression models were tested. Models with 10 or more features performed exceptionally well, as indicated by the very low RMSE values and high R² values (close to 1). This suggests that the linear relationship between the features and the insurance premium is well captured by the linear regression model. 
The best Linear Regression Model is summarized below (and in Table 1):  
Features: 19  
Train RMSE: 117.5607  
Train R²: 0.9949  
Test RMSE: 118.0066  
Test R²: 0.9950  
```
lr_f19 = LinearRegression(
    featuresCol='predict_features_19', 
    labelCol='totalinsurancepremiumofthepolicy', 
    predictionCol='predicted_premium_f19', 
    regParam=0.3
)
lr_model_f19 = lr_f19.fit(train_data_f19)
predictions_rmse_train_f19 = lr_model_f19.transform(train_data_f19)
evaluator_f19 = RegressionEvaluator(labelCol = 'totalinsurancepremiumofthepolicy', predictionCol = 'predicted_premium_f19', metricName = 'rmse')
rmse_train_f19 = evaluator_f19.evaluate(predictions_rmse_train_f19)
print("Root Mean Squared Error (RMSE) on train data for Linear Regression using 19 features: {:.7f}".format(rmse_train_f19))
```
```
evaluator_r2_f19 = RegressionEvaluator(labelCol = 'totalinsurancepremiumofthepolicy', predictionCol = 'predicted_premium_f19', metricName = 'r2')
r2_train_f19 = evaluator_r2_f19.evaluate(predictions_rmse_train_f19)
print("R-squared (R2) on train data for Linear Regression using 19 features: {:.7f}".format(r2_train_f19))
```
```
predictions_rmse_test_f19 = lr_model_f19.transform(test_data_f19)
evaluator_f19 = RegressionEvaluator(labelCol = 'totalinsurancepremiumofthepolicy', predictionCol = 'predicted_premium_f19', metricName = 'rmse')
rmse_test_f19 = evaluator_f19.evaluate(predictions_rmse_test_f19)
print("Root Mean Squared Error (RMSE) on test data for Linear Regression using 19 features: {:.7f}".format(rmse_test_f19))
```
```
evaluator_r2_f19 = RegressionEvaluator(labelCol = 'totalinsurancepremiumofthepolicy', predictionCol = 'predicted_premium_f19', metricName = 'r2')
r2_test_f19 = evaluator_r2_f19.evaluate(predictions_rmse_test_f19)
print("R-squared (R2) on test data for Linear Regression using 19 features: {:.7f}".format(r2_test_f19))
```
##### Interpretation:
Low Train RMSE and Test RMSE values indicate that the model can predict insurance premiums with high accuracy.  
Train R² and Test R² values are both very close to 1, signifying that the model explains nearly all the variability in the response data around its mean.  
**** FIGURE 1 GOES HERE ****

#### Model 2 – Decision Trees
Five different Decision Tree models were tested, each with a different number of features ranging from 1 to 19. Results consistently showed that Linear Regression models significantly outperformed Decision Tree models. Linear Regression models, especially using 19 features, demonstrated superior predictive accuracy and generalizability, making Linear Regression the favorable choice for predicting flood insurance premiums in this dataset. The best Linear Decision Tree Model is summarized below:  
Features: 10  
Train RMSE: 1141.8631  
Train R²: 0.5219  
Test RMSE: 1129.1083  
Test R²: 0.5284  
```
dt_f10 = DecisionTreeRegressor(featuresCol='predict_features_10', labelCol='totalinsurancepremiumofthepolicy', predictionCol='predicted_premium_f10')
dt_model_f10 = dt_f10.fit(train_data_f10)
# RMSE for Train data
predictions_rmse_train_f10 = dt_model_f10.transform(train_data_f10)

evaluator_f10 = RegressionEvaluator(labelCol='totalinsurancepremiumofthepolicy', predictionCol='predicted_premium_f10', metricName='rmse')
rmse_train_f10 = evaluator_f10.evaluate(predictions_rmse_train_f10)
print("Root Mean Squared Error (RMSE) on train data for Decision Tree using 10 features: {:.7f}".format(rmse_train_f10))

# R-squared for Train data
evaluator_r2_f10 = RegressionEvaluator(labelCol='totalinsurancepremiumofthepolicy', predictionCol='predicted_premium_f10', metricName='r2')
r2_train_f10 = evaluator_r2_f10.evaluate(predictions_rmse_train_f10)
print("R-squared (R2) on train data for Decision Tree using 10 features: {:.7f}".format(r2_train_f10))

# RMSE for Test data
predictions_rmse_test_f10 = dt_model_f10.transform(test_data_f10)

evaluator_f10 = RegressionEvaluator(labelCol='totalinsurancepremiumofthepolicy', predictionCol='predicted_premium_f10', metricName='rmse')
rmse_test_f10 = evaluator_f10.evaluate(predictions_rmse_test_f10)
print("Root Mean Squared Error (RMSE) on test data for Decision Tree using 10 features: {:.7f}".format(rmse_test_f10))

# R-squared for Test data
evaluator_r2_f10 = RegressionEvaluator(labelCol='totalinsurancepremiumofthepolicy', predictionCol='predicted_premium_f10', metricName='r2')
r2_test_f10 = evaluator_r2_f10.evaluate(predictions_rmse_test_f10)
print("R-squared (R2) on test data for Decision Tree using 10 features: {:.7f}".format(r2_test_f10))
```
##### Interpretation:
High Train RMSE and Test RMSE values indicate that the model cannot predict insurance premiums with high accuracy.  
Train R² and Test R²  values are both approximately 0.52, signifying that the model only explains about half of the variability in the response data around its mean.  
**** FIGURE 2 GOES HERE ****

**** TABLE 2 GOES HERE ****

#### Discussion
#### Data Exploration
#### Preprocessing
To prepare our data for modeling, we first identified numerical columns and created a dataframe comprising a subset of the 45 original features. Next, column means were imputed to prevent exclusion of large numbers of observations due to missing data. After these initial steps were carried out, we were ready to begin testing our models.  
#### Model 1- Linear Regression
Linear regression models were able to effectively capture relationships between predictors in the dataset (e.g., policy cost, elevation, coverage amounts) and the target variable (total insurance premiums). We observed that our linear regression models were able to handle multicollinear features in the data more effectively than decision trees.  
In general, our linear regression models appeared to control overfitting, helping to maintain balance between bias and variance, leading to overall better performance on test data.  
In summary, the Linear Regression models performed well, capturing linear relationships in the dataset, were ability to handle multicollinearity, and worked efficiently with our large (~16gb) dataset.  

#### Model 2 – Decision Trees
Our Decision Tree models did not match performance of the Linear Regression Models, especially with the introduction of additional features. Decision trees in general are prone to overfitting, especially when dealing with high-dimensional data or datasets with many features such as this one. 
Decision trees do well at capturing non-linear interactions between features. However, if the interactions are predominantly linear (as is the case with this dataset), linear regression models will perform better, which is indeed what we observed.  

#### Conclusion 
Decision tree models, while providing better R² values and lower RMSEs compared to linear regression with fewer features, did not match the performance of the linear regression models when more features were incorporated. This was likely due to the decision tree model’s tendency to overfit and its suboptimal handling of linear relationships in the data.  

Linear Regression with 10 or more features showed decent performance, although relatively low RMSE and R² values of approximately 52%, indicate suboptimal predictive power and fit compared to Linear Regression. The decision tree models did not perform as well as the linear regression models with the same number of features, particularly in terms of generalization to the data. Hence, we recommend use of a Linear Regression model utilizing 19 features is for optimum performance in predicting insurance premiums.  

#### Recommendations for Future Work:
Advanced visualization tools like Tableau and Power BI could also be employed to create a robust, interactive dashboard. These sorts of tools offer superior performance optimization, interactivity, and visual appeal and can be built to handle large datasets such as this one efficiently and are highly customizable and can provide real-time or near real-time data integration. This approach would provide a useful framework to showcase geographical data, visualize flood risk estimates and insurance premiums, allowing users to explore and interact with the dataset more effectively.  

#### Collaboration
- Sneha Shah : 
Helped in preprocessing the dataset through data imputation, created a correlation matrix of all numerical columns, made the three scatterplots based on a sample of the dataset, aided Joyce with creating the linear models, and wrote the Introduction and Methods section of the ReadMe.
 
- Mengkong Aun:
Contributions included working on the introduction of the abstract and preprocessing stages, detailing variables, identifying missing values, and summarizing data statistics and distribution. Proposed models included Logistic Regression, Random Forest, K-Means Clustering, and DBSCAN. Developed and explained the results of the second model, the Decision Tree, and contributed to the conclusion section.

- Joyce Shiah:

- Sean Deering:





