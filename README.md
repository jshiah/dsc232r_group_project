# dsc232r_group_project
UCSD Master of Data Science DSC232R - PySpark Data Analysis Group Project - Spring 2024

# Introduction
Within the United States, flooding is a common hazard that affects all 50 states. Multiple factors such as weather trends and natural environment modifications influence flood risk, not just history.<sup>1</sup>
Physical damage to structures can be covered by the National Flood Insurance Policy (NFIP) to repair damage when flooding occurs. Building coverage and content coverage are two kinds of policies provided by NFIP to keep structures and belongings protected in the event that flood-related damage occurs.<sup>2</sup>
The way that the Federal Emergency Management Agency (FEMA) estimates flood policy pricing has changed, and is dependent on many features such as levels of rainfall, rebuilding costs, and a specific property’s distance from the ocean. For instance, premiums are expected to increase by 342% in the South Florida neighborhoods of Tequesta and Jupiter, which are especially prone to flooding. This area is already the most expensive South Florida neighborhood for flood insurance, having premiums in excess of $7000 dollars per year. Florida on the whole faces increased flood risks due to climate change and inclement weather, and new structures being built are driving increases in newly issued insurance premiums.<sup>3</sup>  

The FEMA National Flood Insurance Policy database<sup>4</sup> includes 45 feature columns and 50,406,943 observations of policy transactions collected from all states in the U.S. that are participating in the National Flood Insurance Program (NFIP). This dataset was chosen for a number of reasons. First, it is adequately sized, and contains many features which should allow us to carry out predictive modeling. While this dataset contains many missing values, we plan to implement data imputation to enable the dataset to be fully utilized as well as allow for creation of more accurate and robust models. The data is provided by FEMA, a reputable source of government data, which provides valuable insight into the conditions in the United States with respect to flooding and empowers users of the data to make informed, data-driven decisions. Next, the columns within the dataset were easy to interpret, enabling creation of accurate feature descriptions, making it easier to carry out predictive modeling. Finally, this dataset was chosen because we were interested in investigating ways in which floods can be predicted and ultimately mitigated or prevented, as they are so prevalent in the United States as discussed above.


-- Please refer to the [FEMA Data Exploration Jupyter Notebook](pyspark.ipynb) to review data exploration on summary statistics and data distribution. --

----

 <sup>1</sup>Floods - FEMA building science branch hazard overview. (2017, March). https://www.fema.gov/sites/default/files/2020-07/fema_p1086_flood_2017.pdf  
 <sup>2</sup>From the pages of The Cost of Flooding and What’s Covered? pages of Win this hurricane season. | The National Flood Insurance Program. (n.d.). https://www.floodsmart.gov/  
 <sup>3</sup>Rivero, N. (2023, May 21). Flood insurance costs will soar in Florida. See the expected increases in your zip code. WUSF. https://www.wusf.org/weather/2023-05-21/flood-insurance-costs-soar-florida-see-expected-increases-zip-code  
 <sup>4</sup>FEMA’s National Flood Insurance Policy Database. Kaggle. (2020, May 29). https://www.kaggle.com/datasets/lynma01/femas-national-flood-insurance-policy-database 







----
# Methods
## Data Exploration
1. Printed schema and created descriptions of all the columns.  
1. Displayed the Number of Variables (Columns), Number of Observations(first few rows).
1. Identified the number of missing values for each column within the dataframe. 
1. Constructed a summary of the statistics and distribution of numerical data in the dataset.
1. Ran correlations among all the numerical variables to create scatterplots of variables that were highly correlated with each other.   

## Preprocessing
1. Created separate data frame for numerical columns  
1. Imputed null elements with mean of all values from respective column from this data frame which resolves the issue of skewing data visualization due to null values.  
1. Utilized methods like .sample() and .toPandas() to create and visualize a sample dataset that is 0.1 of the original dataset (since the dataset  is too large to create scatterplots using all of the data).  

## Model 1 – Linear Regression
1. Feature importance scores are displayed for the 19 independent variables of the Linear Regression model out of the 20 numerical columns where the dependent variable is the total insurance premium of the policy  
1. Various Linear Regression models were constructed with 1 Feature being the column that had the highest feature importance score, 5 features, 10 features, 15 features and finally 19 features  
1. Dataset is split into 70% train data and 30% test data  
1. Models are evaluated based on train and test root mean squared error (RMSE) and R-squared values using the RegressionEvaluator library  
1. A fitting graph is constructed such that x = number of features in model and y = error.   
``` python
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
``` python
evaluator_r2_f19 = RegressionEvaluator(labelCol = 'totalinsurancepremiumofthepolicy', predictionCol = 'predicted_premium_f19', metricName = 'r2')
r2_train_f19 = evaluator_r2_f19.evaluate(predictions_rmse_train_f19)
print("R-squared (R2) on train data for Linear Regression using 19 features: {:.7f}".format(r2_train_f19))
```
``` python
predictions_rmse_test_f19 = lr_model_f19.transform(test_data_f19)
evaluator_f19 = RegressionEvaluator(labelCol = 'totalinsurancepremiumofthepolicy', predictionCol = 'predicted_premium_f19', metricName = 'rmse')
rmse_test_f19 = evaluator_f19.evaluate(predictions_rmse_test_f19)
print("Root Mean Squared Error (RMSE) on test data for Linear Regression using 19 features: {:.7f}".format(rmse_test_f19))
```
``` python
evaluator_r2_f19 = RegressionEvaluator(labelCol = 'totalinsurancepremiumofthepolicy', predictionCol = 'predicted_premium_f19', metricName = 'r2')
r2_test_f19 = evaluator_r2_f19.evaluate(predictions_rmse_test_f19)
print("R-squared (R2) on test data for Linear Regression using 19 features: {:.7f}".format(r2_test_f19))
```
## Model 2 – Decision Trees
1. Feature importance scores are found for the 19 independent variables of the Decision Tree model out of the 20 numerical columns where the dependent variable is the total insurance premium of the policy  
1. Decision Tree models were constructed with 1 Feature being the column that had the highest feature importance score, 5 features, 10 features, 15 features and finally 19 features  
1. Dataset is split into 70% train data and 30% test data  
1. Models are evaluated based on train and test root mean squared error (RMSE) and R-squared values using the RegressionEvaluator library  
1. A fitting graph is constructed such that x = number of features in model and y = error    
``` python
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
# Results
## Data Exploration
## Column Descriptions
Descriptions of the columns from the FEMA's National Flood Insurance Policy Database, grouped by their data types and purposes:

## Geographic and Location Details  
`censustract (long)`: Census tract number indicating the specific area where the property is located, used for demographic analysis.  
`countycode (integer)`: Numeric code representing the county in which the property is insured.  
`floodzone (string)`: Designation of the flood zone according to FEMA's mapping, crucial for assessing the property's flood risk.  
`latitude (double)`, longitude (double): Geographic coordinates specifying the precise location of the property.  
`propertystate (string)`: The U.S. state where the property is located.  
`reportedcity (string)`: The city reported for the insured property.  
`reportedzipcode (integer)`: Zip code where the property is situated, used for localizing insurance coverage and risk.  
## Property and Construction Details  
`agriculturestructureindicator (string)`: Indicates whether the property is used for agricultural purposes.  
`basementenclosurecrawlspacetype (integer)`: Type of basement or crawlspace present at the property, affecting flood risk assessment.  
`construction (string)`: Describes the type of construction materials and methods used, which can affect the property's vulnerability to flood damage.  
`numberoffloorsininsuredbuilding (integer)`: Total floors in the insured building, important for determining potential flood damage and insurance coverage needs.  
`elevatedbuildingindicator (string)`: Indicates whether the building is elevated, a key factor in reducing flood risk.  

## Policy Details
`policycost (integer)`: The total cost of the flood insurance policy.  
`policycount (integer)`: The number of policies associated with a single property or account.  
`policyeffectivedate (date)`, `policyterminationdate (date)`: Start and end dates of the flood insurance coverage.  
`totalbuildinginsurancecoverage (integer)`, `totalcontentsinsurancecoverage (integer)`: The amount of insurance coverage for the building and its contents, respectively.  
`totalinsurancepremiumofthepolicy (integer)`: Total premium amount for the flood insurance policy.  

## Flood Risk Assessment Specifics
`basefloodelevation (double)`: The base flood elevation expected for a particular area, critical for understanding flood risk levels.  
`elevationcertificateindicator (string)`, `elevationdifference (integer)`: Presence of an elevation certificate and the difference in elevation, respectively, both crucial for assessing compliance with floodplain management.  
`lowestadjacentgrade (double)`, `lowestfloorelevation (double)`: Measures of elevation that help determine the property's flood exposure.  

## Insurance Policy Features
`crsdiscount (double)`: Community Rating System discount applied to the policy, which can reduce insurance premiums based on community flood preparedness.  
`deductibleamountinbuildingcoverage (integer)`, `deductibleamountincontentscoverage (integer):` Deductible amounts for building and contents coverage, influencing out-of-pocket costs after a flood.  
`hfiaasurcharge (integer)`: Surcharge applied under the Homeowner Flood Insurance Affordability Act.  
`federalpolicyfee (integer)`: A fee associated with the federal policy governing flood insurance.  

## Special Indicators
`condominiumindicator (string)`, `primaryresidenceindicator (string)`: Indicate whether the insured property is a condominium or the primary residence of the owner.  
`houseofworshipindicator (string)`, `nonprofitindicator (string)`: Indicators of whether the property is used as a house of worship or is owned by a nonprofit organization, affecting policy terms and possibly qualifying for special considerations.  
`postfirmconstructionindicator (string)`: Indicates if the building was constructed after the community's first Flood Insurance Rate Map was issued, which can affect insurance rates.  
`smallbusinessindicatorbuilding (string)`: Indicates whether the insured building is used for small business purposes.  
## Additional Policy and Coverage Information
`originalconstructiondate (date)`, `originalnbdate (date)`: Dates of original construction and the building's initial notebook entry, important for historical property assessments.  
`cancellationdateoffloodpolicy (date)`: Date when the flood policy was cancelled, if applicable.  
`regularemergencyprogramindicator (string)`: Indicates the type of FEMA program under which the policy is covered, distinguishing between regular and emergency management programs.  
`ratemethod (integer)`: Describes the method used to calculate the insurance rate, impacting how premiums are determined.  
`locationofcontents (string)`: Specifies where within the property the insured contents are located, relevant for claims and risk assessments.  

## Missing Data

|Feature:                           | Missing Values:|
| -----------------------------------|----------------|
 cancellationdateoffloodpolicy      | 43614057
 obstructiontype                    | 40629070
 agriculturestructureindicator      | 38923313 
 lowestadjacentgrade                | 34940579 
 nonprofitindicator                 | 34493094
 houseofworshipindicator            | 34476251
 basefloodelevation                 | 33636759 
 smallbusinessindicatorbuilding     | 33451148 
 lowestfloorelevation               | 33060602
 elevationcertificateindicator      | 32606397
 locationofcontents                 | 15389767
 deductibleamountincontentscoverage | 5561584
 ratemethod                         | 902967     
 deductibleamountinbuildingcoverage | 661993   
 censustract                        | 467119  
 latitude                           | 338699   
 longitude                          | 338699
 originalconstructiondate           | 180318 
 postfirmconstructionindicator      | 180276      
 floodzone                          | 169145 
 numberoffloorsininsuredbuilding    | 162301    
 countycode                         | 48999
 primaryresidenceindicator          | 21884        
 basementenclosurecrawlspacetype    | 802      
 elevatedbuildingindicator          | 258      
 construction                       | 13 
 reportedzipcode                    | 7
 occupancytype                      | 6              
 condominiumindicator               | 6 
 policytermindicator                | 3
 regularemergencyprogramindicator   | 2        
 reportedcity                       | 2           
 crsdiscount                        | 0        
 elevationdifference                | 0        
 federalpolicyfee                   | 0        
 hfiaasurcharge                     | 0        
 originalnbdate                     | 0        
 policycost                         | 0        
 policycount                        | 0        
 policyeffectivedate                | 0        
 policyterminationdate              | 0        
 propertystate                      | 0               
 totalbuildinginsurancecoverage     | 0        
 totalcontentsinsurancecoverage     | 0        
 totalinsurancepremiumofthepolicy   | 0  

**Table 1**: This table outlines the number of missing values in the dataset. This is described in further detail below.

### High Missing Values:
Cancellation Date of Policy, Obstruction Type, Agriculture Structure Indicator, Lowest Adjacent Grade, Non-Profit Indicator, House of Worship Indicator, Base Flood Elevation, Small Business Indicator Building, Lowest Floor Elevation, Elevation Certificate Indicator: These features have approximately 43-32 million missing values each.   

### Moderate Missing Values:
Location of Contents, Deductible Amount in Contents, Rate Method: These features range between approximately 15.3 million and 902 thousand million missing values.  

### Low Missing Values:
Deductible Amount in Building Coverage, Census Tract, Latitude, Longitude, Original Construction Date, Post Firm Construction Indicator, Flood Zone, Number of Floors in Insured Building, County Code, Primary Residence Indicator: These fields range from approximately 661 thousand to 21 thousand missing values.  

### Minimal to No Missing Values:
The remaining features in the dataframe have between approximately 800 and 0 missing values. A full breakdown of missing values can be observed in Table 1 above. 

## Measures of Central Tendency and Dispersion
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

### Extremes (Minimum and Maximum Values)
Notable minimums include negative values in Federal Policy Fee and HFIAA Surcharge, indicating refunds or adjustments.  
The maximum values in Total Building Insurance Coverage and Total Insurance Premium of the Policy reach into the hundreds of millions, highlighting cases with exceptionally high insurance coverage.  
Implications The substantial missing data in critical geographical and elevation columns could significantly hinder risk assessment accuracy. The wide variability in policy costs and coverage levels underscores the diverse nature of the insured properties. Accurate and complete data in these fields are crucial for effective risk management and policy pricing in flood insurance.  
The initial exploration provided a basis for further data cleaning, particularly in addressing missing values and outliers, which were essential for improving data quality and reliability.  

# Preprocessing
1. A numerical dataframe was created by dropping non-numeric features, leaving the 20 features that were ultimately used for modeling. Column means were imputed, eliminating all missing data.
1. Outliers were examined, but ultimately retained. Initially, negative values in columns were scrutinized but found to represent refunds on insurance policies and premiums.  

# Model 1 – Linear Regression
1. Five different Linear regression models were tested.
1. Models with 10 or more features performed exceptionally well, as indicated by the very low RMSE values and high R² values (close to 1).
1. This suggests that the linear relationship between the features and the insurance premium is well captured by the linear regression model. 
- The best Linear Regression Model is summarized below (and in Table 1):  
Features: 19  
Train RMSE: 117.5607  
Train R²: 0.9949  
Test RMSE: 118.0066  
Test R²: 0.9950  
``` python
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
``` python
evaluator_r2_f19 = RegressionEvaluator(labelCol = 'totalinsurancepremiumofthepolicy', predictionCol = 'predicted_premium_f19', metricName = 'r2')
r2_train_f19 = evaluator_r2_f19.evaluate(predictions_rmse_train_f19)
print("R-squared (R2) on train data for Linear Regression using 19 features: {:.7f}".format(r2_train_f19))
```
``` python
predictions_rmse_test_f19 = lr_model_f19.transform(test_data_f19)
evaluator_f19 = RegressionEvaluator(labelCol = 'totalinsurancepremiumofthepolicy', predictionCol = 'predicted_premium_f19', metricName = 'rmse')
rmse_test_f19 = evaluator_f19.evaluate(predictions_rmse_test_f19)
print("Root Mean Squared Error (RMSE) on test data for Linear Regression using 19 features: {:.7f}".format(rmse_test_f19))
```
``` python
evaluator_r2_f19 = RegressionEvaluator(labelCol = 'totalinsurancepremiumofthepolicy', predictionCol = 'predicted_premium_f19', metricName = 'r2')
r2_test_f19 = evaluator_r2_f19.evaluate(predictions_rmse_test_f19)
print("R-squared (R2) on test data for Linear Regression using 19 features: {:.7f}".format(r2_test_f19))
```
### Interpretation:
Low Train RMSE and Test RMSE values indicate that the model can predict insurance premiums with high accuracy.  
Train R² and Test R² values are both very close to 1, signifying that the model explains nearly all the variability in the response data around its mean. 

  ![image](https://github.com/deerings/dsc232r_group_project/assets/12570888/2385a3af-9212-4267-b17e-6478f230e317)

**Figure 1.** Fitting Graph for Model 1 - Linear Regression. RMSE Decreases slightly going from one to four features, and decreases significantly going from five to ten features. RMSE remains consistently low at fifteen and nineteen features, indicating that the model fits the data relatively well with the inclusion of ten features and beyond.![image]
 
## Model 2 – Decision Trees
1. Five different Decision Tree models were tested, each with a different number of features ranging from 1 to 19.
1. Results consistently showed that Linear Regression models significantly outperformed Decision Tree models.
1. Linear Regression models, especially using 19 features, demonstrated superior predictive accuracy and generalizability, making Linear Regression the favorable choice for predicting flood insurance premiums in this dataset.
- The best Decision Tree Model is summarized below:  
Features: 10  
Train RMSE: 1141.8631  
Train R²: 0.5219  
Test RMSE: 1129.1083  
Test R²: 0.5284  
``` python
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
### Interpretation:
High Train RMSE and Test RMSE values indicate that the model cannot predict insurance premiums with high accuracy.  
Train R² and Test R²  values are both approximately 0.52, signifying that the model only explains about half of the variability in the response data around its mean.  

 ![image](https://github.com/deerings/dsc232r_group_project/assets/12570888/18f00a63-d522-42e1-ac49-fee32ab61fda)

 **Figure 2**. Fitting Graph for Model 2 - Decision Trees. RMSE decreases slightly moving from one to five features, but does not decrease significantly moving from five features to ten and on. This indicates that the decision tree models do not fit the data well.

## Model Performance Comparison

|  Model | Features | Train RMSE | Train R<sup>2</sup>| Test RMSE | Test R<sup>2</sup>|
|--------|----------|------------|--------------------|-----------|-------------------|
| Linear Regression | predict_features_1 | 1643.9153 | 0.0035 | 1652.3239 | 0.0034 |
| Decision Tree | predict_features_1 | 1292.9461 | 0.387 | 1280.3065 | 0.3938 |
 Linear Regression | predict_features_5 | 1407.3977 | 0.2815 | 1375.844 | 0.2816 |
 Decision Tree | predict_features_5 | 1152.8593 | 0.5126 |1137.6243 | 0.5214 |
 Linear Regression | predict_features_10 | 118.1326 | 0.9949 |117.1185 | 0.9949 |
 Decision Tree | predict_features_10 | 1141.8631 | 0.5219 | 1129.1083 | 0.5284|
 Linear Regression | predict_features_15 | 117.6226 | 0.9949 |118.2676 | 0.9949|
 Decision Tree | predict_features_15 | 1146.1381 | 0.5183| 1132.0466 |0.5261|
 Linear Regression | predict_features_19 | 117.5607 | 0.9949 | 118.0066 |0.995 |
 Decision Tree | predict_features_19 | 1142.108 | 0.5217 | 1127.5757 | 0.5298 |
 
**Table 2**. Comparison of Performance Metrics for Predicting Flood Insurance Premiums for Linear Regression and Decision Tree Models. The best model was Linear Regression with 19 features (highlighted).

# Discussion
## Data Exploration
## Preprocessing
1. To prepare our data for modeling, we first identified numerical columns and created a dataframe comprising a subset of the 45 original features.
1. Next, column means were imputed to prevent exclusion of large numbers of observations due to missing data.
1. After these initial steps were carried out, we were ready to begin testing our models.  
## Model 1- Linear Regression
Linear regression models were able to effectively capture relationships between predictors in the dataset (e.g., policy cost, elevation, coverage amounts) and the target variable (total insurance premiums). We observed that our linear regression models were able to handle multicollinear features in the data more effectively than decision trees.  
In general, our linear regression models appeared to control overfitting, helping to maintain balance between bias and variance, leading to overall better performance on test data.  
In summary, the Linear Regression models performed well, capturing linear relationships in the dataset, were ability to handle multicollinearity, and worked efficiently with our large (~16gb) dataset.  

## Model 2 – Decision Trees
Our Decision Tree models did not match performance of the Linear Regression Models, especially with the introduction of additional features. Decision trees in general are prone to overfitting, especially when dealing with high-dimensional data or datasets with many features such as this one. 
Decision trees do well at capturing non-linear interactions between features. However, if the interactions are predominantly linear (as is the case with this dataset), linear regression models will perform better, which is indeed what we observed.  

# Conclusion 
Decision tree models, while providing better R² values and lower RMSEs compared to linear regression with fewer features, did not match the performance of the linear regression models when more features were incorporated. This was likely due to the decision tree model’s tendency to overfit and its suboptimal handling of linear relationships in the data. 

![barplot](https://github.com/deerings/dsc232r_group_project/assets/12570888/24276bd9-e91e-457d-8a0a-4047095a970b)

**Figure 3**. Barplot comparing R<sup>2</sup> for train and test data for Linear Regression and Decision Tree Models.

Decision Tree Models with 10 or more features showed decent performance, although relatively low RMSE and R² values of approximately 52%, indicate suboptimal predictive power and fit compared to Linear Regression which had R² of 99.5% and a much lower RMSE. The decision tree models did not perform as well as the linear regression models with the same number of features, particularly in terms of generalization to the data. Hence, we recommend use of a Linear Regression model utilizing 19 features for optimum performance in predicting insurance premiums.  

# Recommendations for Future Work:
Advanced visualization tools like Tableau and Power BI could also be employed to create a robust, interactive dashboard. These sorts of tools offer superior performance optimization, interactivity, and visual appeal and can be built to handle large datasets such as this one efficiently and are highly customizable and can provide real-time or near real-time data integration. This approach would provide a useful framework to showcase geographical data, visualize flood risk estimates and insurance premiums, allowing users to explore and interact with the dataset more effectively.  

<img width="888" alt="Screenshot 2024-06-07 at 00 41 30" src="https://github.com/deerings/dsc232r_group_project/assets/12570888/cd4e1589-3410-479f-bea9-d6bf255df178">

**Figure 4**. A sample static map we created showing flood risk color-coded by region.

# Collaboration
All group members located sample data sets to use in this project, co-wrote the readme, and developed the `pyspark.ipynb` file.  
- Sneha Shah : 
Helped in preprocessing the dataset through data imputation, created a correlation matrix of all numerical columns, made the three scatterplots based on a sample of the dataset, aided Joyce with creating the linear models, and wrote the Introduction and Methods section of the ReadMe.  
 
- Mengkong Aun:
Contributions included working on the introduction of the abstract and preprocessing stages, detailing variables, identifying missing values, and summarizing data statistics and distribution. Proposed models included Logistic Regression, Random Forest, K-Means Clustering, and DBSCAN. Created flood map visualization. Developed and explained the results of the second model, the Decision Tree, and contributed to the conclusion section.  

- Joyce Shiah:
Contributed by coding the Feature Importance linear regression model to find and sort all 19 numerical features by descending order of importance, coded the 5 linear regression models with 1, 5, 10, 15, and 19 selected features, and coded the methods in finding RMSE and R-squared values for each of the 5 linear regression models. Participated in writing the abstract, as well as finalizing and compiling the abstract for Milestone 1 submission.

- Sean Deering:
Cowrote the abstract, assisted with initial data import to Spark dataframe, performed data exploration and descriptive statistics. Created fitting graphs for models 1 and 2 and other figures for readme. Co-wrote the results and discussion sections, and created the final `Readme.md` file.





