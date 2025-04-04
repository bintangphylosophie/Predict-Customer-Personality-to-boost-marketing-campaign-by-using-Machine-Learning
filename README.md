# Predict Customer Personality to boost marketing campaign by using Machine Learning

<br>
<p align="center">
<img src="https://github.com/user-attachments/assets/7711749d-4bce-4b24-8c21-2a6b40947c31" width="500">
</p>
<br>
This dataset contains customer information related to demographics, purchasing behavior, and marketing campaign responses. It includes details such as age, education, marital status, income, and household composition. Additionally, it tracks spending across different product categories, purchase channels, and engagement with marketing campaigns. The dataset also records customer complaints and recent interactions, making it valuable for analyzing consumer behavior, predicting marketing campaign success, and improving customer relationship management strategies. This dataset is from a fictional company in 2012-2014. 

## 📚 Installation

This project requires Python and the following Python libraries installed:

- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scipy
- Sklearn

If you don't have Python installed yet, it's recommended that you install [the Anaconda distribution](https://www.anaconda.com/distribution/) of Python, which already has the above packages and more included.

To install the Python libraries, you can use pip:

```bash
pip install numpy pandas matplotlib seaborn scipy sklearn imbalanced-learn
```
To run the Jupyter notebook, you also need to have Jupyter installed. If you installed Python using Anaconda, you already have Jupyter installed. If not, you can install it using pip:
```bash
pip install jupyter
```

Once you have Python and the necessary libraries, you can run the project using Jupyter Notebook:
```bash
jupyter notebook Improving Employee Retention by Predicting Employee Attrition Using Machine Learning.ipynb
```
## Project Overview
A company can develop rapidly when it knows its customers' personality behavior, so that it can provide better services and benefits to customers who have the potential to become loyal customers. By processing historical marketing campaign data to improve performance and target the right customers so they can make transactions on the company's platform, from this data insight our focus is to create a cluster prediction model to make it easier for companies to make decisions.

## Problem
The company wants to better understand its customer base by identifying distinct customer segments based on demographic information, purchasing behavior, and engagement with marketing campaigns. Currently, marketing strategies are applied uniformly across all customers, leading to suboptimal targeting and lower campaign effectiveness. By implementing a clustering model, the company aims to segment customers into meaningful groups to personalize marketing efforts and improve customer retention.

## 🎯 Goals 
The goal of the company is to enhance customer relationship management by developing data-driven marketing strategies that improve engagement, increase sales, and optimize resource allocation based on customer segmentation insights.

## 🏁 Obective 
- Segment Customers: Apply clustering techniques to group customers based on demographics, purchasing behavior, and engagement with marketing campaigns.
- Analyze Customer Profiles: Identify key characteristics of each segment, such as spending habits, preferred shopping channels, and responsiveness to promotions.
- Marketing Strategies: Develop targeted campaigns tailored to distinct customer segments to enhance engagement and conversion rates.
- Boost Customer Retention: Detect high-value and at-risk customers to implement proactive retention strategies.
-Optimize Resource Allocation: Improve business efficiency by focusing marketing efforts on the most profitable and responsive customer groups.


The process will go through the following steps to achieve the objectives:
1. Data Understanding
2. Data Preprocessing
3. Feature Engineering
4. Insight
5. Exploratory Data Analysis
6. Data Preparation
7. Machine Learning
<br>
<br>

# 🔎Stage 1: Data Understanding and Preprocessing 

<br>
<p align="center">
<img src="https://user-images.githubusercontent.com/74038190/212749726-d36b8253-74bb-4509-870d-e29ed3b8ff4a.gif" width="500">
</p>
<br>

## 📊 About Dataset 

### 📋Dataset Information and Preparation:
- There are 2240 rows and 30 columns in the dataset. 
- The dataset contains 24 rows with missing values only in the `Income` column. After careful consideration, I decided to fill them with the median value.
- The dataset contains 183 duplicate entries, which I decided to drop as they are unnecessary for our machine learning modeling.
- The data distribution of the features appears to be fine, except for two columns: `Year_Birth` and `Income`. I removed outliers from these columns using the Z-score method.


### 📝Features
**Identification & Demographic Features**
| Feature | Explanation |
|---------|-------------|
| Unnamed: 0 | Likely an unnecessary index column from data export. |
| ID | Unique identifier for each customer. |
| Year_Birth | The birth year of the customer. |
| Education | The education level of the customer |
| Marital_Status | Marital status of the customer (e.g., Single, Married, Divorced). |
| Income | Annual income of the customer. |

**Household Information**
| Feature | Explanation |
|---------|-------------|
| Kidhome | Number of small children in the household. |
| Teenhome | Number of teenagers in the household. |

**Customer Relationship & Activity**
| Feature | Explanation |
|---------|-------------|
| Dt_Customer | Date when the customer was enrolled in the company’s database. |
| Recency | Number of days since the last purchase. |

**Product Purchases**
| Feature | Explanation |
|---------|-------------|
| MntCoke | Amount spent on soft drinks. |
| MntFruits | Amount spent on fruits. |
| MntMeatProducts | Amount spent on meat products. |
| MntFishProducts | Amount spent on fish products. |
| MntSweetProducts | Amount spent on sweets. |
| MntGoldProds | Amount spent on gold products. |

**Purchase Behavior (Number of Purchases by Channel)**
| Feature | Explanation |
|---------|-------------|
| NumDealsPurchases | Number of purchases made using a discount or deal. |
| NumWebPurchases | Number of purchases made through the company's website. |
| NumCatalogPurchases | Number of purchases made via catalog orders. |
| NumStorePurchases | Number of purchases made in physical stores. |
| NumWebVisitsMonth | Number of website visits in the last month|

**Marketing Campaign Responses**
| Feature | Explanation |
|---------|-------------|
| AcceptedCmp1 | Whether the customer accepted Campaign 1. |
| AcceptedCmp2 | Whether the customer accepted Campaign 2. |
| AcceptedCmp3 | Whether the customer accepted Campaign 3. |
| AcceptedCmp4 | Whether the customer accepted Campaign 4. |
| AcceptedCmp5 | Whether the customer accepted Campaign 5. |
| Response | Whether the customer accepted the most recent marketing campaign. |

**Customer Feedback & Revenue Features**
| Feature | Explanation |
|---------|-------------|
| Complain | Whether the customer has made a complaint in the past. |
| Z_CostContact | A constant value (likely for standardization in cost-related calculations). |
| Z_Revenue | A derived revenue-related feature (possibly customer lifetime value or standardized revenue metric). |
<br>
<br>



# ⚙️ Stage 2: Feature Engineering

<br>
<p align="center">
<img src="https://user-images.githubusercontent.com/74038190/221352995-5ac18bdf-1a19-4f99-bbb6-77559b220470.gif" width="400">
</p>


This is the second stage of  focusing on feature engineering of the dataset. The main goal of this stage is to clean and transform the raw data to make it suitable for data analysis. 🧹🔄

<br>

**Key steps in this stage include:**
## 1. Column: `Dt_Customer1`
- Years_Joined
- Days_Joined
## 2. Column: `Year_Birth`
- Age
- Age Group
## 3. Column: `Marital_Status`
- Marital_Status_Simplified
## 4. Column: `Income`
- Income_Group
## 5. Column: `Total_Amount`
## 6. Column: `Recency`
- Recency_Group
## 7. Column: `Total_Purchase`
## 8. Column: `Preferred_Channel`
## 9. Column: `Total_Campaigns_Accepted`
## 10. Column: `Conversion_Rate`
## 11. Column: `Total_Children`
<br>
<br>

# 🚀 Stage 3: Insight

<br>
<p align="center">
<img src="https://media0.giphy.com/media/Y4PkFXkfTeEKqGBBsC/giphy.gif?cid=ecf05e47numetagrmbl2rxf6x0lpcjgq1s340dfdh1oi0x9w&ep=v1_gifs_related&rid=giphy.gif&ct=g" width="420">
</p>

This is the next phase of the project, focusing on gaining insights. Here are some valuable insights derived from the dataset
<div align="center">
  <table>
    <tr>
      <td><img src="https://github.com/user-attachments/assets/9b5557c4-1eb4-4500-be16-60101a749776" width="300"></td>
      <td><img src="https://github.com/user-attachments/assets/dc040fd0-69ca-490e-ad8b-6755b34cd12d" width="300"></td>
    </tr>
    <tr>
      <td><img src="https://github.com/user-attachments/assets/de2b3f3d-67c0-4032-a22a-16f910dca774" width="300"></td>
      <td><img src="https://github.com/user-attachments/assets/fad627a6-402a-46cd-a794-4dab51f40c12" width="300"></td>
    </tr>
    <tr>
      <td><img src="https://github.com/user-attachments/assets/778114d6-466e-4566-85fb-23ff3dd0b8bf" width="300"></td>
      <td><img src="https://github.com/user-attachments/assets/74cad2f4-d6ba-44aa-a2e3-81724bf5839e" width="300"></td>
    </tr>
    <tr>
      <td><img src="https://github.com/user-attachments/assets/9b1980ff-255b-4ce2-adba-87bb5542f6a2" width="300"></td>
      <td><img src="https://github.com/user-attachments/assets/a2aac737-95c8-4492-8cb4-78a1fd8c927b" width="300"></td>
    </tr>
    <tr>
      <td><img src="https://github.com/user-attachments/assets/56202fb5-0ac3-4445-8b2e-557d5b003378" width="300"></td>
      <td><img src="https://github.com/user-attachments/assets/0b5f4514-94c6-4a47-9821-e1e092a2274a" width="300"></td>
    </tr>
    <tr>
      <td><img src="https://github.com/user-attachments/assets/d6fb16ad-0081-4162-8b8b-228729b2b93f" width="300"></td>
      <td><img src="https://github.com/user-attachments/assets/1568c9ad-d8e2-49b9-bacb-54d11102acd1" width="300"></td>
    </tr>
    <tr>
      <td><img src="https://github.com/user-attachments/assets/f4d1faa3-1a0d-41e3-8dc6-734c0be86b66" width="300"></td>
      <td><img src="https://github.com/user-attachments/assets/50a57ef2-f242-4d9b-b304-3273803818a3" width="300"></td>
    </tr>
    <tr>
      <td><img src="https://github.com/user-attachments/assets/86359cd9-82ef-4a10-91f4-1fade10fa45b" width="300"></td>
      <td><img src="https://github.com/user-attachments/assets/0f208e7d-847a-4415-9359-eaa29e351761" width="300"></td>
    </tr>
    <tr>
      <td><img src="https://github.com/user-attachments/assets/11216b0d-660e-466a-a7c0-8f9d947f1898" width="300"></td>
    </tr>
  </table>
</div>
For further and deeper analysis and explanation, kindly check:

[Improving Employee Retention (PDF)](https://github.com/bintangphylosophie/Improving-Employee-Retention-by-Predicting-Employee-Attrition-Using-Machine-Learning/blob/main/Improving%20Employee%20Retention%20by%20Predicting%20Employee%20Attrition%20Using%20Machine%20Learning.pdf)



# ⚙️ Stage 4: Data Preprocessing
This stage is focusing on data preprocessing of the dataset transform the data to make it suitable for modeling
## 1. **Feature Encoding** 🏷️<br>
I encode all of our categorical features (strings) using the label encoding method. All features have been encoded at the feature extraction stage, given that our features are ordinal data and the majority of machine learning algorithms perform better with numerical data.

## 2. **Data Scaling** 🏷️<br>
Data scaling is the process of transforming feature values within a dataset to ensure they have a uniform range. I performed using MinMaxScaler to improve machine learning algorithm performance that will be perform in next stage.

## 3. **Feature Selection** 🎯<br>
Since I am working with categorical features and a classification problem, SelectKBest with chi2 is a great choice because it is: fast, helps remove irrelevant features, works well with encoded categorical data. This method is commonly used to reduce dimensionality by keeping only the most relevant features for predictive modeling. I only keep top 15 features.

## 4. **Data Splitting**<br>
Data splitting is the process of dividing a dataset into different subsets to train, validate, and test a machine learning model. 
The main goal of this stage is to build and evaluate models that can predict the target variable based on the preprocessed data. The data splitted into 20% train and 80% test.


Prior to this, the dataset was split into training and testing sets. After completing all preprocessing steps, the data is clean and ready for machine learning to predict the target.🤖

<br>
<br>


# 🚀 Stage 5: Modelling
<br>
<p align="center">
<img src="https://media0.giphy.com/media/Y4PkFXkfTeEKqGBBsC/giphy.gif?cid=ecf05e47numetagrmbl2rxf6x0lpcjgq1s340dfdh1oi0x9w&ep=v1_gifs_related&rid=giphy.gif&ct=g" width="420">
</p>
The main goal of this stage is to build and evaluate models that can predict the target variable based on the preprocessed data. 🎯

## 📚 Installation

This project requires Python and the following Python libraries installed:

- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scipy
- Sklearn
- shap

If you don't have Python installed yet, it's recommended that you install [the Anaconda distribution](https://www.anaconda.com/distribution/) of Python, which already has the above packages and more included.

To install the Python libraries, you can use pip:

```bash
pip install numpy pandas matplotlib seaborn scipy sklearn imbalanced-learn xgboost shap
```
To run the Jupyter notebook, you also need to have Jupyter installed. If you installed Python using Anaconda, you already have Jupyter installed. If not, you can install it using pip:
```bash
pip install jupyter
```

Once you have Python and the necessary libraries, you can run the project using Jupyter Notebook:
```bash
jupyter notebook Improving Employee Retention by Predicting Employee Attrition Using Machine Learning.ipynb
```


**Key steps in this last stage include:**

## 1. **Models**: 🏗️<br>

I experimented with other algorithms. A total of 5 algorithms were tested during the experiment, including:
- **Support Vector Machine**
- **Gradient Boosting**
- **Decision Tree**
- **Random Forest**
- **Logistic Regression**

<br> 

**Tuning:** Hyperparameter tuning was performed only on the 1 best algorithms (those with the highest ROC AUC score).

## 2. **Model Training and Evaluation**: 🏋️‍♀️🎯<br>

The following are the prediction results with the highest Accurcy and ROC-AUC:

### Model Results
| Model Name | Accuracy | ROC AUC | 
|------------|--------------|-------------|
| Support Vector Machine | 0.67 | 0.50 | 
| Gradient Boosting | 0.94 | 0.94 | 
| Decision Tree | 0.93 | 0.93 | 
| Random Forest | 0.89 | 0.85 | 
| Logistic Regression | 0.70 | 0.57 | 


We discovered that the Gradient Boosting Model with the highest Accuracy (0.94) and ROC AUC (0.94), stability compared to other models. 

## 3. **Model Selection**: 🥇<br>

### Model Results
| Model Name | Accuracy | ROC AUC | 
|------------|--------------|-------------|
| Gradient Boosting | **0.94** | **0.94** |

with confussion matrich and ROC AUC score like this:

<div align="center">
  <table>
    <tr>
      <td><img src="https://github.com/user-attachments/assets/259784b2-2b39-4912-b555-26b4d56dfdeb" width="400"></td>
      <td><img src="https://github.com/user-attachments/assets/ffb76db3-751f-4ef9-b82a-56c05b3ec943" width="400"></td>
    </tr>
  </table>
</div>

## 🔑 Feature Importance 
Based on the Gradient Boosting model Feature Importance:

<div align="center">
  <img src="https://github.com/user-attachments/assets/200d233e-1cdc-403e-bb6a-8887b523f375" width="400">
</div>

<br>

The top importance feature with score > 0.01 are :
- `AlasanResign`
- `AsalDaerah_JakartaSelatan`
- `UsiaKaryawan`
- `HiringPlatform_Diversity_Jobfair`


## ✅ Business Recommendation
To improve retention, the company deploys a machine learning model to identify key factors leading to resignations. with AI-driven insights, the company designs targeted retention strategies:

### The model strongly relies on "AlasanResign" (Resignation Reason), meaning that resignation trends must be analyzed deeply to take proactive action. Since working hour is the dominant factor, some action recommend:
- **Analyze Overtime Trends:** Check which departments consistently work overtime and why.
- **Survey Employees:** Get feedback on workload, stress levels, and preferred work hours.
- **Compare Productivity Metrics:** Measure performance vs. hours worked to find the optimal balance.
- **Implement Flexible Working Hours:** Allow employees to choose a start time (e.g., 7 AM - 3 PM, 9 AM - 5 PM, 11 AM - 7 PM)
- **Implement Hybrid & Remote Work Options:** Allow employees to work 2-3 days from home per week.
- **Reduce Unnecessary Meetings & Improve Time Efficiency:** Limit Meetings to 30-45 Minutes, Set "No-Meeting Days“,Block 1-2 days per week for deep work without interruptions.

### The model detects that employees from Jakarta Selatan and mid-employees have a higher risk of resigning. 
- **Personalized Exit Interviews:** Focus on  mid-employee and Jakarta Selatan employees to understand their concerns, "Stay Interviews": Monthly check-ins with at-risk employees
- **Internal Mentorship Programs:** Pair mid-level employees with senior mentors.

#### Machine learning provided valuable insights, but human intervention was key in designing effective retention strategies. XYZ Corp launches AI-powered retention programs for 6 months and tracks the impact.

## 📝 Business Impact Simulation  
Company Name Profile: **XYZ Corp**
A tech company, XYZ Corp, has been struggling with high employee turnover, affecting productivity, morale, and recruitment costs. To improve retention, the company deploys a machine learning model to identify key factors leading to resignations. with AI-driven insights, the company designs targeted retention strategies.

<div align="center">
  <img src="https://github.com/user-attachments/assets/78c8ecef-dbf8-444e-9c8c-4906da6ebf09" width="600">
</div>


## Acknowledgements🌟

I would like to express our deepest appreciation to Rakamin Academy for providing the opportunity to work on this exciting project. The experience and knowledge we gained throughout this journey have been invaluable.

Finally, I would like to thank those who provided their support and encouragement throughout my journey.

Regards, Bintang Phylosophie

<br>
<p align="center">
<img src="https://media1.giphy.com/media/3ohs7JG6cq7EWesFcQ/giphy.gif?cid=ecf05e47v1a2kre6ziee4vjvkc67vxhrwh8ho4089wc0aqli&ep=v1_gifs_search&rid=giphy.gif&ct=g" width="800">
</p>


