# Predict Customer Personality to Boost Marketing Campaign by Using Machine Learning

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
### 1. Column: `Dt_Customer1`
- Years_Joined
- Days_Joined
### 2. Column: `Year_Birth`
- Age
- Age Group
### 3. Column: `Marital_Status`
- Marital_Status_Simplified
### 4. Column: `Income`
- Income_Group
### 5. Column: `Total_Amount`
### 6. Column: `Recency`
- Recency_Group
### 7. Column: `Total_Purchase`
### 8. Column: `Preferred_Channel`
### 9. Column: `Total_Campaigns_Accepted`
### 10. Column: `Conversion_Rate`
### 11. Column: `Total_Children`
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
      <td colspan="2" align="center"><img src="https://github.com/user-attachments/assets/11216b0d-660e-466a-a7c0-8f9d947f1898" width="300"></td>
    </tr>
  </table>
</div>
For further and deeper analysis and explanation, kindly check:

[Improving Employee Retention (PDF)](https://github.com/bintangphylosophie/Improving-Employee-Retention-by-Predicting-Employee-Attrition-Using-Machine-Learning/blob/main/Improving%20Employee%20Retention%20by%20Predicting%20Employee%20Attrition%20Using%20Machine%20Learning.pdf)



# ⚙️ Stage 4: Data Preprocessing
This stage is focusing on data preprocessing of the dataset transform the data to make it suitable for modeling
## 1. **Feature Encoding** 🏷️<br>
I will manually encode the values for the columns `Education`, `Marital_Simplified`, and `Age_Group` using .map to ensure they are ordered correctly.

## 2. **Data Scaling** 🏷️<br>
Data scaling is the process of transforming feature values within a dataset to ensure they have a uniform range. I performed the numerical features using MinMaxScaler to improve machine learning algorithm performance that will be perform in next stage.

## 3. **Feature Selection** 🎯<br>
For segmenting customer, there is a method called RFM Analysis, 
for you want to know deeply about RFM can read this reference : https://www.barilliance.com/rfm-analysis/#:~:text=RFM%20analysis%20is%20a%20data,much%20they've%20spent%20overall .
- Recency      : Date of Last of Purchases `Recency`
- Frequency    : Total Number of Orders    `Total_Purchases`
- Monetization : Total order value         `Total_Amount`
- Loyalty      : Total campaign accepted   `Total_Campaigns_Accepted`

After completing all preprocessing steps, the data is now clean and ready for machine learning to perform customer segmentation through clustering. 🤖

<br>
<br>


# 🚀 Stage 5: Modelling
<br>
<p align="center">
<img src="https://media0.giphy.com/media/Y4PkFXkfTeEKqGBBsC/giphy.gif?cid=ecf05e47numetagrmbl2rxf6x0lpcjgq1s340dfdh1oi0x9w&ep=v1_gifs_related&rid=giphy.gif&ct=g" width="420">
</p>
At this stage, the primary objective is to develop and assess models for clustering customer segments using the preprocessed data.🎯

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
jupyter notebook Predict Customer Personality to Boost Marketing Campaign by Using Machine Learning.ipynb
```


**Key steps in this last stage include:**

## 1. **Number of CLuster**: 🏗️<br>

<div align="center">
  <img src="https://github.com/user-attachments/assets/3c678e3d-b699-4f2e-bb57-9d35255b607e" width="300">
</div>

I experimented with both inertia and the elbow method to determine the optimal number of clusters for this machine learning task. As shown in the image below, considering both the silhouette score and the inertia score, we can observe that the elbow point—along with the peak silhouette score—occurs at 4 clusters.
Next, I will apply the K-Means algorithm for clustering, using 4 as the optimal number of clusters based on our previous analysis.
<br> 

## 2. **Principal Component Analysis (PCA)**: 🏋️‍♀️🎯<br>

To visualize the clustered data, I applied Principal Component Analysis (PCA) with n_components = 2, reducing the data to two dimensions.
<div align="center">
  <img src="https://github.com/user-attachments/assets/80450b2f-3043-4d59-9f2d-d3fa40658fce" width="300">
</div>

## 3. **Cluster Visualization**: 🥇<br>
<div align="center">
  <table>
    <tr>
      <td><img src="https://github.com/user-attachments/assets/3fd3f0b5-bf86-4237-bcdb-8e81233dacae" width="400"></td>
    </tr>
    <tr>
      <td><img src="https://github.com/user-attachments/assets/26ee320d-3ead-4d3c-90f3-b055bbf0dd5f" width="400"></td>
    </tr>
    <tr>
      <td><img src="https://github.com/user-attachments/assets/c5249833-0cb8-4962-b2dc-f77420f9c503" width="400"></td>
    </tr>
    <tr>
      <td><img src="https://github.com/user-attachments/assets/7122bdcf-a4e1-4973-8b86-bec6f8ecc47e" width="400"></td>
    </tr>
  </table>
</div>



## ✅ Business Recommendation
To improve retention, the company deploys a machine learning model to identify key factors leading to resignations. with AI-driven insights, the company designs targeted retention strategies:

- Double down on High-Valued Customers: Maximize revenue through loyalty and expansion.
- Revive Medium-Valued Customers: They’ve shown spending potential, just need the right trigger.
- Test and Learn with Low/Very Low Segments: Use low-risk, low-cost experiments to re-engage—or accept potential churn and focus resources elsewhere.
## Acknowledgements🌟

I would like to express our deepest appreciation to Rakamin Academy for providing the opportunity to work on this exciting project. The experience and knowledge we gained throughout this journey have been invaluable.

Finally, I would like to thank those who provided their support and encouragement throughout my journey.

Regards, Bintang Phylosophie

<br>
<p align="center">
<img src="https://media1.giphy.com/media/3ohs7JG6cq7EWesFcQ/giphy.gif?cid=ecf05e47v1a2kre6ziee4vjvkc67vxhrwh8ho4089wc0aqli&ep=v1_gifs_search&rid=giphy.gif&ct=g" width="800">
</p>


