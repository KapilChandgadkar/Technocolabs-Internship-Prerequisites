#!/usr/bin/env python
# coding: utf-8

# <h1>BigMart Sales Mini Project

# <h3>Name- Kapil Kishor Chandgadkar
# 

# <h3>TechnoColabs Data Science Internship

# In[315]:


#Importing essential libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[316]:


#Loading the dataset
train = pd.read_csv("C:/Users/kapil/OneDrive/Desktop/TechnoColabs-Internship-Prerequisites/Technocolabs Mini Project/Train.csv")
test = pd.read_csv("C:/Users/kapil/OneDrive/Desktop/TechnoColabs-Internship-Prerequisites/Technocolabs Mini Project/Test.csv")


# In[317]:


print(train.shape, test.shape)


# In[318]:


train


# In[319]:


test


# In[320]:


#Combine test and train into one file
train['source']='train'
test['source']='test'
data = pd.concat([train, test],ignore_index=True)
print(train.shape, test.shape, data.shape)


# In[321]:


#Returns the first x number of rows when head(num).Without a number it returns first 5 rows.
data.head()


# In[322]:


#Returns basic statistics on numeric columns
data.describe().T


# <h1>Exploratory Data Analysis on train data

# In[323]:


plt.hist(train['Item_Outlet_Sales'], bins = 20, color = 'pink')
plt.title('Target Variable')
plt.xlabel('Item Outlet Sales')
plt.ylabel('count')
plt.show()


# In[324]:


# checking the columns of the data set
print(train.columns)


# In[325]:


#Returns different datatypes for each columns(float,int,string,bool etc.)
train.dtypes


# In[326]:


#Checking the different items in Item Identifier
train['Item_Identifier'].value_counts()


# In[327]:


#We will analyze only the training set
train['Item_Identifier'].value_counts(normalize = True)
train['Item_Identifier'].value_counts().plot.hist()
plt.title('Different types of item available in the store')
plt.xlabel('Item Identifier')
plt.ylabel('Number of Items')
plt.legend()
plt.show()


# In[328]:


# checking the different items in Item Fat Content
train['Item_Fat_Content'].value_counts()


# In[329]:


#Checking different varieties of item fat content
train['Item_Fat_Content'].value_counts(normalize = True)
train['Item_Fat_Content'].value_counts().plot.bar()
plt.title('Different varieties of fats in item in the store')
plt.xlabel('Fat')
plt.ylabel('Number of Items')
plt.show()


# In[330]:


#Checking the different items in Item Type
train['Item_Type'].value_counts()


# In[331]:


#We will analyze only the training set
train['Item_Type'].value_counts(normalize = True)
train['Item_Type'].value_counts().plot.bar()
plt.title('Different types of item available in the store')
plt.xlabel('Item')
plt.ylabel('Number of Items')
plt.show()


# In[332]:


#Checking the different types of Outlet Identifier
train['Outlet_Identifier'].value_counts()


# In[333]:


#We will analyze only the training set
train['Outlet_Identifier'].value_counts(normalize = True)
train['Outlet_Identifier'].value_counts().plot.bar()
plt.title('Different types of outlet identifier in the store')
plt.xlabel('Item')
plt.ylabel('Number of Items')
plt.show()


# In[334]:


#Checking the different types of Outlet Size
train['Outlet_Size'].value_counts()


# In[335]:


#We will analyze only the training set
train['Outlet_Size'].value_counts(normalize = True)
train['Outlet_Size'].value_counts().plot.bar()
plt.title('Different types of outlet sizes in the store')
plt.xlabel('Item')
plt.ylabel('Number of Items')
plt.show()


# In[336]:


#Checking different types of items in Outlet Location Type
train['Outlet_Location_Type'].value_counts()


# In[337]:


#We will analyze only the training set
train['Outlet_Location_Type'].value_counts(normalize = True)
train['Outlet_Location_Type'].value_counts().plot.bar()
plt.title('Different types of outlet location types in the store')
plt.xlabel('Item')
plt.ylabel('Number of Items')
plt.show()


# In[338]:


#Checking different types of item in Outlet Type
train['Outlet_Type'].value_counts()


# In[339]:


#We will analyze only the training set
train['Outlet_Type'].value_counts(normalize = True)
train['Outlet_Type'].value_counts().plot.bar()
plt.title('Different types of outlet types in the store')
plt.xlabel('Item')
plt.ylabel('Number of Items')
plt.show()


# <h1>Data Preprocessing

# In[340]:


#Checking unique values in the columns of both train and test dataset
data.apply(lambda x: len(x.unique()))


# Missing values

# In[341]:


data.apply(lambda x: sum(x.isnull()))


# Item_Weight, Outlet_Size have missing values and On test data set have no column Item_Outlet_Sales , we combain test set and train set into one data set so it shows 5681 null values in test dataset

# 
# 
# *   Medium    2793
# *   Small     2388
# *   High       932
# 
# On the Outlet_Size Medium size is more than other size so we fill NAN with **medium**

# In[342]:


data.Outlet_Size = data.Outlet_Size.fillna('Medium')


# Item_Weight datatype is numeric so we replace NULL value with **MEAN** of that column

# In[343]:


data.Item_Weight = data.Item_Weight.fillna(data.Item_Weight.mean())


# Item_Outlet_Sales datatype is numeric so we replace NULL value with MEAN of that column

# In[344]:


data.Item_Outlet_Sales = data.Item_Outlet_Sales.fillna(data.Item_Outlet_Sales.mean())


# In[345]:


data.apply(lambda x: sum(x.isnull()))


# In[346]:


#Returns the first x number of rows when head(num).Without a number it returns first 5 rows.
data.head()


# On Item_Fat_Content column the unique values are Low Fat, Regular, LF,reg,low fat.Here we can see Low fat, LF and low fat are same so we replace these with Low fat and Regular, reg are same so we replace these values with Regular. 

# In[347]:


data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({'LF':'Low Fat', 'reg':'Regular', 'low fat':'Low Fat'})
data['Item_Fat_Content'].value_counts()


# Create small values for establishment year by find the difference with current year

# In[348]:


data['Outlet_Years'] = 2022 - data['Outlet_Establishment_Year']


# In[349]:


data['Outlet_Years']


# Item type combine wih Item_Identifier FD = Food,  NC = Non-Consumable, DR = Drinks
# 

# In[350]:


#Item type combine:
data['Item_Identifier'].value_counts()
data['Item_Type_Combined'] = data['Item_Identifier'].apply(lambda x: x[0:2])
data['Item_Type_Combined'] = data['Item_Type_Combined'].map({'FD':'Food',  'NC':'Non-Consumable', 'DR':'Drinks'})
data['Item_Type_Combined'].value_counts()


# In[351]:


#Returns the first x number of rows when head(num).
data.head(100)


# <h1>Exploratory Data Analysis(EDA)-Outliers

# In[352]:


sns.boxplot(data=data["Item_Weight"],orient="h")


# <h1>Exploratory Data Analysis after Data Preprocessing

# In[353]:


sns.countplot(data['Item_Type_Combined'])
plt.xticks(rotation = 'vertical')


# In[354]:


sns.countplot(data['Outlet_Type'])
plt.xticks(rotation = 'vertical')


# In[355]:


sns.countplot(data['Outlet_Location_Type'])
plt.xticks(rotation = 'vertical')


# In[356]:


sns.countplot(data['Outlet_Size'])
plt.xticks(rotation = 'vertical')


# In[357]:


sns.countplot(data['Outlet_Establishment_Year'])


# In[358]:


sns.countplot(data['Outlet_Years'])


# In[359]:


corr = data.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')


# In[360]:


#Returns the first x number of rows when head(num).Without a number it returns first 5 rows.
data.head()


# <h3>Converting Categorical To Numerical by using LabelEncoder and One Hot Encoder

# In[361]:


from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le=LabelEncoder()
data.columns


# In[362]:


#Returns the first x number of rows when head(num).Without a number it returns first 5 rows.
data.head()


# In[363]:


cat_var=["Item_Fat_Content","Outlet_Size","Outlet_Location_Type","Outlet_Type","Item_Type_Combined"]
for i in cat_var:
    data[i]=le.fit_transform(data[i])


# In[364]:


#Returns the first x number of rows when head(num).Without a number it returns first 5 rows.
data.head()


# In[365]:


data=pd.get_dummies(data,columns=["Item_Fat_Content","Outlet_Size","Outlet_Location_Type","Outlet_Type","Item_Type_Combined"])
data.dtypes


# In[366]:


data.drop(["Item_Identifier","Item_Type","Outlet_Identifier","Outlet_Establishment_Year"],axis=1,inplace=True)


# In[367]:


train_df=data.loc[data["source"]=="train"]
test_df=data.loc[data["source"]=="test"]


# In[368]:


train_df.drop(["source"],axis=1,inplace=True)
test_df.drop(["source","Item_Outlet_Sales"],axis=1,inplace=True)


# In[369]:


train_df.head()


# In[370]:


test_df.head()


# In[371]:


train_df.to_csv("C:/Users/kapil/OneDrive/Desktop/TechnoColabs-Internship-Prerequisites/Technocolabs Mini Project/train_modified.csv",index=False)
test_df.to_csv("C:/Users/kapil/OneDrive/Desktop/TechnoColabs-Internship-Prerequisites/Technocolabs Mini Project/test_modified.csv",index=False)


# In[372]:


train_data_mod=pd.read_csv("C:/Users/kapil/OneDrive/Desktop/TechnoColabs-Internship-Prerequisites/Technocolabs Mini Project/train_modified.csv")


# In[373]:


X = train_data_mod.drop(['Item_Outlet_Sales'], axis=1)
y = train_data_mod.Item_Outlet_Sales


# <h2>Seperating Train And Test Data

# In[374]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,random_state=101,test_size=0.2)


# <h2>Model Building

# In[375]:


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn import metrics
from sklearn.metrics import r2_score,mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# <h2>Linear Regression

# In[376]:


model1 = LinearRegression()
model1.fit(x_train, y_train)
#Predicting the test set results
y_pred_linear= model1.predict(x_test)
print(y_pred_linear)
print('R2 score',r2_score(y_test,y_pred))


# In[377]:


print('Mean Absolute Error: ',metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error: ',metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error: ',np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('Explained Variance Score: ',metrics.explained_variance_score(y_test, y_pred))


# <h2>Ridge

# In[378]:


model2 = Ridge(alpha=10)
model2.fit(x_train,y_train)
#Predicting the test set results
y_pred_ridge = model2.predict(x_test)
print(y_pred_ridge)
print('R2 score',r2_score(y_test,y_pred_ridge))


# In[379]:


print('Mean Absolute Error: ',metrics.mean_absolute_error(y_test,y_pred_ridge))
print('Mean Squared Error: ',metrics.mean_squared_error(y_test, y_pred_ridge))
print('Root Mean Squared Error: ',np.sqrt(metrics.mean_squared_error(y_test, y_pred_ridge)))
print('Explained Variance Score: ',metrics.explained_variance_score(y_test, y_pred_ridge))


# <h2>Lasso

# In[380]:


model3 = Lasso(alpha=0.001)
model3.fit(x_train,y_train)
#Predicting the test set results
y_pred_lasso = model3.predict(x_test)
print(y_pred_lasso)
print('R2 score',r2_score(y_test,y_pred_lasso))


# In[381]:


print('Mean Absolute Error: ',metrics.mean_absolute_error(y_test,y_pred_lasso))
print('Mean Squared Error: ',metrics.mean_squared_error(y_test, y_pred_lasso))
print('Root Mean Squared Error: ',np.sqrt(metrics.mean_squared_error(y_test, y_pred_lasso)))
print('Explained Variance Score: ',metrics.explained_variance_score(y_test, y_pred_lasso))


# <h2>GradientBoosting

# In[382]:


from sklearn.ensemble import GradientBoostingRegressor
model4 = GradientBoostingRegressor()
model4.fit(x_train, y_train)
#Predicting the test set results
y_pred_XG = model4.predict(x_test)
print(y_pred_XG)
print('R2 score',r2_score(y_test,y_pred_XG))


# In[383]:


print('Mean Absolute Error: ',metrics.mean_absolute_error(y_test,y_pred_XG))
print('Mean Squared Error: ',metrics.mean_squared_error(y_test, y_pred_XG))
print('Root Mean Squared Error: ',np.sqrt(metrics.mean_squared_error(y_test, y_pred_XG)))
print('Explained Variance Score: ',metrics.explained_variance_score(y_test, y_pred_XG))


# In[384]:


pip install xgboost


# <h2>XGBoost

# In[385]:


import xgboost as xgb
data_dmatrix = xgb.DMatrix(data=x_train,label=y_train)
xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', 
                          learning_rate =0.1,
                          n_estimators=41,
                          max_depth=3,
                          min_child_weight=5,
                          gamma=0,
                          subsample=0.75,
                          colsample_bytree=0.95,
                          nthread=4,
                          scale_pos_weight=1,
                          reg_alpha=0.021,
                          seed=42,
                          )
xg_reg.fit(x_train,y_train)


# In[386]:


y_predregxs = xg_reg.predict(x_test)
print('R2 score',r2_score(y_test,y_predregxs))


# In[387]:


print('Mean Absolute Error: ',metrics.mean_absolute_error(y_test,y_predregxs))
print('Mean Squared Error: ',metrics.mean_squared_error(y_test, y_predregxs))
print('Root Mean Squared Error: ',np.sqrt(metrics.mean_squared_error(y_test, y_predregxs)))
print('Explained Variance Score: ',metrics.explained_variance_score(y_test, y_predregxs))


# <h2>KFold method in cross-validation on XGBoost

# In[388]:


from sklearn.model_selection import cross_val_score, KFold
kfold = KFold(n_splits=10, shuffle=True)
kf_cv_scores = cross_val_score(xg_reg, x_train, y_train, cv=kfold )
print("K-fold CV average score: %.2f" % kf_cv_scores.mean())


# <h1>Save model with XGBoost model

# In[389]:


import pickle
#Open a file, where you want to store the data
file = open('C:/Users/kapil/OneDrive/Desktop/TechnoColabs-Internship-Prerequisites/Technocolabs Mini Project/xg_model.pkl', 'wb')
#Dump information to that file
pickle.dump(xg_reg, file)
file.close()


# <h1>Read the model

# In[390]:


with open("C:/Users/kapil/OneDrive/Desktop/TechnoColabs-Internship-Prerequisites/Technocolabs Mini Project/xg_model.pkl","rb") as file1:
    model=pickle.load(file1)


# <h1>Predict with Test CSV file

# In[391]:


test_df.head()


# In[392]:


test_df['Item_Outlet_Sales'] = (model.predict(test_df)).tolist()


# In[393]:


test_df.head()


# In[403]:


#Save Output dataframe in csv file
test_df.to_csv("C:/Users/kapil/OneDrive/Desktop/TechnoColabs-Internship-Prerequisites/Technocolabs Mini Project/Test_Output.csv",index=False)


# In[409]:


from typing_extensions import Final
final=pd.concat([test["Item_Identifier"],test["Outlet_Identifier"],pd.DataFrame(test_df["Item_Outlet_Sales"].tolist(),columns=["Item_Outlet_Sales"])],)


# In[410]:


final.head()


# In[ ]:




