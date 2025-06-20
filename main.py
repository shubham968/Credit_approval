#!/usr/bin/env python
# coding: utf-8

# In[1]:


from ucimlrepo import fetch_ucirepo 
  
credit_approval = fetch_ucirepo(id=27) 
X = credit_approval.data.features 
y = credit_approval.data.targets 
 
print(credit_approval.variables) 


# In[2]:


import pandas as pd 
df = pd.concat([X,y],axis=1)
df_original = df.copy()


# In[3]:


df.head()


# ## EDA

# In[4]:


df.info()


# In[5]:


df.isnull().sum()


# In[6]:


df['A16'].head()


# In[7]:


df['A16'].unique()


# In[8]:


print(df['A16'].value_counts(dropna=False))


# In[9]:


df['A16'] = df['A16'].astype(str).str.strip()
df['A16'] = df['A16'].map({'+': 1, '-': 0})

df = df[df['A16'].notna()]

print(df['A16'].unique())


# ## Count Plot

# In[10]:


import seaborn as sns
import matplotlib.pyplot as plt

cat_features =['A1', 'A4', 'A5', 'A6', 'A7', 'A9', 'A10', 'A12', 'A13']
for col in cat_features:
    plt.figure(figsize=(6,4))
    sns.countplot(x=col, hue='A16', data=df)
    plt.title(f'Count plot of {col} grouped by Target')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# In[11]:


cat_features = ['A1', 'A4', 'A5', 'A6', 'A7', 'A9', 'A10', 'A12', 'A13']

for col in cat_features:
    print(f"\n--{col}--")
    print(df[col].value_counts(normalize=True))


# In[12]:


for col in cat_features:
    print(f"\n---- {col} vs Target ----")
    print(pd.crosstab(df[col], df['A16'], normalize='index'))


# In[13]:


df['A16'].value_counts(normalize=True)


# In[14]:


for col in cat_features:
    freqs = df[col].value_counts(normalize=True)
    rare = freqs[freqs <0.05].index
    df[col] = df[col].apply(lambda x:'Other' if x in rare else x)


# In[15]:


for col in cat_features:
        print(df[col].isnull().mean() *100)


# In[16]:


len(df['A16'])


# In[17]:


df['A15'].isnull().sum()


# ## Handling missing values

# In[18]:


for col in cat_features:
    df[col].fillna(df[col].mode()[0], inplace=True)


# In[19]:


numeric_cols = df.select_dtypes(include=['float64','int64']).columns.drop('A15','A16')


# In[20]:


print(df[numeric_cols].dtypes)


# In[21]:


df[numeric_cols].hist(figsize=(12,8),bins=20)
plt.suptitle("Histograms of Numerical Features",fontsize=16)
plt.show()


# In[22]:


print(df[numeric_cols].dtypes)


# In[23]:


for col in numeric_cols:
    df[col].fillna(df[col].median(), inplace=True)


# In[24]:


import numpy as np
for col in numeric_cols:
    np.log1p(df[col])


# ## Outlier detection

# In[25]:


outlier_info = {}
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3-Q1
    lower_bound = Q1-1.5*IQR
    upper_bound = Q3+1.5*IQR
    
    outliers = df[(df[col] < lower_bound) | (df[col]>upper_bound)]
    num_outliers = outliers.shape[0]
    percent_outliers = (num_outliers / df.shape[0]) *100

    outlier_info[col] = {
            "Q1": Q1,
            "Q3": Q3,
            "IQR": IQR,
            "Lower Bound": lower_bound,
            "Upper Bound": upper_bound,
            "Outliers Count": num_outliers,
            "Outliers %": round(percent_outliers, 2)
        }

import pandas as pd
outlier_df = pd.DataFrame(outlier_info).T
outlier_df.sort_values(by="Outliers %", ascending=False)
    
    
    


# In[26]:


print(numeric_cols)


# In[27]:


for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3-Q1
    lower_bound = Q1-1.5*IQR
    upper_bound = Q3+1.5*IQR
    
    outliers = df[(df[col] < lower_bound) | (df[col]>upper_bound)]
    num_outliers = outliers.shape[0]
    percent_outliers = (num_outliers / df.shape[0]) *100
    
    if percent_outliers < 5 :
        df = df[(df[col]>= lower_bound) & (df[col]<= upper_bound)]
        


# In[28]:


skewness  = df[numeric_cols].skew()
skewed_cols = skewness[skewness > 1].index.tolist()
print(skewed_cols)


# In[29]:


for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = np.where(df[col] < lower, lower,
                       np.where(df[col] > upper, upper, df[col]))
    outlier_percent = ((df[col] < lower) | (df[col] > upper)).mean() * 100
    print(f"{col}: {outlier_percent:.2f}% outliers remaining")


# In[38]:


import seaborn as sns
import matplotlib.pyplot as plt

for col in numeric_cols:
    plt.figure(figsize=(6, 1.5))
    sns.boxplot(x=df[col], color='skyblue')
    plt.title(f'Boxplot of {col} after outlier handling')
    plt.show()



# In[31]:


from numpy import log1p
for col in skewed_cols:
    df[col] = log1p(df[col])


# ## Data splitting and model training (Logistic Regression)

# In[32]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report


# In[33]:


X = df.drop(columns='A16')
y = df['A16']


# In[34]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[35]:


import category_encoders as ce

cat_cols = ['A1','A6','A4' ,'A5', 'A7', 'A9', 'A10', 'A12', 'A13'] 

encoder = ce.TargetEncoder(cols=cat_cols)

X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols], y_train)

X_test[cat_cols] = encoder.transform(X_test[cat_cols])




# ## Model Training

# In[36]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

models = {
    'Random Forest': RandomForestClassifier(),
    'XGBoost': XGBClassifier(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
}

for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"{name}: {scores.mean():.4f}")


# In[37]:


from sklearn.model_selection import cross_val_score

for name, model in models.items():
    f1 = cross_val_score(model, X_train, y_train, cv=5, scoring='f1').mean()
    roc = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc').mean()
    print(f"{name}: F1 = {f1:.4f}, ROC-AUC = {roc:.4f}")


# In[ ]:




