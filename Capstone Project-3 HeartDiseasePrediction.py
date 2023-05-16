#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


pd.set_option('display.max_rows',None)           #to display all the rows


# In[3]:


pd.set_option('display.max_columns',None)        #to display all the columns


# # Load the Dataset

# In[4]:


data=pd.read_csv('values.csv')
data


# In[5]:


#dropping 'patient_id' column
data.drop('patient_id',axis=1,inplace=True)


# # Business Statement
# - By using the given details of a patient we have to predict whether a patient has a heart disease or not.

# # Domain Analysis
# - patient_id: unique identifier for each patient in the dataset.
# 
# - slope_of_peak_exercise_st_segment: the slope of the peak exercise ST segment, which is a measurement of the ST segment on an electrocardiogram (ECG) during exercise. It is a categorical variable with values of 1, 2, or 3.
# 
# - thal: results of a thallium stress test, which is used to diagnose heart disease. It is a categorical variable with values of normal, fixed defect, reversible defect, or unknown.
# 
# - resting_blood_pressure: the resting blood pressure of the patient in mm Hg.
# 
# - chest_pain_type: the type of chest pain experienced by the patient, which is a categorical variable with values of typical angina, atypical angina, non-anginal pain, or asymptomatic.
# 
# - num_major_vessels: the number of major blood vessels colored by flourosopy, which is a diagnostic test that uses a special dye and X-rays to show the inside of blood vessels. It is a categorical variable with values of 0, 1, 2, 3, or 4.
# 
# - fasting_blood_sugar_gt_120_mg_per_dl: whether the patient's fasting blood sugar is greater than 120 mg/dL or not. It is a binary variable with values of 0 or 1.
# 
# - resting_ekg_results: results of a resting electrocardiogram (ECG) test, which measures the electrical activity of the heart. It is a categorical variable with values of normal, ST-T wave abnormality, or left ventricular hypertrophy.
# 
# - serum_cholesterol_mg_per_dl: the level of serum cholesterol in the patient's blood in mg/dL.
# 
# - oldpeak_eq_st_depression: the ST depression induced by exercise relative to rest, which is a measure of the heart's response to exercise. It is a numerical variable.
# 
# - sex: the gender of the patient, which is a binary variable with values of 0 for female and 1 for male.
# 
# - age: the age of the patient in years.
# 
# - max_heart_rate_achieved: the maximum heart rate achieved during exercise.
# 
# - exercise_induced_angina: whether the patient experienced angina (chest pain or discomfort) during exercise or not. It is a binary variable with values of 0 or 1.
# 
# - heart_disease_present: whether or not the patient has heart disease. It is a binary variable with values of 0 or 1.

# # Basic Checks

# In[6]:


data.info()                           #information of the dataset


# In[7]:


data.describe()                                #statistical information of the dataset


# In[8]:


data.describe(include='O')               #categorical columns statistical information of the dataset


# In[9]:


data.head()                            #first five rows of the dataset


# In[10]:


data.tail()                           #last five rows of the dataset


# In[11]:


data.shape                            #shape of the dataset


# In[12]:


data.size                             #size of the dataset


# In[13]:


data.isnull().sum()                              #checking for null values


# In[14]:


data.isna().sum()                               #checking for missing values


# In[15]:


data.duplicated().sum()                         #checking for duplicated values


# # EDA

# # Univariate Analysis

# In[16]:


import sweetviz as sv                             #library for univariate analysis
my_report=sv.analyze(data)                        #pass the original dataframe
my_report.show_html()                             #Default arguments will generate to "SWEETVIZ_REPORT.html"                          


# # Insights of Univariate Analysis
# - 52% of the patients have the priority of slope exercise segment.
# - 54% of the patients have the normal(no defect) of thal.
# - 14.4% of the patients have the 130 as the resting blood pressure.
# - 46% of the patients have the value 4 of chest pain type.
# - 59% of the patients have no major blood vessel blockages.
# - 84% of the patients have fasting blood sugar levels below 120 mg/dl, which is considered normal.
# - 52% of the patients have normal resting ECG results.
# - 33% of the patients have normal cholesterol levels, but there is a significant minority with high cholesterol levels.
# - 47% of the patients have have no or minimal ST segment depression during exercise, but there is a significant minority with moderate or severe depression, which can indicate a higher risk of heart disease.
# - 69% of the males are more likely to develop heart disease than females.
# - 20% of the Older individuals are more likely to develop heart disease than younger individuals.
# - 20% of the individuals achieve a maximum heart rate of around 150-170 beats per minute during exercise.
# - 68% of the individuals experience exercise-induced angina may be at higher risk of heart disease.
# - target column shows proportion of individuals without heart disease than with heart disease.

# # Bivariate Analysis

# In[17]:


data.columns                             #fetching columns


# In[18]:


#dividing the data into numerical and categorical data
numerical_data=data[['slope_of_peak_exercise_st_segment','resting_blood_pressure','chest_pain_type','num_major_vessels','fasting_blood_sugar_gt_120_mg_per_dl','resting_ekg_results','serum_cholesterol_mg_per_dl','oldpeak_eq_st_depression','sex','age','max_heart_rate_achieved','exercise_induced_angina','heart_disease_present']]
categorical_data=data['thal']


# In[19]:


#plotting numerical_data
plt.figure(figsize=(20,20))            #plotting the figure
plotnumber=1                           #plotnumber
for i in numerical_data:
    plt.subplot(7,2,plotnumber)        #subplot
    sns.histplot(data=numerical_data,x=i,hue=data['heart_disease_present'])        #using histplot for the data
    plotnumber+=1                      #plotnumber increment
    plt.xlabel(i,fontsize=15)
    plt.ylabel(i,fontsize=15)
plt.tight_layout()


# ### Conversion of categorical data into numerical data

# In[20]:


from sklearn.preprocessing import LabelEncoder              #importing label encoder
lc=LabelEncoder()                                           #object creation for label encoder
data.thal=lc.fit_transform(data.thal)                       #fitting thal column data to convert into numerical data


# In[21]:


data.thal.value_counts()                                    #value counts of thal column


# In[22]:


data


# # Insights of Bivariate Analysis
# ### Slope of peak exercise ST segment vs heart disease present: 
# - Slope of peak exercise ST segment and heart disease present: 
# - Individuals with a downsloping ST segment during peak exercise are more likely to have heart disease than individuals with a flat or upsloping ST segment.
# ### thal vs heart disease present
# - Thal and heart disease present: 
# - Individuals with thalassemia are more likely to have heart disease than individuals without thalassemia.
# ### resting blood pressure vs heart disease present
# - Resting blood pressure and heart disease present: 
# - Individuals with high resting blood pressure are more likely to have heart disease than individuals with normal or low resting blood pressure.
# ### chest pain type vs heart disease present
# - Chest pain type and heart disease present: 
# - Individuals with non-anginal chest pain or atypical angina are less likely to have heart disease than individuals with typical angina.
# ### Number of major vessels vs heart disease present
# - Number of major vessels and heart disease present: 
# - Individuals with one or more blocked major blood vessels are more likely to have heart disease than individuals with no blocked vessels.
# ### Fasting blood pressure vs heart disease present
# - Fasting blood sugar > 120 mg/dl and heart disease present: 
# - Individuals with high fasting blood sugar levels are more likely to have heart disease than individuals with normal fasting blood sugar levels.
# ### resting ecg results vs heart disease present
# - Resting ECG results and heart disease present: 
# - Individuals with abnormal resting ECG results are more likely to have heart disease than individuals with normal resting ECG results.
# ### Serum cholesterol vs heart disease present
# - Serum cholesterol and heart disease present: 
# - Individuals with high serum cholesterol levels are more likely to have heart disease than individuals with normal serum cholesterol levels.
# ### Oldpeak equal to ST depression vs heart disease present
# - Oldpeak equal to ST depression and heart disease present: 
# - Individuals with moderate or severe ST segment depression during exercise are more likely to have heart disease than individuals with no or minimal depression.
# ### sex vs heart disease present
# - Sex and heart disease present: 
# - Males are more likely to have heart disease than females.
# ### age vs heart disease present
# - Age and heart disease present: 
# - Older individuals are more likely to have heart disease than younger individuals.
# ### Max heart rate achieved vs heart disease present
# - Max heart rate achieved and heart disease present: 
# - Individuals with lower maximum heart rates during exercise are more likely to have heart disease than individuals with higher maximum heart rates.
# ### Exercise induced angina vs heart disease present
# - Exercise induced angina and heart disease present: 
# - Individuals who experience angina during exercise are more likely to have heart disease than individuals who do not experience angina during exercise.

# # Data Preprocessing

# In[23]:


data.isnull().sum()                                #checking for null values


# In[24]:


data.isna().sum()                                 #checking for missing values


# In[25]:


#checking for outliers
plt.figure(figsize=(20,20))            #plotting the figure
plotnumber=1                           #plotnumber
for i in data:
    plt.subplot(7,2,plotnumber)        #subplot
    sns.boxplot(data=data,x=i,hue=data['heart_disease_present'])        #using boxplot for finding outliers of the data
    plotnumber+=1                      #plotnumber increment
    plt.xlabel(i,fontsize=15)
    plt.ylabel(i,fontsize=15)
plt.tight_layout()


# # Using IQR to remove Outliers

# In[26]:


from scipy import stats             #importing stats for scipy library
IQR=stats.iqr(data.slope_of_peak_exercise_st_segment,interpolation='midpoint')        #calculating Inter quantile range
IQR


# In[27]:


Q1=data.slope_of_peak_exercise_st_segment.quantile(0.25)   #defining 25% of data
Q1


# In[28]:


Q3=data.slope_of_peak_exercise_st_segment.quantile(0.75)  #defining 75% of data
Q3


# In[29]:


min_limit=Q1-1.5*IQR              #setting minimum limit
min_limit


# In[30]:


max_limit=Q3+1.5*IQR              #setting maximum limit
max_limit


# In[31]:


data.loc[data['slope_of_peak_exercise_st_segment']<min_limit]         #checking values which are less than minimum limit


# In[32]:


data.loc[data['slope_of_peak_exercise_st_segment']>max_limit]         #checking values which are greater than maximum limit


# In[33]:


IQR=stats.iqr(data.thal,interpolation='midpoint')        #calculating Inter quantile range
IQR


# In[34]:


Q1=data.thal.quantile(0.25)   #defining 25% of data
Q1


# In[35]:


Q3=data.thal.quantile(0.75)  #defining 75% of data
Q3


# In[36]:


min_limit=Q1-1.5*IQR              #setting minimum limit
min_limit


# In[37]:


max_limit=Q3+1.5*IQR              #setting maximum limit
max_limit


# In[38]:


data.loc[data['thal']<min_limit]         #checking values which are less than minimum limit


# In[39]:


data.loc[data['slope_of_peak_exercise_st_segment']>max_limit]         #checking values which are greater than maximum limit


# In[40]:


IQR=stats.iqr(data.resting_blood_pressure,interpolation='midpoint')        #calculating Inter quantile range
IQR


# In[41]:


Q1=data.resting_blood_pressure.quantile(0.25)   #defining 25% of data
Q1


# In[42]:


Q3=data.resting_blood_pressure.quantile(0.75)  #defining 75% of data
Q3


# In[43]:


min_limit=Q1-1.5*IQR              #setting minimum limit
min_limit


# In[44]:


max_limit=Q3+1.5*IQR              #setting maximum limit
max_limit


# In[45]:


data.loc[data['resting_blood_pressure']<min_limit]         #checking values which are less than minimum limit


# In[46]:


data.loc[data['resting_blood_pressure']>max_limit]         #checking values which are greater than maximum limit


# In[47]:


data.loc[data['resting_blood_pressure']>max_limit]=np.median(data['resting_blood_pressure'])    #imputing value with median


# In[48]:


data.resting_blood_pressure=np.sqrt(data.resting_blood_pressure)


# In[49]:


IQR=stats.iqr(data.chest_pain_type,interpolation='midpoint')        #calculating Inter quantile range
IQR


# In[50]:


Q1=data.chest_pain_type.quantile(0.25)   #defining 25% of data
Q1


# In[51]:


Q3=data.chest_pain_type.quantile(0.75)  #defining 75% of data
Q3


# In[52]:


min_limit=Q1-1.5*IQR              #setting minimum limit
min_limit


# In[53]:


max_limit=Q3+1.5*IQR              #setting maximum limit
max_limit


# In[54]:


data.loc[data['chest_pain_type']<min_limit]         #checking values which are less than minimum limit


# In[55]:


data.loc[data['chest_pain_type']<min_limit]=np.median(data['chest_pain_type'])    #imputing value with median


# In[56]:


data.loc[data['chest_pain_type']>max_limit]         #checking values which are greater than maximum limit


# In[57]:


data.loc[data['chest_pain_type']>max_limit]=np.median(data['chest_pain_type'])    #imputing value with median


# In[58]:


data.chest_pain_type=np.sqrt(data.chest_pain_type)


# In[59]:


IQR=stats.iqr(data.num_major_vessels,interpolation='midpoint')        #calculating Inter quantile range
IQR


# In[60]:


Q1=data.num_major_vessels.quantile(0.25)   #defining 25% of data
Q1


# In[61]:


Q3=data.num_major_vessels.quantile(0.75)  #defining 75% of data
Q3


# In[62]:


min_limit=Q1-1.5*IQR              #setting minimum limit
min_limit


# In[63]:


max_limit=Q3+1.5*IQR              #setting maximum limit
max_limit


# In[64]:


data.loc[data['num_major_vessels']<min_limit]         #checking values which are less than minimum limit


# In[65]:


data.loc[data['chest_pain_type']>max_limit]         #checking values which are greater than maximum limit


# In[66]:


IQR=stats.iqr(data.fasting_blood_sugar_gt_120_mg_per_dl,interpolation='midpoint')        #calculating Inter quantile range
IQR


# In[67]:


Q1=data.fasting_blood_sugar_gt_120_mg_per_dl.quantile(0.25)   #defining 25% of data
Q1


# In[68]:


Q3=data.fasting_blood_sugar_gt_120_mg_per_dl.quantile(0.75)   #defining 75% of data
Q3


# In[69]:


min_limit=Q1-1.5*IQR              #setting minimum limit
min_limit


# In[70]:


max_limit=Q3+1.5*IQR              #setting maximum limit
max_limit


# In[71]:


data.loc[data['fasting_blood_sugar_gt_120_mg_per_dl']<min_limit]         #checking values which are less than minimum limit


# In[72]:


data.loc[data['fasting_blood_sugar_gt_120_mg_per_dl']>max_limit]         #checking values which are greater than maximum limit


# In[73]:


data.loc[data['fasting_blood_sugar_gt_120_mg_per_dl']>max_limit]=np.median(data['fasting_blood_sugar_gt_120_mg_per_dl'])       #imputing value with median


# In[74]:


data.fasting_blood_sugar_gt_120_mg_per_dl=np.sqrt(data.fasting_blood_sugar_gt_120_mg_per_dl)


# In[75]:


IQR=stats.iqr(data.resting_ekg_results,interpolation='midpoint')        #calculating Inter quantile range
IQR


# In[76]:


Q1=data.resting_ekg_results.quantile(0.25)   #defining 25% of data
Q1


# In[77]:


Q3=data.resting_ekg_results.quantile(0.75)   #defining 75% of data
Q3


# In[78]:


min_limit=Q1-1.5*IQR              #setting minimum limit
min_limit


# In[79]:


max_limit=Q3+1.5*IQR              #setting maximum limit
max_limit


# In[80]:


data.loc[data['resting_ekg_results']<min_limit]         #checking values which are less than minimum limit


# In[81]:


data.loc[data['resting_ekg_results']>max_limit]         #checking values which are greater than maximum limit


# In[82]:


IQR=stats.iqr(data.serum_cholesterol_mg_per_dl,interpolation='midpoint')        #calculating Inter quantile range
IQR


# In[83]:


Q1=data.serum_cholesterol_mg_per_dl.quantile(0.25)   #defining 25% of data
Q1


# In[84]:


Q3=data.serum_cholesterol_mg_per_dl.quantile(0.75)   #defining 75% of data
Q3


# In[85]:


min_limit=Q1-1.5*IQR              #setting minimum limit
min_limit


# In[86]:


max_limit=Q3+1.5*IQR              #setting maximum limit
max_limit


# In[87]:


data.loc[data['serum_cholesterol_mg_per_dl']<min_limit]         #checking values which are less than minimum limit


# In[88]:


data.loc[data['serum_cholesterol_mg_per_dl']<min_limit]=np.median(data['serum_cholesterol_mg_per_dl'])      #imputing value with median


# In[89]:


data.loc[data['serum_cholesterol_mg_per_dl']>max_limit]         #checking values which are greater than maximum limit


# In[90]:


data.loc[data['serum_cholesterol_mg_per_dl']>max_limit]=np.median(data['serum_cholesterol_mg_per_dl'])


# In[91]:


data.serum_cholesterol_mg_per_dl=np.sqrt(data.serum_cholesterol_mg_per_dl)


# In[92]:


IQR=stats.iqr(data.oldpeak_eq_st_depression,interpolation='midpoint')        #calculating Inter quantile range
IQR


# In[93]:


Q1=data.oldpeak_eq_st_depression.quantile(0.25)   #defining 25% of data
Q1


# In[94]:


Q3=data.oldpeak_eq_st_depression.quantile(0.75)   #defining 75% of data
Q3


# In[95]:


min_limit=Q1-1.5*IQR              #setting minimum limit
min_limit


# In[96]:


max_limit=Q3+1.5*IQR              #setting maximum limit
max_limit


# In[97]:


data.loc[data['oldpeak_eq_st_depression']<min_limit]         #checking values which are less than minimum limit


# In[98]:


data.loc[data['oldpeak_eq_st_depression']>max_limit]         #checking values which are greater than maximum limit


# In[99]:


data.loc[data['oldpeak_eq_st_depression']>max_limit]=np.median(data['oldpeak_eq_st_depression'])        #imputing value with median


# In[100]:


data.oldpeak_eq_st_depression=np.sqrt(data.oldpeak_eq_st_depression)


# In[101]:


IQR=stats.iqr(data.sex,interpolation='midpoint')        #calculating Inter quantile range
IQR


# In[102]:


Q1=data.sex.quantile(0.25)   #defining 25% of data
Q1


# In[103]:


Q3=data.sex.quantile(0.75)   #defining 75% of data
Q3


# In[104]:


min_limit=Q1-1.5*IQR              #setting minimum limit
min_limit


# In[105]:


max_limit=Q3+1.5*IQR              #setting maximum limit
max_limit


# In[106]:


data.loc[data['sex']<min_limit]         #checking values which are less than minimum limit


# In[107]:


data.loc[data['sex']>max_limit]         #checking values which are greater than maximum limit


# In[108]:


IQR=stats.iqr(data.age,interpolation='midpoint')        #calculating Inter quantile range
IQR


# In[109]:


Q1=data.age.quantile(0.25)   #defining 25% of data
Q1


# In[110]:


Q3=data.age.quantile(0.75)   #defining 75% of data
Q3


# In[111]:


min_limit=Q1-1.5*IQR              #setting minimum limit
min_limit


# In[112]:


max_limit=Q3+1.5*IQR              #setting maximum limit
max_limit


# In[113]:


data.loc[data['age']<min_limit]         #checking values which are less than minimum limit


# In[114]:


data.loc[data['age']<min_limit]=np.median(data['age'])           #imputing value with median


# In[115]:


data.loc[data['age']>max_limit]         #checking values which are greater than maximum limit


# In[116]:


data.age=np.sqrt(data.age)


# In[117]:


IQR=stats.iqr(data.max_heart_rate_achieved,interpolation='midpoint')        #calculating Inter quantile range
IQR


# In[118]:


Q1=data.max_heart_rate_achieved.quantile(0.25)   #defining 25% of data
Q1


# In[119]:


Q3=data.max_heart_rate_achieved.quantile(0.75)   #defining 75% of data
Q3


# In[120]:


min_limit=Q1-1.5*IQR              #setting minimum limit
min_limit


# In[121]:


max_limit=Q3+1.5*IQR              #setting maximum limit
max_limit


# In[122]:


data.loc[data['max_heart_rate_achieved']<min_limit]         #checking values which are less than minimum limit


# In[123]:


data.loc[data['max_heart_rate_achieved']>max_limit]         #checking values which are greater than maximum limit


# In[124]:


IQR=stats.iqr(data.exercise_induced_angina,interpolation='midpoint')        #calculating Inter quantile range
IQR


# In[125]:


Q1=data.exercise_induced_angina.quantile(0.25)   #defining 25% of data
Q1


# In[126]:


Q3=data.exercise_induced_angina.quantile(0.75)   #defining 75% of data
Q3


# In[127]:


min_limit=Q1-1.5*IQR              #setting minimum limit
min_limit


# In[128]:


max_limit=Q3+1.5*IQR              #setting maximum limit
max_limit


# In[129]:


data.loc[data['exercise_induced_angina']<min_limit]         #checking values which are less than minimum limit


# In[130]:


data.loc[data['exercise_induced_angina']>max_limit]         #checking values which are greater than maximum limit


# In[131]:


data.loc[data['exercise_induced_angina']>max_limit]=np.median(data['exercise_induced_angina'])       #imputing value with median


# In[132]:


data.exercise_induced_angina=np.sqrt(data.exercise_induced_angina)


# In[133]:


IQR=stats.iqr(data.heart_disease_present,interpolation='midpoint')        #calculating Inter quantile range
IQR


# In[134]:


Q1=data.heart_disease_present.quantile(0.25)   #defining 25% of data
Q1


# In[135]:


Q3=data.heart_disease_present.quantile(0.75)   #defining 75% of data
Q3


# In[136]:


min_limit=Q1-1.5*IQR              #setting minimum limit
min_limit


# In[137]:


max_limit=Q3+1.5*IQR              #setting maximum limit
max_limit


# In[138]:


data.loc[data['heart_disease_present']<min_limit]         #checking values which are less than minimum limit


# In[139]:


data.loc[data['heart_disease_present']>max_limit]         #checking values which are greater than maximum limit


# In[140]:


data.heart_disease_present=np.sqrt(data.heart_disease_present)


# # Feature Selection

# In[141]:


plt.figure(figsize=(20,20))                                  #plotting figure
numerical_data.drop('heart_disease_present',axis=1)          #dropping output column
sns.heatmap(numerical_data.corr(),annot=True)                #plotting heatmap
plt.show()


# # Model Creation

# In[142]:


data.columns


# In[143]:


#splitting the data into x and y
x=data[['slope_of_peak_exercise_st_segment','thal','resting_blood_pressure','chest_pain_type','num_major_vessels','fasting_blood_sugar_gt_120_mg_per_dl','resting_ekg_results','serum_cholesterol_mg_per_dl','oldpeak_eq_st_depression','sex','age','max_heart_rate_achieved', 'exercise_induced_angina']]
y=data.heart_disease_present


# In[144]:


x


# In[145]:


y


# # Scaling

# In[146]:


from sklearn.preprocessing import MinMaxScaler        #importing minmaxscaler
scaler=MinMaxScaler()                                 #objecct creation for scaler
x_scaled=scaler.fit_transform(x)                      #fitting the data


# In[147]:


x_scaled                                              #x_scaled values


# In[148]:


from sklearn.model_selection import train_test_split                    #importing train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_scaled,y,test_size=0.25,random_state=42)   


# In[149]:


x_train                                                           #x_train values


# In[150]:


x_test                                                              #x_test values


# In[151]:


y_train                                                           #y_train values


# In[152]:


y_test                                                                   #y_test values


# In[153]:


print(x_train.shape)                                 #shape of x_train
print(x_test.shape)                                  #shape of x_test
print(y_train.shape)                                 #shape of y_train
print(y_test.shape)                                  #shape of y_test


# In[154]:


#checking for imbalancing data
data.heart_disease_present.value_counts()


# In[155]:


#smoting
from imblearn.over_sampling import SMOTE
smote=SMOTE()           #creating object for smote


# In[156]:


x_smote,y_smote=smote.fit_resample(x_train,y_train)       #fitting the training data to overcome imbalancing


# In[157]:


from collections import Counter          #importing counter
print('Actual Classes',Counter(y_train))    #printing actual classes
print('Smote Classes',Counter(y_smote))       #printing smote classes


# In[158]:


y_smote.value_counts()                #checking if the data is balanced or not


# # Logistic Regression

# In[159]:


from sklearn.linear_model import LogisticRegression              #importing logistic regression
LR=LogisticRegression()                         #object creation for logistic regression
LR.fit(x_train,y_train)                       #fitting training data


# In[160]:


y_pred=LR.predict(x_test)               #predicting x_test data
y_pred


# # Evaluation Metrics

# In[161]:


#importing evaluation metrics
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,recall_score,precision_score,classification_report


# In[162]:


acc=accuracy_score(y_test,y_pred)     #accuracy score of logistic regression
acc


# In[163]:


cm=confusion_matrix(y_test,y_pred)     #confusion matrix of logistic regression
cm


# In[164]:


f1=f1_score(y_test,y_pred)                #f1 score of logistic regression
f1


# In[165]:


re=recall_score(y_test,y_pred)           #recall score of logistic regression
re


# In[166]:


pr=precision_score(y_test,y_pred)         #precision score of logistic regression
pr


# In[167]:


print(classification_report(y_test,y_pred))        #classification report  of logistic regression


# # SVM

# In[168]:


from sklearn.svm import SVC                 #importing svc
svclassifier=SVC()                          #base model with default parameters
svclassifier.fit(x_smote,y_smote)           #fitting smoting data


# In[169]:


y_hat=svclassifier.predict(x_test)          #predicting x_test data
y_hat                                       


# # Evaluation Metrics for SVM Classifier

# In[170]:


from sklearn.metrics import accuracy_score,recall_score,precision_score,classification_report,f1_score,confusion_matrix
acc=accuracy_score(y_test,y_hat)                   #accuracy score of svm classifier
acc


# In[171]:


f1=f1_score(y_test,y_hat)                          #f1_score ofsvm classifier
f1


# In[172]:


re=recall_score(y_test,y_hat)                      #recall score of svm classifier
re


# In[173]:


pr=precision_score(y_test,y_hat)                   #precision score of svm classifier
pr


# In[174]:


cm=confusion_matrix(y_test,y_hat)                  #confusion matrix of svm classifier
cm


# In[175]:


from sklearn.model_selection import cross_val_score                     #importing cross val score
scores = cross_val_score(svclassifier,x,y,cv=3,scoring='f1')
print(scores)
print("Cross validation Score:",scores.mean())
print("Std :",scores.std())                                             #std score of <0.05 is good


# # Decision Tree

# In[176]:


from sklearn.tree import DecisionTreeClassifier     #importing decision tree classifier
dt=DecisionTreeClassifier()                  #creating an object for decision tree
dt.fit(x_train,y_train)                      #fitting the training data


# In[177]:


y_hat=dt.predict(x_test)                 #predicting the testing data
y_hat


# In[178]:


y_train_predict=dt.predict(x_train)               #predicting the training data
y_train_predict


# # Evaluation Metrics for Decision Tree

# In[179]:


from sklearn.metrics import accuracy_score,classification_report,f1_score,precision_score,recall_score
acc_train=accuracy_score(y_train,y_train_predict)             #accuracy of training data of decision tree classifier
acc_train


# In[180]:


f1_score=f1_score(y_train,y_train_predict)                  #f1_score of training data of decision tree
f1_score 


# In[181]:


precision_score=precision_score(y_train,y_train_predict)         #precision score of training data of decision tree
precision_score


# In[182]:


print(classification_report(y_train,y_train_predict))            #classification report for training data of decision tree


# In[183]:


#testing data accuracy
test_acc=accuracy_score(y_test,y_hat)
test_acc


# In[184]:


print(classification_report(y_test,y_hat))            #classification report of testing data of decision tree


# # Random Forest

# In[185]:


from sklearn.ensemble import RandomForestClassifier       #importing random forest classifier
rf_clf = RandomForestClassifier(n_estimators=100)         #creating an object for random forest classifier
rf_clf.fit(x_train,y_train)                               #fitting training data


# In[186]:


y_predict=rf_clf.predict(x_test)                          #predicting x_test data
y_predict                                       


# # Evaluation Metrics for Random Forest

# In[187]:


acc=accuracy_score(y_test,y_predict)                      #accuracy score of random forest classifier
acc


# In[188]:


print(classification_report(y_test,y_predict))             #classification report of random forest classifier


# # GB Classifier

# In[189]:


from sklearn.ensemble import GradientBoostingClassifier             #importing GB Classifier
gbm=GradientBoostingClassifier()                                    #object creation for GB Classifier
gbm.fit(x_train,y_train)                                            #fitting the data


# In[190]:


y_gbm=gbm.predict(x_test)                                          #predicting the price
y_gbm


# # Evaluation Metrics for GB Classifier

# In[191]:


from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,classification_report  #to check model performance
accuracy=accuracy_score(y_test,y_gbm)          #accuracy_score of GB Classifier
accuracy                               


# In[192]:


f1=f1_score(y_test,y_gbm)                      #f1_score of GB Classifier
f1


# In[193]:


prec=precision_score(y_test,y_gbm)             #precision_score of GB Classifier
prec


# In[194]:


rec=recall_score(y_test,y_gbm)                #recall_score of GB Classifier
rec


# In[195]:


print(classification_report(y_test,y_gbm))


# # XGB Classifier

# In[196]:


from xgboost import XGBClassifier                #importing XGB classifier
xgb_r=XGBClassifier()                            #object creation for XGB Classifier
xgb_r.fit(x_train,y_train)                       #fitting the data


# In[197]:


y_hat=xgb_r.predict(x_test)                       #predicting x_test data
y_hat


# # Evaluation Metrics of XGB Classifier

# In[198]:


from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,classification_report  #to check model performance
accuracy=accuracy_score(y_test,y_hat)          #accuracy_score of XGB Classifier
accuracy                               


# In[199]:


print(classification_report(y_test,y_hat))         #classification report of XGB Classifier


# # ANN

# In[200]:


from sklearn.neural_network import MLPClassifier    #importing ANN Classifier
model = MLPClassifier(hidden_layer_sizes=(50,3),learning_rate_init=0.1,max_iter=100,random_state=42) ## model object creation max_iter=Stopping parameter
model.fit(x_train,y_train)                         #training the data


# In[201]:


y_predict_proba=model.predict_proba(x_test) ## predicting the probaility of class
y_predict_proba


# In[202]:


y_predict=model.predict(x_test)                           #predicting x_test data
y_predict


# In[203]:


y_train_predict=model.predict(x_train)                     #predicting x_train data
y_train_predict


# # Evaluation Metrics Of ANN Classifier

# In[204]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report    #importing evaluation metrics
print("Train accuracy :",accuracy_score(y_train,y_train_predict))
print("Test accuracy :",accuracy_score(y_test,y_predict))


# In[205]:


print(classification_report(y_test,y_predict))             #classification_report of ANN Classifier


# # Hyper Parameter Tuning

# # GridSearchCV

# In[206]:


from sklearn.model_selection import GridSearchCV
  
# defining parameter range
param_grid = {'C': [0.1, 5, 10,50,60,70], 
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
             'random_state':(list(range(1, 20)))} 
model=SVC()
grid = GridSearchCV(model, param_grid, refit = True, verbose = 2,scoring='f1',cv=5)
  
# fitting the model for grid search
grid.fit(x,y)


# In[207]:


# print best parameter after tuning
print(grid.best_params_)


# In[208]:


# print best estimator after tuning
print(grid.best_estimator_)


# In[209]:


#clf=SVC(C=100, gamma=0.001,random_state=42) ##0.1
clf=SVC(C=5, gamma=0.1,random_state=1) ##0.1


# In[210]:


clf.fit(x_smote, y_smote)


# In[211]:


y_clf=clf.predict(x_test)
y_clf


# In[212]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report    #importing evaluation metrics
acc=accuracy_score(y_test,y_clf)
acc


# In[213]:


f1=f1_score(y_test,y_clf)
f1


# In[214]:


print(classification_report(y_test,y_clf))


# In[215]:


scores_after = cross_val_score(clf,x,y,cv=3,scoring='f1')
print(scores_after)
print("Cross validation Score:",scores_after.mean())
print("Std :",scores.std())
#std of < 0.05 is good. 


# # RandomSearchCV

# In[216]:


from sklearn.model_selection import RandomizedSearchCV
  
# defining parameter range
param_random = {'C': [0.1, 5, 10,50,60,70], 
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
             'random_state':(list(range(1, 20)))} 
model=SVC()
random = RandomizedSearchCV(model, param_grid, refit = True, verbose = 2,scoring='f1',cv=5)
  
# fitting the model for grid search
random.fit(x,y)


# In[217]:


# print best parameter after tuning
print(random.best_params_)


# In[218]:


# print best estimator after tuning
print(random.best_estimator_)


# In[219]:


#clf=SVC(C=100, gamma=0.001,random_state=42) ##0.1
clf=SVC(C=5, gamma=0.1,random_state=1) ##0.1


# In[220]:


clf.fit(x_smote, y_smote)


# In[221]:


y_clf=clf.predict(x_test)
y_clf


# In[222]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report    #importing evaluation metrics
acc=accuracy_score(y_test,y_clf)
acc


# In[223]:


f1=f1_score(y_test,y_clf)
f1


# In[224]:


print(classification_report(y_test,y_clf))


# In[225]:


scores_after = cross_val_score(clf,x,y,cv=3,scoring='f1')
print(scores_after)
print("Cross validation Score:",scores_after.mean())
print("Std :",scores.std())
#std of < 0.05 is good. 


# # Comparison Table of the Training Models

# In[226]:


from prettytable import PrettyTable
x=PrettyTable()


# In[227]:


x.field_names = ["Model","accuracy_score"]
x.add_row(["Logistic Regression","86.6%"])
x.add_row(["SVM Classifier","84%"])
x.add_row(["Decision Tree Classifier","78%"])
x.add_row(["Random Forest Classifier","84.4%"])
x.add_row(["GB Classifier", "84%"])
x.add_row(["XGB Classifier","89%"])
x.add_row(["ANN Classifier","51%"])


# In[228]:


print(x)


# # Conclusion Report of Training Models
# - Based on the given values, it appears that the XGB Classifier has the highest accuracy of 89% for heart disease prediction.
# - followed by Logistic Regression with an accuracy of 86.6% and Random Forest Classifier with an accuracy of 84.4%. 
# - The SVM Classifier, GB Classifier, and Decision Tree Classifier have accuracies of 84%, 84%, and 78% respectively. 
# - However, the ANN Classifier has the lowest accuracy of 51%.

# In[ ]:




