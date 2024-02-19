#!/usr/bin/env python
# coding: utf-8

# In[1]:


pwd


# In[2]:


cd C:\Users\Radhika Gupta\OneDrive\Documents\


# In[3]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score , classification_report , confusion_matrix
df = pd.read_csv('data.csv')
print(df)


# In[4]:


print("Null values in the dataset")
print(df.isnull().sum())


# In[5]:


df.dropna(inplace=True)
x = df.drop('target' , axis=1)
y = df['target']
print(df)


# In[6]:


print("Null values in the dataset:")
print(df.isnull().sum())


# In[7]:


x = df.drop('target', axis=1)
y = df['target']


# In[8]:


x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.2,random_state = 42)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
model = LogisticRegression(random_state = 42)
model.fit(x_train_scaled , y_train)
y_pred = model.predict(x_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)


# In[9]:


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
corr_matrix = df.corr()
plt.figure(figsize = (12,8))
sns.heatmap(corr_matrix , annot=True , cmap='coolwarm' , fmt=".2f")
plt.title('Correlation Matrix Heatmap')
plt.show()


# In[10]:


target_correlation = corr_matrix['target'].sort_values(ascending=False)


# In[11]:


print("Variables with the highest correlation with 'target':")
print(target_correlation)


# In[12]:


sns.histplot(data=df , x='chest pain' , hue = 'target' , multiple='stack' , bins=4)
plt.title('Distribution of Chest Pain by Target')
print('Visualization of Chest Pain and its relationship with Target')
plt.show()
      
sns.histplot(data=df , x='slope' , hue = 'target' , multiple='stack' , bins=4)
plt.title('Distribution of Slope by Target')
print('Visualization of Slope and its relationship with Target')
plt.show()
      


# In[13]:


sns.histplot(data=df, x='maximun heart rate achieved', hue='target', multiple='stack', bins=20)
plt.title('Distribution of Maximum Heart Rate Achieved by Target')
plt.xlabel('Maximum Heart Rate Achieved')
plt.ylabel('Count')
print('Visualization of Maximum Heart Rate Achieved and its Relationship with Target')
plt.show()


# In[14]:


pip install nbconvert[webpdf]


# In[17]:


sns.histplot(data=df , x='exercise induced angina (1:yes, 0:no)' , hue = 'target' , multiple='stack' , bins=10)
plt.title('Exercise-induced angina by Target')
print('Visualization of Exercise-induced angina and its relationship with Target')
plt.show()
      
sns.histplot(data=df , x='oldpeak' , hue = 'target' , multiple='stack' , bins=10)
plt.title('Distribution of Oldpeak by Target')
print('Visualization of Oldpeak and its relationship with Target')
plt.show()

sns.histplot(data=df , x='ca' , hue = 'target' , multiple='stack' , bins=10)
plt.title('CA [number of major vessels colored by fluoroscopy] by Target')
print('Visualization of CA and its relationship with Target')
plt.show()


# In[18]:


sns.countplot(x='sex', data=df, hue='target')
plt.title('Distribution of Heart Disease by Gender')
plt.show()


# In[26]:


sns.countplot(x='sex', data=df, hue='target', palette='dark')
plt.title('Distribution of Heart Disease by Gender')
plt.xlabel('Gender (0: Female, 1: Male)')
plt.show()


# In[ ]:





# In[25]:


male_heart_disease = df[(df['sex'] == 1) & (df['target'] == 1)].shape[0]
female_heart_disease = df[(df['sex'] == 0) & (df['target'] == 1)].shape[0]
total_males = df[df['sex'] == 1].shape[0]
total_females = df[df['sex'] == 0].shape[0]
print(f"Percentage of males with heart disease: {100 * male_heart_disease / total_males:.2f}%")
print(f"Percentage of females with heart disease: {100 * female_heart_disease / total_females:.2f}%")


# In[27]:


# Analyze chest pain types with vibrant colors
sns.countplot(x='chest pain', data=df, hue='target', palette='viridis')
plt.title('Distribution of Heart Disease by Chest Pain Type')
plt.xlabel('Chest Pain Type (0-3)')
plt.show()

for i in range(4):  
    total_with_pain = df[df['chest pain'] == i].shape[0]
    with_heart_disease = df[(df['chest pain'] == i) & (df['target'] == 1)].shape[0]
    
    percentage_with_heart_disease = 100 * with_heart_disease / total_with_pain if total_with_pain > 0 else 0
    
    print(f"Percentage of individuals with Chest Pain Type {i} and heart disease: {percentage_with_heart_disease:.2f}%")


# In[30]:


sns.scatterplot(x='age', y='maximun heart rate achieved', data=df, hue='target', palette='muted')
plt.title('Scatter Plot of Age vs. Max Heart Rate by Heart Disease Presence')
plt.xlabel('Age')
plt.ylabel('maximun heart rate achieved')
plt.show()

with_heart_disease = df[df['target'] == 1]
without_heart_disease = df[df['target'] == 0]

avg_age_with_heart_disease = with_heart_disease['age'].mean()
avg_max_heart_rate_with_heart_disease = with_heart_disease['maximun heart rate achieved'].mean()

avg_age_without_heart_disease = without_heart_disease['age'].mean()
avg_max_heart_rate_without_heart_disease = without_heart_disease['maximun heart rate achieved'].mean()

print(f"Average Age with Heart Disease: {avg_age_with_heart_disease:.2f}")
print(f"Average Max Heart Rate with Heart Disease: {avg_max_heart_rate_with_heart_disease:.2f}\n")
print(f"Average Age without Heart Disease: {avg_age_without_heart_disease:.2f}")
print(f"Average Max Heart Rate without Heart Disease: {avg_max_heart_rate_without_heart_disease:.2f}")


# In[33]:


sns.countplot(x='exercise induced angina (1:yes, 0:no)', data=df, hue='target', palette='Set2')
plt.title('Distribution of Heart Disease by Exercise-Induced Angina')
plt.xlabel('Exercise-Induced Angina (0: No, 1: Yes)')
plt.show()

total_no_angina = df[df['exercise induced angina (1:yes, 0:no)'] == 0].shape[0]
total_with_angina = df[df['exercise induced angina (1:yes, 0:no)'] == 1].shape[0]

with_heart_disease_no_angina = df[(df['exercise induced angina (1:yes, 0:no)'] == 0) & (df['target'] == 1)].shape[0]
with_heart_disease_with_angina = df[(df['exercise induced angina (1:yes, 0:no)'] == 1) & (df['target'] == 1)].shape[0]

percentage_with_heart_disease_no_angina = 100 * with_heart_disease_no_angina / total_no_angina if total_no_angina > 0 else 0
percentage_with_heart_disease_with_angina = 100 * with_heart_disease_with_angina / total_with_angina if total_with_angina > 0 else 0

print(f"Percentage of individuals without exercise-induced angina and with heart disease: {percentage_with_heart_disease_no_angina:.2f}%")
print(f"Percentage of individuals with exercise-induced angina and with heart disease: {percentage_with_heart_disease_with_angina:.2f}%")


# In[34]:


sns.histplot(x='age', data=df, hue='target', kde=True, bins=30, palette='Set3')
plt.title('Age Distribution by Heart Disease Presence')
plt.xlabel('Age')
plt.show()

with_heart_disease = df[df['target'] == 1]
without_heart_disease = df[df['target'] == 0]

avg_age_with_heart_disease = with_heart_disease['age'].mean()
avg_age_without_heart_disease = without_heart_disease['age'].mean()

print(f"Average Age with Heart Disease: {avg_age_with_heart_disease:.2f}")
print(f"Average Age without Heart Disease: {avg_age_without_heart_disease:.2f}")


# In[ ]:





# In[ ]:




