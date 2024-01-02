#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics


# In[8]:


l1=['back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
    'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
    'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
    'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs',
    'fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
    'irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs',
    'swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
    'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips',
    'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
    'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
    'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine',
    'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
    'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',
    'abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum',
    'rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion',
    'receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen',
    'history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf',
    'palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
    'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose',
    'yellow_crust_ooze']


# In[9]:


disease=['Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis',
       'Drug Reaction', 'Peptic ulcer diseae', 'AIDS', 'Diabetes ',
       'Gastroenteritis', 'Bronchial Asthma', 'Hypertension ', 'Migraine',
       'Cervical spondylosis', 'Paralysis (brain hemorrhage)', 'Jaundice',
       'Malaria', 'Chicken pox', 'Dengue', 'Typhoid', 'hepatitis A',
       'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E',
       'Alcoholic hepatitis', 'Tuberculosis', 'Common Cold', 'Pneumonia',
       'Dimorphic hemmorhoids(piles)', 'Heart attack', 'Varicose veins',
       'Hypothyroidism', 'Hyperthyroidism', 'Hypoglycemia',
       'Osteoarthristis', 'Arthritis',
       '(vertigo) Paroymsal  Positional Vertigo', 'Acne',
       'Urinary tract infection', 'Psoriasis', 'Impetigo']


# In[18]:


l2=[]
for i in range(0,len(l1)):
    l2.append(0)
print(l2)


# In[19]:


# Load the training data
df = pd.read_csv("training.csv")


# In[20]:


disease_mapping = {
    'Fungal infection': 0, 'Allergy': 1, 'GERD': 2, 'Chronic cholestasis': 3, 'Drug Reaction': 4,
    'Peptic ulcer diseae': 5, 'AIDS': 6, 'Diabetes ': 7, 'Gastroenteritis': 8, 'Bronchial Asthma': 9,
    'Hypertension ': 10, 'Migraine': 11, 'Cervical spondylosis': 12, 'Paralysis (brain hemorrhage)': 13,
    'Jaundice': 14, 'Malaria': 15, 'Chicken pox': 16, 'Dengue': 17, 'Typhoid': 18, 'hepatitis A': 19,
    'Hepatitis B': 20, 'Hepatitis C': 21, 'Hepatitis D': 22, 'Hepatitis E': 23, 'Alcoholic hepatitis': 24,
    'Tuberculosis': 25, 'Common Cold': 26, 'Pneumonia': 27, 'Dimorphic hemmorhoids(piles)': 28,
    'Heart attack': 29, 'Varicose veins': 30, 'Hypothyroidism': 31, 'Hyperthyroidism': 32, 'Hypoglycemia': 33,
    'Osteoarthristis': 34, 'Arthritis': 35, '(vertigo) Paroymsal  Positional Vertigo': 36, 'Acne': 37,
    'Urinary tract infection': 38, 'Psoriasis': 39, 'Impetigo': 40
}


# In[21]:


df['prognosis'].replace(disease_mapping, inplace=True)


# In[22]:


df.head(10)


# ## Testing Data

# In[23]:


tr=pd.read_csv("testing.csv")

tr.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
    'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
    'Migraine':11,'Cervical spondylosis':12,
    'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
    'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
    'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
    'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
    '(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
    'Impetigo':40}},inplace=True)
tr.head()


# ## Visualization of Data
# ### Frequency Distribution of Key Features

# In[58]:


def plot_per_column_distribution(df, n_graph_shown, n_graph_per_row):
    n_unique = df.nunique()
    df = df[[col for col in df if n_unique[col] > 1 and n_unique[col] < 50]]
    n_row, n_col = df.shape
    column_names = list(df)
    n_graph_row = (n_col + n_graph_per_row - 1) // n_graph_per_row

    # Set seaborn style
    sns.set(style="whitegrid", palette="muted")

    plt.figure(num=None, figsize=(6 * n_graph_per_row, 8 * n_graph_row), dpi=80, facecolor='w', edgecolor='k')
    for i in range(min(n_col, n_graph_shown)):
        plt.subplot(n_graph_row, n_graph_per_row, i + 1)
        column_df = df.iloc[:, i]
        if not pd.api.types.is_numeric_dtype(column_df):
            value_counts = column_df.value_counts()
            sns.barplot(x=value_counts.index, y=value_counts.values, palette="viridis")
        else:
            sns.histplot(column_df, color="skyblue", kde=False)
        plt.ylabel('Counts')
        plt.xticks(rotation=90)
        plt.title(f'{column_names[i]} (column {i})')
    plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
    plt.show()


# In[59]:


plot_per_column_distribution(df, 10, 5)


# In[26]:


plot_per_column_distribution(tr, 10, 5)


# ### Scatter Plot Matrix for Multivariate Analysis

# In[27]:


def plot_scatter_matrix(df, plot_size, text_size):
    df = df.select_dtypes(include=[pd.np.number])
    df = df[[col for col in df if df[col].nunique() > 1]]
    column_names = list(df)
    if len(column_names) > 10:
        column_names = column_names[:10]
    df = df[column_names]
    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plot_size, plot_size], diagonal='kde')
    corrs = df.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k=1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=text_size)
    plt.suptitle('Scatter and Density Plot')
    plt.show()


# In[28]:


plot_scatter_matrix(df, 20, 10)


# In[29]:


plot_scatter_matrix(tr, 20, 10)


# In[30]:


X_test= tr[l1]
y_test = tr[["prognosis"]]
np.ravel(y_test)
print(X_test)


# In[31]:


y_test = np.ravel(y_test)


# In[32]:


print(y_test)


# In[33]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df[l1], df['prognosis'], test_size=0.2, random_state=42)


# In[34]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


# In[35]:


# Decision Tree
clf_dt = DecisionTreeClassifier()
clf_dt.fit(X_train, y_train)


# In[36]:


# Random Forest
clf_rf = RandomForestClassifier()
clf_rf.fit(X_train, y_train)


# In[37]:


# KNearestNeighbour
clf_knn = KNeighborsClassifier()
clf_knn.fit(X_train, y_train)


# In[38]:


# Naive Bayes
clf_nb = GaussianNB()
clf_nb.fit(X_train, y_train)


# In[39]:


from sklearn.metrics import accuracy_score, precision_score, classification_report

def evaluate_model(clf, X, y):
    y_pred = clf.predict(X)
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='weighted')
    report = classification_report(y, y_pred)
    
    return accuracy, precision, report


# In[40]:


# Decision Tree
acc_dt, prec_dt, report_dt = evaluate_model(clf_dt, X_test, y_test)


# In[41]:



# Random Forest
acc_rf, prec_rf, report_rf = evaluate_model(clf_rf, X_test, y_test)


# In[42]:


# KNearestNeighbour
acc_knn, prec_knn, report_knn = evaluate_model(clf_knn, X_test, y_test)


# In[43]:


# Naive Bayes
acc_nb, prec_nb, report_nb = evaluate_model(clf_nb, X_test, y_test)


# In[44]:


print("Decision Tree Metrics:")
print(f"Accuracy: {acc_dt}")
print(f"Precision: {prec_dt}")
print("Classification Report:\n", report_dt)


# In[45]:


print("\nRandom Forest Metrics:")
print(f"Accuracy: {acc_rf}")
print(f"Precision: {prec_rf}")
print("Classification Report:\n", report_rf)


# In[46]:


print("\nKNearestNeighbour Metrics:")
print(f"Accuracy: {acc_knn}")
print(f"Precision: {prec_knn}")
print("Classification Report:\n", report_knn)


# In[47]:


print("\nNaive Bayes Metrics:")
print(f"Accuracy: {acc_nb}")
print(f"Precision: {prec_nb}")
print("Classification Report:\n", report_nb)


# In[ ]:




