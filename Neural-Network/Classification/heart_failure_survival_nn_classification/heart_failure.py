#!/usr/bin/env python
# coding: utf-8

# In[315]:


# Loading te data #
## 1. and 2. Load as pandas DataFrame and visualise
import pandas as pd
data = pd.read_csv("heart_failure.csv")


# In[316]:


### See all columns and their types
data.info()


# In[317]:


### See the statistical data of each column
data.describe()


# In[318]:


### See a snippet of data
data.head(5)


# In[319]:


## 3. Print ditribution of death_event
from collections import Counter
counter_death_event = Counter(data["death_event"])
print('Classes and number of values in the dataset: ', counter_death_event)


# In[320]:


## 4. Assign label
y = data.pop("death_event")
y.head(5)


# In[321]:


## 5. Extract useful features
X = data.loc[:, ['age','anaemia','creatinine_phosphokinase','diabetes','ejection_fraction','high_blood_pressure','platelets','serum_creatinine','serum_sodium','sex','smoking','time']]
X.head(5)


# In[322]:


# Data Preprocessing
## 6. One-hot encoding for categorical features
cat_cols = X.select_dtypes(include = 'object').columns.tolist()
print(cat_cols)

X = pd.get_dummies(X, columns = cat_cols)
print(X.columns.tolist())


# In[323]:


## 7. Split the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size = 0.20, train_size = 0.80, random_state = 42
  )


# In[324]:


### Apply SMOTE as the data is imbalanced
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)

# Use y_train before one-hot encoding
X_train, y_train_res = smote.fit_resample(X_train, y_train)

print("New class distribution after SMOTE:", Counter(y_train_res))


# In[325]:


## 8. and 9. Scale the numeric features for normalisation
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

### get numeric column names
numeric_cols = X.select_dtypes(include = ["int64", "float64"]).columns.tolist()

### define the tranformer (necessary for ColumnTransformer)
transformer = [("My_Standard_Scaler",  StandardScaler(), numeric_cols)]

### set the column transformer settings
my_ct = ColumnTransformer(transformer, remainder='passthrough', verbose=False) 

### fit the scaler and transform data
X_train_scaled = my_ct.fit_transform(X_train)
X_test_scaled = my_ct.transform(X_test)


# In[326]:


### Visualise the normalised data
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns.tolist())
X_train_scaled_df.head(5)


# In[327]:


# Prepare labels for classification #
## 11. Initialise label encoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
print("un-processed labels: \n", y_train[0:4])

## 12. and 13. Fit the encoder and transform labels
y_train = le.fit_transform(y_train_res)
y_test = le.transform(y_test)

## 14. and 15. Convert encoded labels into binary vector
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print("label-encoded, categorical labels: \n", y_train[0:4])


# In[328]:


# Design the model
## 16. Initialise a sequential model
from tensorflow.keras.models import Sequential
my_model = Sequential()

## 17. Create and add the input layer to the model
from tensorflow.keras.layers import InputLayer
my_model.add(
  InputLayer(
    shape=(X_train_scaled.shape[1], ),
    name="Input_Layer",   
  )
)

## 18. Create and add the hidden dense layer(s) to the model
from tensorflow.keras.layers import Dense
from tensorflow.keras import regularizers

my_model.add(
  Dense(
    units = 8,
    activation = 'relu',    
    use_bias = True,
    kernel_regularizer = regularizers.L1L2(0.01),
    name = "Hidden_Layer1"
  )
)

## 19. Create and add the output layer to the model
my_model.add(
  Dense(
    units = len(y_train[1]),
    activation = 'softmax',    
    use_bias = True,
    kernel_regularizer = regularizers.L1L2(0.001),
    name = "Output_Layer"
  )
)

## Visualise the model
print(my_model.summary())


# In[329]:


## 20. Compile the model
### Create Optimizer
from tensorflow.keras.optimizers import Adam
my_optimizer = Adam(learning_rate=0.001)

### Define Loss
from tensorflow.keras.losses import CategoricalCrossentropy
my_loss = CategoricalCrossentropy()

### Compile
from tensorflow.keras.metrics import AUC, Precision, Recall

my_model.compile(
    optimizer = my_optimizer,
    loss = my_loss,
    metrics = [
        'categorical_accuracy',
        AUC(name='auc'),
        Precision(name='precision'),
        Recall(name='recall')
    ]
)


# In[330]:


# Train and ealuate model # 
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

## 21. Train the model
history = my_model.fit(
              x=X_train_scaled,
              y=y_train,
              batch_size=2,
              epochs=100,
              verbose=1,
              callbacks=[early_stop], # can add layter on
              validation_data=(X_test_scaled, y_test),
              shuffle=True
              )


# In[331]:


import matplotlib.pyplot as plt

# Create subplots: 1 row, 2 columns
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# Plot Loss
axs[0].plot(history.history['loss'], label='Training Loss')
axs[0].plot(history.history['val_loss'], label='Validation Loss')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')
axs[0].set_title('Loss over Epochs')
axs[0].legend()
axs[0].grid(True)

# Plot Categorical Accuracy
axs[1].plot(history.history['categorical_accuracy'], label='Training Accuracy')
axs[1].plot(history.history['val_categorical_accuracy'], label='Validation Accuracy')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Categorical Accuracy')
axs[1].set_title('Accuracy over Epochs')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()


# In[332]:


## 22. Evaluate the model
loss, acc, auc, prec, recall = my_model.evaluate(
                    x=X_test_scaled,
                    y=y_test,
                    batch_size=2,
                    verbose=0
                    )
print(f"Loss = {loss:.4f}")
print(f"Accuracy = {acc*100:.2f}%")
print(f"AUC = {auc:.4f}, Precision = {prec:.4f}, Recall = {recall:.4f}")


# In[337]:


# Generate Classification report
## 23. Predictions
y_estimate = my_model.predict(x = X_test_scaled)

## 24. - 26. Print classification report
import numpy as np
from sklearn.metrics import classification_report

y_estimate = np.argmax(y_estimate, axis = 1)
y_true = np.argmax(y_test, axis = 1)

print(classification_report(y_true, y_estimate))
print("Note: Class 0 -> Survival, Class 1 -> Death")


# Interpretation of each metric:
# Precision for Class 0 (Survival):
# Out of all instances predicted as survival, 76% were actually survival.
# 
# Recall for Class 0:
# The model correctly identified 89% of all survival cases (high recall means few false negatives for survival).
# 
# Precision for Class 1 (Death):
# Out of all instances predicted as death, 79% were actually death.
# 
# Recall for Class 1:
# The model correctly identified 60% of all death cases (moderate recall means it missed 40% of deaths).

# In[336]:


## ROC curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np

# Assuming y_test and y_estimate are already defined as one-hot and probabilities respectively:

# If y_test is one-hot encoded, get the true labels for positive class (e.g. class 1)
y_true = y_test[:, 1]

# Get predicted probabilities for positive class from your model
y_scores = my_model.predict(X_test_scaled)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_scores)

# Calculate AUC
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line (random guess)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

