#!/usr/bin/env python
# coding: utf-8

# # Data loading and observing #
# 

# In[145]:


## 1. Load the dataset into a pandas DataFrame
import pandas as pd
dataset = pd.read_csv("life_expectancy.csv")
dataset.columns = dataset.columns.str.strip()


# In[146]:


## 2. Observe the data
print(dataset.head())
print("#"*70)
print(dataset.describe())
#print(len(dataset.columns))


# In[147]:


## 3. Drop the Country column for generalisation
dataset.pop("Country")
print(dataset.head())


# In[148]:


## 4. Assign Labels
labels = dataset.loc[:,"Life expectancy"]
print(labels.head())


# In[149]:


## 5. Assign Features
features = dataset.iloc[:,:-1]
print(features.head())
#print(features.columns)


# # Data Preprocessing #
# 

# In[150]:


## 6. Apply one-hot-encoding on all the categorical columns
features = pd.get_dummies(features)
print(features.head())
print(features.columns.tolist())


# In[151]:


## Identify redundant features
import seaborn as sns
sns.heatmap(features.corr(), annot=False, cmap="coolwarm")


# In[152]:


## Drop Collinear features
features = features.drop(columns=[
    'percentage expenditure', 
    'infant deaths', 
    'thinness 5-9 years'
])
print(features.columns.tolist())


# In[153]:


## 7. Split the data
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, train_size=0.8, random_state=42)


# In[154]:


## 8. and 9. Standardize/Normalize numerical features
from sklearn.compose import ColumnTransformer as ct
from sklearn.preprocessing import MinMaxScaler

### Get the numerical column names
numeric_columns = features.select_dtypes(include=['float64', 'int64'])
numeric_column_names = numeric_columns.columns.tolist()

### Define the scaler and which columns to act on
transformers = [( "normalize_nums", 
                MinMaxScaler(feature_range=(0, 1), copy=False),  
                numeric_column_names
                )]
normaliser_ct = ct( transformers,        
                    remainder='passthrough'
                    )

### Normalise the features
features_train_scaled = normaliser_ct.fit_transform(features_train)
features_test_scaled = normaliser_ct.transform(features_test)

### Visualise the normalised data
features_train_scaled_df = pd.DataFrame(features_train_scaled, columns=features.columns.tolist())
print(features_train_scaled_df.head())


# # Building the model #
# 

# In[155]:


## 11. Create instance of a Sequential Model
from tensorflow.keras.models import Sequential
my_model = Sequential()


# In[156]:


## 12. and 13. Create and add the input layer
from tensorflow.keras.layers import InputLayer
my_model.add(
  InputLayer(
    input_shape = (features.shape[1], ),
    name = "Input_Layer"
    ) 
  )


# In[157]:


## 14. Add hidden dense layer(s)
from tensorflow.keras.layers import Dense
from tensorflow.keras import regularizers
from tensorflow.keras import activations

my_model.add(
  Dense(
    128,
    activation='relu',
    use_bias=True,
    kernel_regularizer=regularizers.l1(0.01),
    name = "Hidden_Layer1"
    )
)

my_model.add(
  Dense(
    64,
    activation='relu',
    use_bias=True,
    kernel_regularizer=regularizers.l1(0.001),
    name = "Hidden_Layer2"
    )
)


# In[158]:


## 15. Add output dense layer
my_model.add(
  Dense(
    1,
    activation='linear',
    use_bias=True,
    kernel_regularizer=regularizers.l1(0.001),
    name = "Output_Layer"
    )
)


# In[159]:


## 16. Print Model summary
print(my_model.summary())


# # Initializing the optimizer and compiling the model #

# In[160]:


## 17. Create Optimizer
from tensorflow.keras.optimizers import Adam
my_optimizer = Adam(learning_rate=0.01)


# In[161]:


## 18. Compile model
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsoluteError
my_model.compile(
    optimizer = my_optimizer,
    loss = 'mse',
    metrics = [MeanAbsoluteError(), RootMeanSquaredError()],
)


# # Fit and evaluate model #
# 

# In[162]:


## Add callbacks
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

early_stop = EarlyStopping(patience=20, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(factor=0.50, patience=5)


# In[163]:


## 19. Train model 
history = my_model.fit(
              x = features_train_scaled,
              y = labels_train,
              batch_size = 2,
              epochs = 80,
              verbose = 1,
              shuffle = True,
              validation_data = (features_test_scaled, labels_test),
              callbacks=[early_stop, reduce_lr]
              )


# In[167]:


import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss [MSE]')
plt.title('Loss over Epochs')
plt.legend()
plt.grid(True)
plt.show()


# In[168]:


## 20. Evaluate Model
res_mse, res_mae, res_rmse = my_model.evaluate(
                                  x = features_test_scaled,
                                  y = labels_test,
                                  batch_size = 2,
                                  verbose = 0
                                  )

print(f"RMSE = {res_rmse}, MAE = {res_mae}") 


# In[169]:


import seaborn as sns

y_pred = my_model.predict(features_test_scaled)
sns.histplot(labels_test, color='blue', label='Actual', kde=True)
sns.histplot(y_pred.flatten(), color='orange', label='Predicted', kde=True)
plt.legend()
plt.title("Life Expectancy: Actual vs Predicted Distribution")
plt.show()


# 
