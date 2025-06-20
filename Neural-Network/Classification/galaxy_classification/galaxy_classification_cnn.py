#!/usr/bin/env python
# coding: utf-8

# In[116]:


import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from utils import load_galaxy_data


# In[117]:


## Use the custom function to load the data
from utils import load_galaxy_data
input_data, labels = load_galaxy_data()


# In[118]:


## Check the shape of data
print("Input Data Shape: ", input_data.shape)
print("Labels Shape: ", labels.shape)

### There are 1400 images which are 128x128 and RGB [0, 255]
### There are 4 classes, in the form [0,1,0,0], etc.


# In[119]:


## Divide the data into training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    input_data, 
    labels, 
    test_size=0.20, 
    random_state=222, 
    shuffle=True, 
    stratify=labels ## This ensures that ratios of galaxies in your testing data will be the same as in the original dataset.
    )


# In[120]:


## Normalise pixel values to [0, 1]
# X_train = X_train / 255.0
# X_test = X_test / 255.0

# print(f"X_train min: {X_train.min()}, max: {X_train.max()}")
# print(f"X_test min: {X_test.min()}, max: {X_test.max()}")

## OR add a layer that can do it inside the model
Rescaling_Pipeline = tf.keras.layers.Rescaling(scale=1./255)


# In[121]:


### Create tf.data Dataset in order to apply transformations (like batching or augmentation) to this dataset.
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))


# In[122]:


## Shuffle the training data to prevent learning patterns based on input order.
train_ds = train_ds.shuffle(buffer_size=train_ds.cardinality()) ## shuffle all the data


# In[123]:


## data augmentation
from tensorflow.keras import Sequential
from tensorflow.keras.layers import RandomRotation, RandomZoom

Augmentation_Pipeline = Sequential([
    RandomRotation(0.1),                                # ±10% of a full circle (~±18 degrees)
    RandomZoom(height_factor=0.1, width_factor=0.1),    # Zoom in/out up to 10%
])

def rescale_augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = Augmentation_Pipeline(image)
    image = Rescaling_Pipeline(image)
    return image, label

### Note: .map() applies transformations at the time the dataset is iterated, which typically happens during training or validation.
train_ds = train_ds.map(rescale_augment, num_parallel_calls=tf.data.AUTOTUNE)


# In[124]:


## resclae test data only:
def rescale_only(image, label):
    image = Rescaling_Pipeline(image)
    return image, label

test_ds = test_ds.map(rescale_only, num_parallel_calls=tf.data.AUTOTUNE)


# In[125]:


## Batch the data for efficiency
batch_size = 5
train_ds = train_ds.batch(batch_size)
test_ds = test_ds.batch(batch_size)


# In[126]:


## Performance optimisation
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)


# In[127]:


## Build the model
from tensorflow.keras import layers, models

def create_model(input_shape=(128, 128, 3), num_classes=4):
    model = models.Sequential([
        layers.Input(shape=input_shape),

        # Block 1
        layers.Conv2D(filters = 16, kernel_size = (3, 3), strides=(1, 1), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPool2D(pool_size=(2, 2)),

        # Block 2
        layers.Conv2D(32, (3, 3), strides=(2, 2), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPool2D(pool_size=(2, 2)),

        # Block 3
        layers.Conv2D(64, (3, 3), strides=(2, 2), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPool2D(pool_size=(2, 2)),

        # Flatten & Dense Layers
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.1),

        # Output Layer
        layers.Dense(num_classes, activation='softmax')
    ])

    return model

model = create_model()
model.summary()


# In[128]:


### Create Optimizer
from tensorflow.keras.optimizers import Adam
my_optimizer = Adam(learning_rate=0.001)

### Define Loss
from tensorflow.keras.losses import CategoricalCrossentropy
my_loss = CategoricalCrossentropy()

### Compile
from tensorflow.keras.metrics import AUC, Precision, Recall

model.compile(
    optimizer = my_optimizer,
    loss = my_loss,
    metrics = [
        'categorical_accuracy',
        AUC(name='auc'),
        Precision(name='precision'),
        Recall(name='recall')
    ]
)


# In[129]:


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    "best_model.keras",
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=50,
    verbose=1,
    callbacks=[early_stop, checkpoint, reduce_lr]
)


# In[135]:


## plot the training
import matplotlib.pyplot as plt
import numpy as np

def plot_history(history):
    metrics = ['loss', 'categorical_accuracy', 'auc', 'precision', 'recall']
    
    # Find epoch with lowest val_loss (best model)
    best_epoch = np.argmin(history.history['val_loss'])
    
    for metric in metrics:
        plt.plot(history.history[metric], label=f'Train {metric}')
        plt.plot(history.history[f'val_{metric}'], label=f'Val {metric}')
        
        best_val_metric = history.history[f'val_{metric}'][best_epoch]
        best_train_metric = history.history[metric][best_epoch]
        
        # Circle the best validation metric point
        plt.scatter(best_epoch, best_val_metric, s=100, facecolors='none', edgecolors='r', label='Best model')
        
        # Print the values as text near the circle (adjust position a bit)
        plt.text(best_epoch, best_val_metric, 
                 f"Test: {best_val_metric:.4f}\nTrain: {best_train_metric:.4f}", 
                 fontsize=9, color='red', verticalalignment='bottom', horizontalalignment='right')
        
        plt.title(metric.upper())
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.legend()
        plt.show()

plot_history(history)

