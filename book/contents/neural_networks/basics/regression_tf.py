# %% [markdown]
"""
# Regression - Tensorflow
"""

# %%
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
print(tf.__version__)

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(precision=3, suppress=True)
import pandas as pd
import seaborn as sns


# %% [markdown]
"""
## Create Data
"""

# %%
np.random.seed(0)
cols = "abcdefghijk"
df = pd.DataFrame({
    col:np.random.normal(0,1,100)
    for col in cols
})

coefs = np.round(np.random.uniform(2,10,len(cols)),0)
print(coefs)
df["y"] = np.array(df) @ coefs

# %%
df.head()

# %% [markdown]
"""
## Split Train/Test
"""

# %%
train, test = train_test_split(df, test_size=0.2, random_state=0)
train_labels = train.pop("y")
test_labels = test.pop("y")

# %% [markdown]
"""
## Single Layer 

"Dense" layer implements `activation(dot(input, kernel) + bias)`

- activation is the element-wise activation function  
- kernel is the weights matrix created by the layer  
- bias is applicable if `use_bias` is `True`

The model is:

$$y = WX + b$$

- y is [1,80]   
- W is [1, 11]  
- X is [11, 80]  
- b is [1, 1], duplicated to [1,80] with broadcasting  
"""

# %%
# build the model
# normalization optional for this simple model

# normalizer = layers.Normalization(axis=-1)
linear_model = tf.keras.Sequential([
    # normalizer,
    layers.Dense(units=1, use_bias=True)
])

# print summary of model (need to build first to print summary)
linear_model.predict(train)
linear_model.summary()

# %%
# configure training procedure
linear_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error'
)

# %%
%%time
# train model
epochs = 100
history = linear_model.fit(
    train,
    train_labels,
    epochs=epochs,
    # Suppress logging.
    verbose=0,
    # Calculate validation results on 20% of the training data.
    validation_split = 0.2
)

# %%
dfhist = pd.DataFrame(history.history)
dfhist["epoch"] = history.epoch
sns.lineplot(dfhist["epoch"],dfhist["loss"])
sns.lineplot(dfhist["epoch"],dfhist["val_loss"])

# %%
# predict using
# linear_model.predict(train)

# %%
# compare weights of dense layer vs actual coefficients
# note that if normalizing, dense layer is in linear_model.layers[1]
print(linear_model.layers[0].kernel)
print(coefs)

# %% [markdown]
"""
## Deep Neural Network

Simply add more layers when building model.

Note that weights won't reflect coefficient values anymore.
"""

# %%
%%time

# normalizer = layers.Normalization(axis=-1)
linear_model = tf.keras.Sequential([
      layers.Dense(64, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(units=1, use_bias=True)
])

linear_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error'
)

epochs = 100
history = linear_model.fit(
    train,
    train_labels,
    epochs=epochs,
    # Suppress logging.
    verbose=0,
    # Calculate validation results on 20% of the training data.
    validation_split = 0.2
)
# %%
dfhist = pd.DataFrame(history.history)
dfhist["epoch"] = history.epoch
sns.lineplot(dfhist["epoch"],dfhist["loss"])
sns.lineplot(dfhist["epoch"],dfhist["val_loss"])

# %%
