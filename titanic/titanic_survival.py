import pandas as pd
import numpy as np
from tensorflow import keras
import tensorflow as tf

np.set_printoptions(suppress=True)

train_df = pd.read_pickle('TitanicSurvivalDataNumeric.pkl')
print(train_df.head())

np.random.shuffle(train_df.values)

target_variable = ['Survived']
predictors = [
    'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S'
]

train_df_x = train_df[predictors].values
train_df_y = train_df[target_variable].values

model = keras.Sequential([
    keras.layers.Dense(24, input_shape=(10,), activation='relu'),
    keras.layers.Dense(14, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[tf.keras.metrics.BinaryAccuracy()])

x = np.column_stack((train_df_x, train_df_y))

model.fit(x, train_df.Survived.values, batch_size=10, epochs=10)

# test_df = pd.read_pickle('TitanicSurvivalDataNumeric.pkl')

# np.random.shuffle(test_df.values)

# test_df_x = test_df[predictors].values
# test_df_y = test_df[target_variable].values

# test_x = np.column_stack((test_df_x, test_df_y))

# print("EVALUATION")
# model.evaluate(test_x, test_df.Survived.values)
