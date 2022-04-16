import pandas as pd
import numpy as np
from tensorflow import keras


train_df = pd.read_csv('train(3).csv')
#line to take out all the colors print(train_df.color.unique())
#uses the dictionary to convert strings into numbers
color_dict = {'red': 0, 'blue': 1, 'green': 2, 'teal': 3, 'orange': 4, 'purple': 5}
#we use this line to apply it to our data frame and we are changing the color collumn
#we use a lambda funnction that says for every x in collumn we want to change every collum to the
#color_dict value
train_df['color'] = train_df.color.apply(lambda x: color_dict[x])
np.random.shuffle(train_df.values)

print(train_df.head())
print(train_df.color.unique())

model = keras.Sequential([
	keras.layers.Dense(32, input_shape=(2,), activation='relu'),
	keras.layers.Dense(32, activation='relu'),
	keras.layers.Dense(6, activation='sigmoid')])

#model.compile(optimizer='adam',
#	          loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#	          metrics=['accuracy'])
model.compile(optimizer='adam',
	          loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
			  metrics="accuracy")

x = np.column_stack((train_df.x.values, train_df.y.values))

model.fit(x, train_df.color.values, batch_size=4, epochs=10)

test_df = pd.read_csv('test(3).csv')
test_x = np.column_stack((test_df.x.values, test_df.y.values))

print("EVALUATION")
test_df['color'] = test_df.color.apply(lambda x: color_dict[x])
model.evaluate(test_x, test_df.color.values)

#it is also really useful to predict the data to a specific point
#and it takes a np 2D array
print("Prediction", model.predict(np.array([[0,3]])))
print("Prediction", np.round(model.predict(np.array([[0,3]]))))