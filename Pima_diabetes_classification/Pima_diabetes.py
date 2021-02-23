#!/usr/bin/env python
# coding: utf-8

# ## PIMA DIABETES CLASSIFICATION
# 
# ### Context
# This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset. Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.
# 
# ### Content
# The datasets consists of several medical predictor variables and one target variable, Outcome. Predictor variables includes the number of pregnancies the patient has had, their BMI, insulin level, age, and so on.
# 
# ### Methodology
# I implemented neural network architecture using <b> Keras API</b> with <b> tensorflow </b> backend. I trained the model and then evaluated on the test set that I created by splitting the original dataset.

# In[1]:


import warnings
warnings.filterwarnings(action='ignore')
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class Pima_classification:
    def __init__(self, parameters):
        self.h1 = parameters['n_hidden_units_fc1']
        self.h2 = parameters['n_hidden_units_fc2']
        self.epochs = parameters['n_epochs']
        self.lr = parameters['learning_rate']
        self.bsz = parameters['batch_size']

    def load_and_split_data(self):
        # Loading data from diabetes.csv file
        df = pd.read_csv(r'C:\Users\karan\Documents\Lakehead\Winter_2021\AI\Bonus_question\diabetes.csv')

        # Extracting the label and features from the dataset
        label = df['Outcome'].to_numpy().reshape(-1, 1)
        X = df.drop('Outcome', axis=1).to_numpy()

        # One-hot encoding the labels
        encoder = OneHotEncoder(sparse=False)
        y = encoder.fit_transform(label)

        # Splitting the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=52)
        return (X_train, X_test, y_train, y_test)

    def create_model(self):
        model = Sequential()
        model.add(Dense(self.h1, input_shape = (8,), activation = 'sigmoid', name = 'fc_1'))
        model.add(Dense(self.h2, activation = 'sigmoid', name = 'fc_2'))
        model.add(Dense(2, activation = 'softmax', name = 'output'))
        optimizer = Adam(lr = self.lr)
        model.compile(loss = 'mean_squared_error', optimizer = optimizer, metrics = ['accuracy'])
        return model

    def train_evaluate(self):
        model = self.create_model()
        X_train, X_test, y_train, y_test = self.load_and_split_data()

        # Fitting the model on the training data
        model.fit(X_train, y_train, epochs = self.epochs, batch_size = self.bsz, validation_data = (X_test, y_test))

        # Evaluate on test data
        results = model.evaluate(X_test, y_test)
        return results


if __name__ == '__main__':
    parameters = dict()

    # Enter the number of neurons in first and second hidden layers
    parameters['n_hidden_units_fc1'] = int(input('Enter the number of hidden units in first fully connected layer'))
    parameters['n_hidden_units_fc2'] = int(input('Enter the number of hidden units in second fully connected layer'))
    parameters['n_epochs'] = int(input('Enter the number of epochs for training.'))
    parameters['learning_rate'] = float(input('Enter the learning rate.'))
    parameters['batch_size'] = int(input('Enter the batch size.'))

    # Build, train and evaluate the model
    pima = Pima_classification(parameters)
    results = pima.train_evaluate()
    print('Final test set loss: {:4f}'.format(results[0]))
    print('Final test set accuracy: {:4f}'.format(results[1]))


# In[ ]:




