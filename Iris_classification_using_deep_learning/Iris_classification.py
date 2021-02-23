#!/usr/bin/env python
# coding: utf-8

# ## IRIS DATASET CLASSIFICATION USING NEURAL NETWORKS
# In the code below, I have used keras with tensorflow backend to classify iris dataset. User is prompted to enter the following values: </br>
# <ol>
#     <li> Number of neurons in the first hidden layer.</li>
#     <li>Number of neurons in the second hidden layer.</li>
#     <li>Number of epochs for which to train the model.</li>
#     <li>The learning rate value.</li>
#     <li>The batch size.</li>
# </ol>

# In[4]:


# Importing the necessary modules
import warnings
warnings.filterwarnings(action='ignore')
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


# In[10]:


class Iris_classification:

    def __init__(self, parameters):
        self.h1 = parameters['n_hidden_units_fc1']
        self.h2 = parameters['n_hidden_units_fc2']
        self.epochs = parameters['n_epochs']
        self.lr = parameters['learning_rate']
        self.bsz = parameters['batch_size']

    def load_and_split_data(self):
        iris_data = load_iris()
        X = iris_data.data
        target = iris_data.target.reshape(-1, 1)
        encoder = OneHotEncoder(sparse=False)
        y = encoder.fit_transform(target)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)
        return (X_train, X_test, y_train, y_test)

    def create_model(self):
        model = Sequential()
        model.add(Dense(self.h1, input_shape = (4,), activation = 'relu', name = 'fc1'))
        model.add(Dense(self.h2, activation = 'relu', name = 'fc2'))
        model.add(Dense(3, activation = 'softmax', name ='output'))
        optimizer = Adam(lr = self.lr)
        model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train_evaluate(self):
        model = self.create_model()
        X_train, X_test, y_train, y_test = self.load_and_split_data()
        model.fit(X_train, y_train, verbose = 2, batch_size = self.bsz, epochs = self.epochs)
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
    i_classification = Iris_classification(parameters)
    results = i_classification.train_evaluate()
    print('Final test set loss: {:4f}'.format(results[0]))
    print('Final test set accuracy: {:4f}'.format(results[1]))


# In[ ]:




