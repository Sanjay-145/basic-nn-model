# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.





## THEORY
A neural network with multiple hidden layers and multiple nodes in each hidden layer is known as a deep learning system or a deep neural network. Deep learning is the development of deep learning algorithms that can be used to train and predict output from complex data.The word “deep” in Deep Learning refers to the number of hidden layers i.e. depth of the neural network. Essentially, every neural network with more than three layers, that is, including the Input Layer and Output Layer can be considered a Deep Learning Model.TensorFlow, an open-source software library for machine learning, offers a robust framework for implementing neural network regression models.The Reluactivation function helps neural networks form deep learning models. Due to the vanishing gradient issues in different layers, you cannot use the hyperbolic tangent and sigmoid activation. You can overcome the gradient problems through the Relu activation function.

## Neural Network Model

Include the neural network model diagram.
![WhatsApp Image 2022-08-29 at 9 11 07 AM (1)](https://user-images.githubusercontent.com/75235426/187336686-84883616-2723-4f4e-beaf-d530a00d7016.jpeg)

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model with hidden layer 1 - 5 neurons , hidden layer 2 - 3 neurons and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
data=pd.read_csv("dataset1.csv")
data.head()
x=data[['Input']].values
x
y=data[['Output']].values
y
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=33)
Scaler=MinMaxScaler()
Scaler.fit(x_train)
Scaler.fit(x_test)
x_train1=Scaler.transform(x_train)
x_train1
x_train
AI_BRAIN=Sequential([
    Dense(5,activation='relu'),
    Dense(3,activation='relu'),
    Dense(1)
])
AI_BRAIN.compile(optimizer='rmsprop', loss='mse')
AI_BRAIN.fit(x_train1,y_train,epochs=2000)
loss_df=pd.DataFrame(AI_BRAIN.history.history)
loss_df.plot()
x_test1=Scaler.transform(x_test)
x_test1
AI_BRAIN.evaluate(x_test1,y_test)
x_n1=[[15]]
x_n1_1=Scaler.transform(x_n1)
AI_BRAIN.predict(x_n1_1)
```
## Dataset Information

Include screenshot of the dataset

## OUTPUT

### Training Loss Vs Iteration Plot
![sanj4](https://user-images.githubusercontent.com/75235426/187336512-4bbe56fa-7540-4db7-9422-5bd8fbc4909c.png)

Include your plot here
![sanj1](https://user-images.githubusercontent.com/75235426/187336522-1de89999-834a-40d8-83a1-a6c3fe95bcb7.png)

### Test Data Root Mean Squared Error

Find the test data root mean squared error
![sanj2](https://user-images.githubusercontent.com/75235426/187336537-dfa0943e-c2b9-4188-8a04-ba2c80ef0a33.png)

### New Sample Data Prediction

Include your sample input and output here
![sanj3](https://user-images.githubusercontent.com/75235426/187336551-17e24f3f-71e4-46f8-8440-6617e3936cab.png)

## RESULT
Thus,the neural network regression model for the given dataset is developed.
