# Machine Learning Project
#### Deep Learning Methods
##### Two-Layer Neural Network
Using the Sequential module to map each input vector of length 26X26 to hidden layers, and essentially to a vector of length 5, which contains class scores or logits. Later, passes the socores directly to a PyTorch’s cross-entropy loss function to build a simple Two-Layer Neural Network to classify images from the QuickDraw dataset as either apple (0), baseball (1), cookie (2), clock (3), or fan (4).

##### Convolutional Neural Network
The key property of a Convolutional Neural Network (CNN) is that hidden units are formed using local spatial regions of the input image, with weights that are shared over spatial regions. Those weights represent the importance to various aspects in the image and are able to differentiate one from the other. CNNs use multilayer perceptrons, as individual neurons respond to stimuli only in a particular region of the image field, and A collection of such fields overlap to cover the entire image.

#### Machine Learning Methods
##### Logistic Regression 
The logistic regression model is used to model binary classification data. Logistic regression is a special case of generalized linear regression where the labels Y are modeled as a linear combination of the data X, but in a transformed space specified by g, a Logistic function. 

##### Perceptron
The perceptron is a mistakedriven online learning algorithm. It takes as input a vector of real-valued inputs x and makes a prediction y = { 1, +1}. Predictions are made using a linear classifier: y = sign(w · x). Updates to w are made only when a prediction is incorrect: y != y_predict. The new weight vector w0 is a function of the current weight vector w and example x, y. The weight vector is updated so as to improve the prediction on the current example. 

##### Support Vector Machine SVM
A Support Vector Machine constructs a hyperplane in high dimensional space, which separates training points of defferent classes while keeping a large margin with regards to the training points closest to the hyperplane.
       
##### Naive Bayes classifier
- Single task Naive Bayes
- Multi-Task Naive Bayes
- Semi-Supervised Naive Bayes
#### How to run the code 
```
python3 classify.py --mode train --algorithm ALGORITHM --model-file DATASET.ALGORITHM.model
                    --data DATASET.train --online-learning-rate X --online-training-iterations X
```
For example 
```
python3 classify.py --mode train --algorithm perceptron --model-file speech.perceptron.model
                    --data speech.train --online-learning-rate 1 --online-training-iterations 5
```

To test the model 
``` 
python3 classify.py --mode test --model-file DATASET.ALGORITHM.model --data DATASET.dev
        --predictions-file DATASET.dev.ALGORITHM
```
For example
```
python3 classify.py --mode test --model-file speech.perceptron.model --data speech.dev
        --predictions-file speech.dev.perceptron
```
#### Component 
• **data.py** 
This file contains the load data function, which parses a given data file and returns features and labels.
The features are stored as a sparse matrix of floats (and in particular as a scipy.sparse.csr matrix of floats),
which has num examples rows and num features columns. The labels are stored as a dense 1-D array of integers with num examples elements.

• **classify.py**
This file is the main testbed to be run from the command line. It takes care of parsing command line arguments,
entering train/test mode, saving models/predictions, etc. 

• **models.py**
This file contains a Model class. Models have (in the very least) a fit method, 
for fitting the model to data, and a predict method, which computes predictions from features. 

• **compute_accuracy.py**
This file is a script which simply compares the true labels from a data file (e.g., bio.dev) 
to the predictions that were saved by running classify.py (e.g., bio.dev.perceptron.predictions).

• **run_on_all_datasets.py** 
The script loops through the datasets to train and test on each file in main data directory, which should contain all of the *.train, *.dev, *.test files, along with an output directory (for models and predictions). 
