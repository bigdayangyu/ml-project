# Machine Learning Project
#### Methods
1. Logistic Regression 
2. Perceptron
3. Naive Bayes classifier
- Single task Naive Bayes
- Multi-Task Naive Bayes
- Semi-Supervised Naive Bayes
#### How to run the code 
```
python3 classify.py --mode train --algorithm perceptron --model-file speech.perceptron.model
                    --data speech.train --online-learning-rate 1 --online-training-iterations 5
```
#### Component 
• **data.py** 
This file contains the load data function, which parses a given data file and returns features and labels.
The features are stored as a sparse matrix of floats (and in particular as a scipy.sparse.csr matrix of floats),
which has num examples rows and num features columns. The labels are stored as a dense 1-D array of integers with num examples elements.

• **classify.py**
This file is the main testbed to be run from the command line. It takes care of parsing command line arguments,
entering train/test mode, saving models/predictions, etc. Once again, do not change the names of existing command-line arguments.

• **models.py**
This file contains a Model class which you should extend. Models have (in the very least) a fit method, 
for fitting the model to data, and a predict method, which computes predictions from features. 
You are free to add other methods as necessary. Note that all predictions from your model must be 0 or 1;
if you use other intermediate values for internal computations, then they must be converted before they are returned.

• **compute_accuracy.py**
This file is a script which simply compares the true labels from a data file (e.g., bio.dev) 
to the predictions that were saved by running classify.py (e.g., bio.dev.perceptron.predictions).

• **run_on_all_datasets.py** 
This file is not necessarily needed, but is included simply to make it easier for you to test your algorithms on all datasets, 
and to make sure that your algorithms even run on the test sets. Inside this script you can specify the main data directory, 
which should contain all of the *.train, *.dev, *.test files, along with an output directory (for models and predictions). 
The script loops through the datasets to train and test on each. Feel free to modify this script as needed; by default, 
it assumes the data directory is ./datasets and that the desired output directory is ./output.
