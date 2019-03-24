import numpy as np
from scipy.special import expit
from data import load_data
from scipy import sparse


class Model(object):

    def __init__(self):
        self.num_input_features = None

    def fit(self, X, y):
        """ Fit the model.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].
            y: A dense array of ints with shape [num_examples].
        """

        raise NotImplementedError()



    def predict(self, X):
        """ Predict.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].

        Returns:
            A dense array of ints with shape [num_examples].
        """
        raise NotImplementedError()

class Model(object):

    def __init__(self):
        self.num_input_features = None

    def fit(self, X, y):
        """ Fit the model.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].
            y: A dense array of ints with shape [num_examples].
        """

        raise NotImplementedError()

    def predict(self, X):
        """ Predict.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].

        Returns:
            A dense array of ints with shape [num_examples].
        """
        raise NotImplementedError()

class Useless(Model):

    def __init__(self):
        super().__init__()
        self.reference_example = None
        self.reference_label = None

    def fit(self, X, y):
        self.num_input_features = X.shape[1]
        # Designate the first training example as the 'reference' example
        # It's shape is [1, num_features]
        self.reference_example = X[0, :]
        # Designate the first training label as the 'reference' label
        self.reference_label = y[0]
        self.opposite_label = 1 - self.reference_label

    def predict(self, X):
        if self.num_input_features is None:
            raise Exception('fit must be called before predict.')
        # Perhaps fewer features are seen at test time than train time, in
        # which case X.shape[1] < self.num_input_features. If this is the case,
        # we can simply 'grow' the rows of X with zeros. (The copy isn't
        # necessary here; it's just a simple way to avoid modifying the
        # argument X.)
        num_examples, num_input_features = X.shape
        if num_input_features < self.num_input_features:
            X = X.copy()
            X._shape = (num_examples, self.num_input_features)

        # Or perhaps more features are seen at test time, in which case we will
        # simply ignore them.
        if num_input_features > self.num_input_features:
            X = X[:, :self.num_input_features]
        # Compute the dot products between the reference example and X examples
        # The element-wise multiply relies on broadcasting; here, it's as if we first
        # replicate the reference example over rows to form a [num_examples, num_input_features]
        # array, but it's done more efficiently. This forms a [num_examples, num_input_features]
        # sparse matrix, which we then sum over axis 1.

        dot_products = X.multiply(self.reference_example).sum(axis=1)
        # dot_products is now a [num_examples, 1] dense matrix. We'll turn it into a
        # 1-D array with shape [num_examples], to be consistent with our desired predictions.
        dot_products = np.asarray(dot_products).flatten()
        # If positive, return the same label; otherwise return the opposite label.
        same_label_mask = dot_products >= 0
        opposite_label_mask = ~same_label_mask
        y_hat = np.empty([num_examples], dtype=np.int)
        y_hat[same_label_mask] = self.reference_label
        y_hat[opposite_label_mask] = self.opposite_label

        return y_hat


class Perceptron(Model):
    """ Perceptron algorithm for binary classification """

    def __init__(self, eta = 1, I = 5):
        super().__init__()
        # nitializations etc. 
        self.eta = eta# learning rate η
        self.I = I# number of iteration 

    def fit(self, X, y):
        # fit the model.
        self.num_examples, self.num_input_features = X.shape
        self.w = np.zeros((1,self.num_input_features))   

        for k in range(self.I):           
            for x , yi in zip(X, y):
                w_dot= np.asarray(self.w).flatten()
                dot_products = x.multiply(w_dot).sum(axis=1)

                if dot_products >= 0.0:
                    y_sign = 1
                else:
                    y_sign = -1
                if yi == 0.0:
                    yi = -1

                if yi != y_sign:
                    self.w += self.eta*(yi)*x        
                
    def predict(self, X):
        """ Predict the label y = sign(w · x )
            Input: sample X
            Output: classification label: Y
        """
        #  Make predictions.
        if self.num_input_features is None:
            raise Exception('fit must be called before predict.')

        num_examples, num_input_features = X.shape

        if num_input_features < self.num_input_features:
            X = X.copy()
            X._shape = (num_examples, self.num_input_features)

        if num_input_features > self.num_input_features:
            X = X[:, :self.num_input_features]
    
        y_store = []
        for x in X:
            w_dot= np.asarray(self.w).flatten()
            dot_products = x.multiply(w_dot).sum(axis=1)
            dot_products = np.asarray(dot_products).flatten()

            if dot_products >= 0:
                y_store.append(1)
            else:               
                y_store.append(0)
        
        y_hat = np.array([y_store])

        return y_hat[0]

        
class Logistic(Model):
    """ Logistic regression for binary classification """

    def __init__(self, eta=0.1, I=100 ):
        super().__init__()
        self.eta = eta# learning rate η
        self.I = I# number of iteration 
    

    def fit(self, X, y):
        # fit the model.
        self.num_examples, self.num_input_features = X.shape
        self.w = np.zeros((1,self.num_input_features)) # Initialize weight
        for k in range(self.I):

            for x, yi in zip(X, y):
                w_dot= np.asarray(self.w).flatten()
                dot_products = x.multiply(w_dot).sum(axis=1)
                dot_products = np.asarray(dot_products).flatten()
                h_x = expit(dot_products)
                self.w += self.eta*(yi-h_x)*x
                
    def predict(self, X):
        # make predictions label y = sign(w · x )
        if self.num_input_features is None:
            raise Exception('fit must be called before predict.')

        num_examples, num_input_features = X.shape

        y_hat = np.zeros([num_examples], dtype=np.int)

        if num_input_features < self.num_input_features:
            X = X.copy()                    
            X._shape = (num_examples, self.num_input_features)


        if num_input_features > self.num_input_features:
            X = X[:, :self.num_input_features]

        y_store = []

        for x in X:
            w_dot= np.asarray(self.w).flatten()
            dot_products = x.multiply(w_dot).sum(axis=1)
            dot_products = np.asarray(dot_products).flatten()

            ypredict = expit(dot_products)#logistic function
            if ypredict >= 0.5:
                y_store.append(1)
            else:               
                y_store.append(0)
               
        y_hat = np.array([y_store])      
        return y_hat[0]

class Pegasos(Model):
    """ Pegasos stochastic gradient descent algorithm 
        to solve a Support Vector Machine(SVM) for 
        binary classification problems.
    """

    def __init__(self, eta=0.1, I=10, lam = 1e-4):
        super().__init__()
        self.eta = eta # learning rate η
        self.I = I # number of iteration 
        self.lam = lam
    

    def fit(self, X, y):
        #fit the model.
        self.num_examples, self.num_input_features = X.shape
        

        self.w = np.zeros((1,self.num_input_features))
        time_step = 1

        for k in range(self.I):
            # time_step = 1
            for x, yi in zip(X, y):
                if yi == 0.0:
                    yi = -1
                self.eta = 1/(time_step*self.lam)
                w_dot= np.asarray(self.w).flatten()
                dot_products = x.multiply(w_dot).sum(axis=1)
                dot_products = np.asarray(dot_products).flatten()

                if yi*dot_products < 1:
                    self.w = (1-1/time_step)*self.w + 1/(self.lam*time_step)*yi*x
                else:
                    self.w = (1-1/time_step)*self.w
                time_step += 1

                
    def predict(self, X):
        #make predictions.
        if self.num_input_features is None:
            raise Exception('fit must be called before predict.')

        num_examples, num_input_features = X.shape

        y_hat = np.zeros([num_examples], dtype=np.int)

        if num_input_features < self.num_input_features:
            X = X.copy()
            X._shape = (num_examples, self.num_input_features)


        if num_input_features > self.num_input_features:
            X = X[:, :self.num_input_features]

        y_store = []

        for x in X:
            w_dot= np.asarray(self.w).flatten()
            dot_products = x.multiply(w_dot).sum(axis=1)
            dot_products = np.asarray(dot_products).flatten()

            
            if dot_products >= 0.0:
                y_store.append(1)
            else:               
                y_store.append(0)
               
        y_hat = np.array([y_store])      
        return y_hat[0]
 
class nb(Model):
    """ Naive Bayes classifier
        Options: - Single label yi per example
                 - Multi-Task Naive Bayes to predict T distinct binary labels 
                   for each example with differet correlations assumption(Independent, Joint).
                 - Semi-Supervised Naive Bayes with latent variable Using 
                   Expectation Maximization(EM) method
    """

    def __init__(self, independent_mode, I , latent_states):
        super().__init__()
        self.independent_mode = independent_mode
        self.I = I # number of iteration 
        self.latent_states = latent_states
        self.muti_task_flag = False
        self.q_y = []
        self.q_x_y = []
        self.multi_task_parameters = []
        self.delta = []
        self.category = []
        self.q_y_z = []
        self.q_z = []
  
    def prior(self,X,  Y):
        """
        output:  Proir P(y|i)

        """
        category, counts = np.unique(Y, return_counts=True)
        # print(len(category), counts)
        for cate_counts in range(2):
            if len(category) == 1 and category == 0:
                return [1, 0.0000000001]

            elif len(category) == 1 and category == 1:
                return [0.0000000001, 1]
            else:
                
                return [(cate_counts +1)/(Y.shape[0] +2 )for cate_counts in category] 

    def prior_joint(self, X, Y):

        category, indexes, counts  = np.unique(Y, return_counts=True, return_index=True)
        prob = [(cate_counts +1)/(Y.shape[0] +len(category) )for cate_counts in counts]
        return prob
     
    def prior_latent(self, Y, z):

        category, indexes, counts  = np.unique(Y, return_counts=True, return_index=True)
        separated = [[x for x, y in zip(z, Y) if y == ck] for ck in np.unique(Y)]

     
        count = np.array([len(i) + 1 for i in separated])
        prob = [(c+1)/(len(z)+ len(category)) for c in count]
        return prob


    def conditional_feature_j(self, X, Y ):
        """
        conditional likelihood for feathure_j 
        p(xj = aj | y = ck)
        
        output: [cond_label0 cond_label1]

        """     
        x_dense = X.toarray()
        category, counts = np.unique(Y, return_counts=True)
        separated = [[x for x, y in zip(x_dense, Y) if y == ck] for ck in np.unique(Y)]
        count = np.array([np.array(i).sum(axis=0) + 1 for i in separated])
  
        return [count[i]/ (counts[i] +2 )for i in range(counts.shape[0])]

    def conditional_feature_j_joint(self, X, Y):
        if isinstance(X,(list,np.ndarray)):
            x_dense = X

            category,indexes, counts = np.unique(x_dense, return_counts=True, return_index=True)
            separated = [[x for x, y in zip(Y, x_dense) if y == ck] for ck in np.unique(x_dense)]
            count = np.array([len(i) + 1 for i in separated])

            return [count[i]/ (counts[i] +2 )for i in range(counts.shape[0])]

        else:
        
            x_dense = X.toarray()
            category,indexes, counts = np.unique(Y, return_counts=True, return_index=True)           
            separated = [[x for x, y in zip(x_dense, Y) if y == ck] for ck in np.unique(Y)]
            count = np.array([np.array(i).sum(axis=0) + 1 for i in separated])
            return [count[i]/ (counts[i] +2 )for i in range(counts.shape[0])]

    def parameterize_y(self, Y):
            category,indexes, counts = np.unique(Y, return_counts=True, return_index=True)
            for c, i in zip(category, indexes):
                Y[Y == c] = int(i)
            Y = list(map(int, Y))
            return Y

    def log_posterior(self, X):
        """
        take in pre-calculated parameters to esitimate the label of unlabeled Xu
        """
        if isinstance(X,(list,np.ndarray)):
            x_dense = np.array(X)
            # print('log_posterir: Y', x_dense)
            return [np.multiply(np.log(self.q_x_y) , x).sum(axis = 0) + np.log(self.q_y) for x in x_dense]
        else:
            x_dense = X.toarray()
        
            return [np.multiply(np.log(self.q_x_y) , x).sum(axis = 1) + np.log(self.q_y) for x in x_dense]

    def get_posterior(self, prior_prob, condi_prob, X):

        if isinstance(X,(list,np.ndarray)):
            x_dense = np.array(X)
            return [np.multiply(np.log(condi_prob), x).sum(axis = 0) + np.log(prior_prob) for x in x_dense]

        else:
            x_dense = X.toarray()
            return [np.multiply(np.log(condi_prob), x).sum(axis = 1) + np.log(prior_prob) for x in x_dense]
    
    def latent_posterior(self, X):
        x_dense = X.toarray()
        prior_prob = np.multiply(self.q_y_z,self.q_z)
        return [np.multiply(np.log(self.q_x_y), x).sum(axis = 1) + np.log(prior_prob) for x in x_dense]
    

    def assign_label(self, posterior_prob):
        """
        calculate argmax_y for log_posterior 
        """
        new_label = [np.argmax(prob, axis = 0) for prob in posterior_prob]
        return np.array(new_label, dtype = int)

    def semisup_train(self, X, Y):
        num_features = X.shape[1] # number of features 

        # initialization 
        Y_assigned  = Y.copy()
        unlabeled_index = np.where(Y == -1)
        labeled_index = np.where(Y != -1)
        Y_unlabeled_initial = Y_assigned[unlabeled_index]
        for i in range(len(Y_unlabeled_initial)):
            if i % self.latent_states == 1:
                Y_unlabeled_initial[i] = 1
            else:
                Y_unlabeled_initial[i] = 0

        # Initialized Y with hard assisgment(i % latent_states) to unlabeled Y  
        Y_assigned[unlabeled_index] = Y_unlabeled_initial 

        # Unlabeled data 
        x_unlabeled = X[unlabeled_index]
        y_unlabeled = Y_assigned[unlabeled_index]
        x_labeled = X[labeled_index]
        y_labeled = Y[labeled_index]

        # Train model with labeled data 
        y_category, y_counts = np.unique(y_labeled, return_counts=True)
       
        self.q_y = self.prior(x_labeled,y_labeled) #proir 
        self.q_x_y = self.conditional_feature_j(x_labeled, y_labeled)#likelihood 

        for i in range(self.I):

            post_prob_inital = self.log_posterior(x_unlabeled)
            # print(post_prob_inital)
            labels = self.assign_label(post_prob_inital)
            self.q_y = self.prior(x_unlabeled,labels) #proir 
            self.q_x_y = self.conditional_feature_j(x_unlabeled, labels)
       
            new_all_x = sparse.vstack((x_labeled, x_unlabeled))
            new_all_y = np.hstack((y_labeled, labels))
            self.q_y = self.prior(new_all_x,new_all_y) #proir 
            self.q_x_y = self.conditional_feature_j(new_all_x,new_all_y)#likelihood 
        
        theta1, theta2 = self.q_y, self.q_x_y
        return theta1, theta2

    def sup_train(self, X, Y):
        num_features = X.shape[1] # number of features 
        x_labeled = X.copy()
        y_labeled  = Y.copy()

        # independent mode: 
        if self.independent_mode == 'independent':
            for i in range(self.I):
                self.q_y = self.prior(x_labeled,y_labeled) 
                self.q_x_y = self.conditional_feature_j(x_labeled, y_labeled)
        # joint mode: 
        elif self.independent_mode == 'joint':
            for i in range(self.I):
                self.q_y = self.prior_joint(x_labeled, y_labeled)
                self.q_x_y = self.conditional_feature_j_joint(x_labeled, y_labeled)


        theta1, theta2 = self.q_y, self.q_x_y
        return theta1, theta2

    def latant_train(self, X, Y):

        z = np.empty(Y.shape[0],)
        for i in range(Y.shape[0]):
            z[i] = i % self.latent_states
        self.q_y = self.prior_joint(X,Y) 
        self.q_x_y = self.conditional_feature_j(X, Y)
        self.q_z = self.prior_latent(Y, z)
        self.q_y_z = self.conditional_feature_j_joint(Y, z)

        parameterized_y = self.parameterize_y(Y)
        post_prob = self.get_posterior(self.q_z, self.q_y_z, parameterized_y) #prior_prob, condi_prob
        
        pridic_prob = self.latent_posterior(X)
        new_z = self.assign_label(post_prob)

        for t in range(self.I):
            self.q_z = self.prior_joint(Y, new_z)
            self.q_y_z = self.conditional_feature_j_joint(Y, new_z) 
            post_prob = self.latent_posterior( X) #prior_prob, condi_prob
            new_z = self.assign_label(post_prob) 

        theta1, theta2 =  np.multiply(self.q_y_z,self.q_z), self.q_x_y
        return theta1, theta2


    def fit(self, X, Y):
        X = X.copy()
        Y  = Y.copy()
        category, indexes, counts  = np.unique(Y, return_counts=True, return_index=True)
        self.category = category

        if np.issubdtype(type(Y[0]), np.dtype(int)):

            self.semisup_train(X, Y)
        else:
            self.muti_task_flag = True
            y_array = [[int(yi) for yi in y ]for y in Y]
            y_multi = np.array(y_array)

            if self.independent_mode == 'independent':
                tasks = [y_multi[:,i] for i in range(y_multi.shape[1])]
                parameters = [self.sup_train(X, yt) for yt in tasks]
                self.multi_task_parameters = parameters
                
            elif self.independent_mode == 'joint':
                parameters = self.sup_train(X, Y)
                self.multi_task_parameters = parameters

            else:
                #initialize zi
                parameters = self.latant_train(X, Y)
            
                self.multi_task_parameters = parameters

    def predict(self, X):
        if self.muti_task_flag == True:
            labels = np.empty((0, X.shape[0]))

            if self.independent_mode == 'independent':
                for parameters in self.multi_task_parameters:
                    post_prob = self.get_posterior(parameters[0], parameters[1], X) #prior_prob, condi_prob
                    task = self.assign_label(post_prob)
                    labels = np.append(labels,[task], axis = 0)

                lb = labels[0]

                for i in range(labels.shape[0]-1):
                    lb = np.column_stack((lb,labels[i+1]))

                container =  []
                for label in lb:
                    test = [''.join(str(int(l))) for l in label]
                    test2 = ''.join([str(x) for x in test])
                    container = np.append(container, [test2])
                return container

            elif self.independent_mode == 'joint':
                parameters =  self.multi_task_parameters
                post_prob = self.get_posterior(parameters[0], parameters[1], X) #prior_prob, condi_prob
                task = self.assign_label(post_prob)
      
                label = self.category[task]
                return label

            elif self.independent_mode == 'latent':
                parameters = self.multi_task_parameters
        
                post_prob = self.get_posterior(parameters[0], parameters[1], X) #prior_prob, condi_prob
                task = self.assign_label(post_prob)

                label = self.category[task]
                return label                                 
        else:
            posterior_prob = self.log_posterior(X)
            labels = self.assign_label(posterior_prob)
            return labels
         

