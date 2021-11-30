import functools
import inspect
import itertools
import warnings

import numpy as np


class _TwoClassesSVC:
    '''A sub SVC classifier for two classes'''

    def __init__(self, X, y, label, gram_matrix, C, kernel, tol, max_iter, **kwargs):
        '''
        args:
            X: X
            y: encoded y. If value == `label`, then +1, else -1
            label: the positive label in this sub SVC
            kernel: packaged as a partial function(x1, x2)
            others: see in `MySVC`
        '''

        ## load params, kwargs are params in kernel function
        ## because the kernel function has been packaged before
        ## these params are useless here
        for arg_name, arg_val in locals().items():
            if arg_name != 'self' and arg_name not in kwargs:
                setattr(self, arg_name, arg_val)

    ## methods
    def cal_kernel(self, i, j, mode='index'):
        '''calculate kernel function using index of data or data itself,
        use gram matrix in terms of the relationship between self.n and self.N
        mode = 'index'(default, in training): `j` is the index of data
        mode = 'data'(in predicting): `j` is the data itself
        '''

        if mode == 'index':
            return self.gram_matrix[i, j]
        else:
            return self.kernel(self.X[i, :], j)

    def f(self, index, mode='index'):
        '''the function to seperate the positive and negative data points
        mode = 'index'(default, in training): `index` is the index of data
        mode = 'data'(in predicting): `index` is the data itself
        '''

        cal_kernel_value = np.vectorize(lambda k: self.cal_kernel(i=k, j=index, mode=mode))
        kernel_value = cal_kernel_value(self.index)
        return np.sum(self.alpha * self.y * kernel_value) + self.b

    def find_first_alpha(self):
        '''find the first changing alpha_i'''
        
        @np.vectorize
        def cal_gap_on_KKT(index):
            '''input the index and calculate the gap on KKT conditions'''

            ## cal y_i(f(x_i))
            loss = self.y[index] * self.f(index)

            ## cal the KKT gap
            if self.alpha[index] == 0:
                return max(1 - loss, 0)
            elif self.alpha[index] == self.C:
                return max(loss - 1, 0)
            else:
                return np.abs(loss - 1)

        gap_on_KKT = cal_gap_on_KKT(self.index)
        first_alpha_index = np.argmax(gap_on_KKT)
        return first_alpha_index, gap_on_KKT[first_alpha_index]

    def find_second_alpha(self, first_alpha_index):
        '''find the second changing alpha_i and calculate its changing value in alpha'''

        ## firstly, calculate the restriction of the alpha_2
        cal_restrict = lambda index: self.y[first_alpha_index] * (
            self.alpha[first_alpha_index] * self.y[first_alpha_index] 
            + self.alpha[index] * self.y[index]
        )

        @np.vectorize
        def get_solution_and_dual_problem_update(index):

            ## secondly, calculate the lower(L) and upper(H) of the alpha_i
            restrict = cal_restrict(index)
            if self.y[first_alpha_index] * self.y[index] == -1:
                L, H = max(0, -restrict), min(self.C, self.C - restrict)
            else:
                L, H = max(0, restrict - self.C), min(self.C, restrict)

            ## thirdly, calculate the unclipped solution of the alpha_i
            ## if index != first_alpha_index and denominator == 0, means the cost function is a linear function
            ## if numerator < 0, means the solution = L
            ## if numerator > 0, means the solution = H
            ## if numerator = 0, the solution will be unchanged
            denominator = (
                self.cal_kernel(first_alpha_index, first_alpha_index) -
                2 * self.cal_kernel(first_alpha_index, index) +
                self.cal_kernel(index, index)
            )
            numerator = self.y[index] * (
                (self.f(first_alpha_index) - self.y[first_alpha_index]) -
                (self.f(index) - self.y[index])
            )
            if denominator == 0:
                if numerator < 0:
                    unclipped_solution = float('-inf')
                elif numerator > 0:
                    unclipped_solution = float('inf')
                else:
                    unclipped_solution = self.alpha[index]
            else:
                unclipped_solution = self.alpha[index] + numerator / denominator

            ## fourthly, calculate the clipped solution
            if unclipped_solution < L:
                clipped_solution = L
            elif unclipped_solution > H:
                clipped_solution = H
            else:
                clipped_solution = unclipped_solution

            ## fifthly, calculate the upgrade between the clipped solution and the old solution
            clipped_solution_upgrade = clipped_solution - self.alpha[index]

            ## sixly, calculate the loss of dual problem caused by the upgrade of the clipped solution
            dual_problem_loss = numerator * clipped_solution_upgrade - denominator * (
                self.alpha[index] * clipped_solution_upgrade 
                + clipped_solution_upgrade ** 2 / 2
            )

            return clipped_solution_upgrade, dual_problem_loss

        ## calculate the solution upgrade and the dual_problem_upgrade
        ## let alpha_2 be the argmax of the dual_problem_upgrade
        ## set the loss of the dual problem on first_alpha_index as -infinity, 
        ## avoiding choosing first alpha as the second alpha
        solution_upgrades, dual_problem_losses = get_solution_and_dual_problem_update(self.index)
        dual_problem_losses[first_alpha_index] = float('-inf')
        second_alpha_index = np.argmax(dual_problem_losses)
        return second_alpha_index, solution_upgrades[second_alpha_index]

    def SMO(self):
        '''calculate the best alpha by SMO method'''

        max_gap_on_KKT = float('inf')
        iter_times = 0
        while iter_times < self.max_iter:
            
            ## choose two alpha_i and update
            first_alpha_index, max_gap_on_KKT = self.find_first_alpha()
            second_alpha_index, solution_upgrade_on_second_alpha = self.find_second_alpha(first_alpha_index)
            self.alpha[second_alpha_index] += solution_upgrade_on_second_alpha
            self.alpha[first_alpha_index] -= (self.y[first_alpha_index] 
                * self.y[second_alpha_index] 
                * solution_upgrade_on_second_alpha
            )
  
            ## update self.b, if none of alpha is support vector, b will be unchanged
            num_possible_bs, sum_possible_bs = 0, 0
            for j in self.index:
                if self.alpha[j] > 0 and self.alpha[j] < self.C:
                    num_possible_bs += 1
                    sum_possible_bs += self.y[j] - (self.f(j) - self.b)
            if num_possible_bs > 0:
                self.b = sum_possible_bs / num_possible_bs

            ## if the max gap of KKT less than the tolerance, exit loop
            if max_gap_on_KKT < self.tol:
                break
            iter_times += 1

    def fit(self):
        '''Calculate the best solution: alpha and b'''

        ## get size and dims of the dataset
        self.N, self.n = self.X.shape[0], self.X.shape[1]

        ## save an index array, which will be reused for many times
        self.index = np.arange(self.N)

        ## init alpha, the solution of the dual problem
        self.alpha = np.zeros(self.N)

        ## init b, the bias of the hypersurface
        ## the value of init value of b does not matter because of the relaxing factor
        self.b = 0

        ## solve by SMO method
        ## self.alpha and self.b will be itered in each iteration
        self.SMO()

    def single_predict(self, x):
        if np.sign(self.f(x, mode='data')) == 1.0:
            return self.label
        return np.nan

    def predict(self, X):
        return np.apply_along_axis(self.single_predict, axis=1, arr=X).astype(np.float)


class MySVC:
    ''''''
    
    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=1e-3, max_iter=-1):
        '''
        args:
            kernel: some special string or a user-defined callable.
            if user-defined, see the rule in the docs of `_KernelLibs`
            same as `sklearn.svm.SVC`
        '''
        
        ## save all params for passing to several two classed SVCs later
        ## `_bye` means the bye label, because for K classes we only need K-1 sub classifiers
        self._params = {attr: val for attr, val in locals().items() if attr != 'self'}
        self._kwargs = self.params.copy()
        self._sub_svcs = {}
        self._bye = None
        
    ## private methods
    def _remap_params(self, X):

        ## must modify gamma first: kernel function relys on gamma
        if self._kwargs['gamma'] == 'scale':
            self._kwargs['gamma'] = 1 / (X.shape[1] * X.var())
        elif self._kwargs['gamma'] == 'auto':
            self._kwargs['gamma'] = 1 / X.shape[1]

        ## kernel: if user has not defined kernel function, find in `_Kernellib`
        ## key thinking: avoiding check the type of kernel function
        ## try to just return a packaged partial function contains only x1, x2 two params
        ## use `inspect.signature` to get possible params
        ## use `functools.partial` to get partial function
        if not callable(self._kwargs['kernel']):

            ## get kernel function from `_Kernellib` first
            kernel_func = _KernelLib.get_kernel(self._kwargs['kernel'])

            ## get params and their values from the signature of the kernel function
            kernel_func_params = inspect.signature(kernel_func).parameters

            ## build params dict
            ## the first two params is x1 and x2, ignore them
            params = {param: self._kwargs[param] for index, param in enumerate(kernel_func_params) if index >= 2}

            ## generate partial function
            ## check value restrictions of params here(check type restrictions while called)
            self._kwargs['kernel'] = functools.partial(kernel_func, **params)

        ## max_iter: if `max_iter` = -1, means infinity
        if self._kwargs['max_iter'] == -1:
            self._kwargs['max_iter'] = float('inf')

    def _get_gram_matrix(self, X):
        '''Build the gram matrix to save time'''

        N = len(X)
        kernel = self._kwargs['kernel']
        gram_matrix = np.empty((N, N))
        for i, j in itertools.product(range(N), range(N)):
            if i <= j:
                gram_matrix[i, j] = kernel(X[i, :], X[j, :])
                gram_matrix[j, i] = gram_matrix[i, j]
        return gram_matrix
        
    def _label_encode(self, X, y):
        '''Seperate data with K classes into K combinations, 
        each combination has a label encoded as +1, 
        and other labels encoded as -1
        '''
        
        labels = np.unique(y)
        np.random.shuffle(labels)
        self._bye = labels[-1]

        ## for K classes, use K-1 sub classifiers is ok
        for label in labels[:-1]:
            yield np.where(y == label, 1, -1), float(label)

    ## public attributes
    @property
    def is_fitted(self) -> int:
        '''Judge the model is fitted or not'''

        if len(self._sub_svcs) > 0:
            return True
        warnings.warn('The model is not fitted.')
        return False

    @property
    def n_sub_classifiers(self) -> int:
        '''Get the number of sub classifiers'''

        return len(self._sub_svcs)

    @property
    def bye(self):
        '''Get the bye label of the labels'''

        return self._bye

    @property
    def dual_coef(self) -> dict:
        '''Get dual coeffients(alphas)
        return:
            dual_coef: dict, each key is the label of the sub svc, value is the alpha
        '''
        if not self.is_fitted:
            return {}
        return {label: sub_svc.alpha for label, sub_svc in self._sub_svcs.items()}

    @property
    def coef(self) -> dict:
        '''Get coeffients if the kernel is linear
        return:
            coef: dict, each key is the label of the sub svc, value is the omega
        '''
        
        if not self.is_fitted:
            return {}
        elif self._kwargs.get('kernel', None) != 'linear':
            raise ValueError('Only linear SVC has explicit coef!')
        return {
            label: np.sum(sub_svc.alpha * sub_svc.y * sub_svc.X, axis=0) 
            for label, sub_svc in self._sub_svcs.items()
        }

    @property
    def intercept(self) -> dict:
        '''Get dual coeffients(alphas)
        return:
            dual_coef: dict, each key is the label of the sub svc, value is the alpha
        '''

        if not self.is_fitted:
            return {}
        return {label: sub_svc.b for label, sub_svc in self._sub_svcs.items()}   
    
    @property
    def params(self) -> dict:
        return self._params 
      
    ## public methods        
    def fit(self, X, y):
        '''Choose the One-VS-Rest(OVR) strategy, use K sub SVC to get proper result'''
        
        ## remap some params: calculate gamma, load right kernel and max_iter
        self._remap_params(X)

        ## calculate the gram matrix
        gram_matrix = self._get_gram_matrix(X)
        
        ## train every single sub SVC
        for encoded_y, label in self._label_encode(X, y):
            sub_svc = _TwoClassesSVC(X=X, y=encoded_y, label=label, gram_matrix=gram_matrix, **self._kwargs)
            sub_svc.fit()
            self._sub_svcs[label] = sub_svc

    def predict(self, X) -> np.ndarray:
        '''Prediction based on given X
        return:
            np.ndarray(len(X), 1)
        '''

        reduced_predicts = np.empty((len(X), self.n_sub_classifiers))
        for index, sub_svc in enumerate(self._sub_svcs.values()):
            reduced_predicts[:, index] = sub_svc.predict(X)

        def find_most_freq(arr):
            labels, counts = np.unique(arr[~np.isnan(arr)], return_counts=True)

            ## if all preds are np.nan, means the label is the bye label
            try:
                most_possible_label_index = np.argmax(counts)
                return labels[most_possible_label_index]
            except:
                return self.bye

        return np.apply_along_axis(find_most_freq, axis=1, arr=reduced_predicts)

    def accuracy(self, X, y):
        return np.sum(y == self.predict(X)) / len(y)


class _KernelLib:
    '''Library of kernels
    pre-load four commun kernel, same as `sklearn.svm.SVC` 
    format of a kernel function:
        any_kernel_func(X1, X2, param1=1.0, param2=1.0, ...)
    keep the first two params are x1, x2, others are params with default values
    '''

    def check_params_wrapper(func):
        '''A decorator to check value restrictions of params
        check value restrictions only, 
        leave every kernel function it self to check type restrictions,
        which can mostly using the exceptions codes of the origin libs
        '''

        @functools.wraps(func)
        def wrapped(*args, **kwargs):

            ## 1) check gamma > 0:
            if 'gamma' in kwargs:
                try:
                    assert kwargs['gamma'] > 0
                    checked = True
                except AssertionError:
                    raise ValueError('Gamma must be positive!')
                except TypeError:
                    pass

            ## 2) check another param...

            result = func(*args, **kwargs)
            return result
        return wrapped

    def get_kernel(kernel_name):
        '''Get the needed kernel function, the keyword is the name of kernel function
        if not in the lib, use the exception codes in `getattr`
        '''

        return getattr(_KernelLib, kernel_name)

    def linear(x1, x2):
        '''Linear kernel'''

        return x1 @ x2

    @check_params_wrapper
    def poly(x1, x2, gamma=1.0, coef0=0.0, degree=1.0):
        '''Poly kernel'''

        return np.power(gamma * linear(x1, x2) + coef0, degree)

    @check_params_wrapper
    def rbf(x1, x2, gamma=0.0):
        '''Gaussian kernel'''

        dis = np.sum(np.power(x1 - x2, 2))
        return np.exp(- gamma * dis)

    @check_params_wrapper
    def sigmoid(x1, x2, gamma=1.0, coef0=0.0):
        '''Sigmoid kernel'''

        return np.tanh(gamma * linear(x1, x2) + coef0)
