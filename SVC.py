import functools
import inspect

import numpy as np


class _TwoClassesSVC:
    '''A sub SVC classifier for two classes'''

    def __init__(self, X, y, label, C, kernel, tol, max_iter, random_state, **kwargs):
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
            
            ## if has gram_matrix, use it
            ## if not, calculate it directly
            if self.gram_matrix:
                return self.gram_matrix[i, j]
            else:
                return self.kernel(self.X[i, :], self.X[j, :])
        else:
            return self.kernel(self.X[i, :], j)

    def f(self, index, mode='index'):
        '''the function to seperate the positive and negative data points
        mode = 'index'(default, in training): `index` is the index of data
        mode = 'data'(in predicting): `index` is the data itself
        '''

        cal_kernel_value = lambda k: self.cal_kernel(i=k, j=index, mode=mode)
        kernel_value = cal_kernel_value(self.index)
        return np.sum(self.alpha * self.y * kernel_value) + self.b

    def get_random_generator(self):
        return np.random.RandomState(self.random_state)

    def set_gram_matrix(self):
        '''build gram matrix if dims of X is more than the length of X'''

        if self.n <= self.N:
            return
        gram_matrix = np.empty(self.N, self.N)
        for i, j in itertools.product(range(self.N), range(self.N)):
            if i <= j:
                gram_matrix[i, j] = self.kernel(self.X[i, :], self.X[j, :])
                gram_matrix[j, i] = gram_matrix[i, j]
        return gram_matrix

    ## procedures
    def SMO(self):
        '''calculate the best alpha by SMO method'''

        max_gap_on_KKT = float('int')
        iter_times = 0
        while iter_times < self.max_iter:
            
            ## choose two alpha_i and update
            first_alpha_index, max_gap_on_KKT = self.find_first_alpha()
            if max_gap_on_KKT < self.tol:
                break
            second_alpha_index, max_change_on_second_alpha = self.find_second_alpha()
            self.alpha[second_alpha_index] += max_change_on_second_alpha
            self.alpha[first_alpha_index] -= (self.y[first_alpha_index] 
                * self.y[second_alpha_index] 
                * max_change_on_second_alpha
            )
            
            ## update self.b
            possible_bs = []
            for j in self.index:
                if self.alpha[j] > 0 and self.alpha[j] < self.C:
                    for i in self.index:
                        possible_b = self.y[j] - (self.f(j) - self.b)
                        possible_bs.append(possible_b)
            self.b = np.array(possible_bs).mean()           
            iter_times += 1

    def find_first_alpha(self):
        '''find the first changing alpha_i in alpha'''
        
        def cal_gap_on_KKT(index):
            '''input the index and calculate the gap on KKT conditions'''

            ## cal y_i(f(x_i))
            loss = self.y * self.f(self.X[index, :])

            ## cal gap
            if self.alpha[index] == 0:
                return max(1 - loss, 0)
            elif self.alpha[index] == self.C:
                return max(loss - 1, 0)
            else:
                return np.abs(loss - 1)

        gap_on_KKT = _cal_gap_on_KKT(self._index)
        first_alpha_index = np.argmax(gap_on_KKT)
        return first_alpha_index, gap_on_KKT[first_alpha_index]

    def find_second_alpha(self, first_alpha_index):
        '''find the second changing alpha_i in alpha'''

        ## first, calculate the unclipped alpha_i
        cal_unclipped_solution = lambda index: self.alpha[index] + self.y[index] * (
            (self.f(first_alpha_index) - self.y[first_alpha_index, :]) -
            (self.f(first_alpha_index) - self.y[index, :])
        ) / (
            self.cal_kernel(first_alpha_index, first_alpha_index) -
            2 * self.cal_kernel(first_alpha_index, index) +
            self.cal_kernel(index, index)
        )

        ## second, calculate the restriction(\gamma)
        cal_restrict = lambda index: self.y[first_alpha_index, :] * (
            self.alpha[first_alpha_index] * self.y[first_alpha_index, :] 
            + self.alpha[index] * self.y[index, :]
        )

        ## third, calculate the lower(L) and upper(H) of the alpha_i
        cal_L = lambda index: max(0, -cal_restrict(index))\
        if self.alpha[first_alpha_index] * self.y[first_alpha_index, :] == -1\
        else max(0, self.C)

        cal_H = lambda index: min(self.C, self.C - cal_restrict(index))\
        if self.alpha[first_alpha_index] * self.y[first_alpha_index, :] == -1\
        else min(self.C, cal_restrict(index))

        ## fourth, clip the solution
        clip = lambda index: cal_L(index) if cal_unclipped_solution(index) < cal_L(index)\
        else (cal_H(index) if cal_unclipped_solution(index) > cal_H(index) else cal_unclipped_solution(index))

        ## finally, calculate the changing range
        cal_changing_range = lambda index: clip(index) - self.alpha[index]

        ## run the functions stack
        changing_range = cal_changing_range(self.index)
        abs_changing_range = np.abs(changing_range)

        ## second alpha index \ne first alpha index
        ## set changing range of first alpha index for avoiding 
        ## selecting first alpha index as second alpha index
        ## (zero also has risk, only negative value is ok)
        abs_changing_range[first_alpha_index] = -1
        second_alpha_index = np.argmax(abs_changing_range)
        return second_alpha_index, changing_range[second_alpha_index]

    def fit(self):

        ## get size and dims of the dataset
        self.N, self.n = self.X.shape[0], self.X.shape[1]

        ## init random generator
        rng = self.get_random_generator()

        ## save an index array, which will be reused for many times
        self.index = np.arange(self.N)

        ## init alpha, the solution of the dual problem
        self.alpha = rng.rand(self.N) * self.C

        ## adjust the last component of alpha to fulfill the restrict of the dual problem
        ## do not calculate the first N-1 sum for avolding slice N-1, which will cost lots of memory
        self.alpha[-1] = self.y[-1] * -(self.alpha * self.y - self.alpha[-1] * self.y[-1])

        ## if n > N, calculate the Gram matrix of the kernel(X1, X2) to short the training time
        ## else, `self._gram_matrix` is `None`
        self.gram_matrix = self.set_gram_matrix()

        ## init b, the bias of the hypersurface
        self.b = rng.rand(self.N)

        ## solve by SMO method
        self.SMO()


class MySVC:
    ''''''
    
    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=1e-3, max_iter=-1, random_state=None):
        '''
        args:
            kernel: some special string or a user-defined callable.
            if user-defined, see the rule in the docs of `_KernelLibs`
            same as `sklearn.svm.SVC`
        '''
        
        ## save all params for passing to several two classed SVCs later
        self._kwargs = {attr: val for attr, val in locals().items() if attr != 'self'}
        self._sub_svcs = []
        
    ## private procedures
    def _remap_params(self):

        ## must modify gamma first: kernel function relys on gamma
        if gamma == 'scale':
            self._kwargs['gamma'] = 1 / (X.shape[1] * X.var())
        elif gamma == 'auto':
            self._kwargs['gamma'] = 1 / X.shape[1]

        ## kernel: if user has not defined kernel function, find in `_Kernellib`
        ## key thinking: avoiding check the type of kernel function
        ## try to just return a packaged partial function contains only x1, x2 two params
        ## use `inspect.signature` to get possible params
        ## use `functools.partial` to get partial function
        if not callable(kernel):

            ## get kernel function from `_Kernellib` first
            kernel_func = _KernelLib.get_kernel(kernel)

            ## get params and their values from the signature of the kernel function
            kernel_func_params = inspect.signature(kernel_func).parameters

            ## build params dict
            ## the first two params is x1 and x2, ignore them
            params = {param: self._kwargs[param] for index, param in enumerate(kernel_func_params) if index >= 2}

            ## generate partial function
            ## check value restrictions of params here(check type restrictions while called)
            self._kwargs['kernel'] = functools.partial(kernel_func, **params)

        ## max_iter: if `max_iter` = -1, means infinity
        if max_iter == -1:
            self._kwargs['max_iter'] = float('inf')
        
    def _label_encode(self, X, y):
        '''Seperate data with K classes into K combinations, 
        each combination has a label encoded as +1, 
        and other labels encoded as -1
        '''
        
        labels = np.unique(y)
        total_indexs = {label: [] for label in labels}
        for index in range(len(X)):
            total_indexs[y[index]].append(index)
            
        ## separate k classes
        for label, indexs in total_indexs.items():
            encoded_y = y.copy()
            other_indexs = []
            for other_label, other_index in total_indexs.items():
                if other_label != label:
                    other_indexs += other_index
            encoded_y[indexs] = 1
            encoded_y[other_indexs] = -1
            yield encoded_y, label
                
    def fit(self, X, y):
        
        ## remap some params: calculate gamma, load right kernel and max_iter
        self._remap_params()
        
        ## train every single sub SVC
        for encoded_y, label in self._label_encode(X, y):
            sub_svc = _TwoClassesSVC(X=X, y=encoded_y, label=label, **self._kwargs)
            sub_svc.fit()
            self._sub_svcs.append(sub_svc)


class _KernelLib:
    '''Library of kernels
    pre-load four commun kernel, same as `sklearn.svm.SVC` 
    format of a kernel function:
        any_kernel_func(X1, X2, param1=1.0, param2=1.0, ...)
    keep the first two params are x1, x2, others are params with default values
    '''

    def check_params_wrapper(func):
        '''a decorator to check value restrictions of params
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
        '''get the needed kernel function
        if not in the lib, use the exception codes in `getattr`
        '''

        return getattr(_KernelLib, kernel_name)

    def linear(x1, x2):
        '''linear kernel'''

        return x1 @ x2

    @check_params_wrapper
    def poly(x1, x2, gamma=1.0, coef0=0.0, degree=1.0):
        '''poly kernel'''

        return np.power(gamma * linear(x1, x2) + coef0, degree)

    @check_params_wrapper
    def rbf(x1, x2, gamma=0.0):
        '''gaussian kernel'''

        dis = np.sum(np.power(x1 - x2, 2))
        return np.exp(- gamma * dis)

    @check_params_wrapper
    def sigmoid(x1, x2, gamma=1.0, coef0=0.0):
        '''Sigmoid kernel'''

        return np.tanh(gamma * linear(x1, x2) + coef0)