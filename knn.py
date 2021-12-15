import collections
import queue

from base import _Core, _Machine


class _KDTreeNode(_Core):
    def __init__(self, upper_class, input_row_index, depth):
        super().__init__(upper_class=upper_class)
        self.left = None
        self.right = None
        self.another_side_is_checked = False

        ## decide the split col index
        self.col_index = depth % self.upper_class.ncol

        ## sort the index of data according the split col
        sorted_index = sorted(input_row_index, key=lambda i: self.upper_class.X[i, self.col_index])
        n_data = len(sorted_index)

        ## split them, define the right-medium as medium
        self.row_index = sorted_index[n_data // 2]
        self.label = self.upper_class.y[self.row_index]
        if n_data == 1:
            return
        elif n_data == 2:
            left_input_row_index = sorted_index[: n_data // 2]
            self.left = self.__class__(upper_class=self.upper_class, input_row_index=left_input_row_index, depth=depth+1)
        else:
            left_input_row_index, right_input_row_index = sorted_index[: n_data // 2], sorted_index[n_data // 2 + 1:]
            self.left = self.__class__(upper_class=self.upper_class, input_row_index=left_input_row_index, depth=depth+1) 
            self.right = self.__class__(upper_class=self.upper_class, input_row_index=right_input_row_index, depth=depth+1)
    
    def __repr__(self):
        return f'KDTreeNode(x={self.upper_class.X[self.row_index]}, y={self.label})'


class MyKNC(_Machine):
    def __init__(self, n=5, p=2, algorithm='kd_tree'):
        super().__init__(n=n, p=p, algorithm=algorithm)

        ## we need to calculate reciprocal sometimes, set presicion avolding ZeroDivisionError
        self.epsilon = 1e-7

    ## private methods
    @staticmethod
    def _get_most_arg(column):
        return max(collections.Counter(column).items(), key=lambda x: x[1])[0]

    def _dis_between_two_dots(self, index1, x2):
        x1 = self.X[index1]
        result = np.power(np.sum(np.power(x1 - x2, self.p)), 1 / self.p)
        if result == 0:
            return result + self.epsilon
        return result

    def _dis_from_boundary(self, row_index, col_index, x):
        result = np.abs(x[col_index] - self.X[row_index, col_index])
        if result == 0:
            return result + self.epsilon
        return result


    def _remap_params(self):
        '''Check value only, let lower codes to check type automatically'''

        ## check n
        try:
            assert self._params['n'] > 0
        except AssertionError:
            raise ValueError('The `n` must be positive!')
        self.n = self._params['n']

        ## check p
        try:
            assert self._params['p'] > 0
        except AssertionError:
            raise ValueError('The `p` must be positive!')
        self.p = self._params['p']

    def _single_predict(self, x):
        return getattr(self, f'_single_{self.algorithm}_predict')(x)

    def _single_kd_tree_predict(self, x):
        node_stack = [self.kd_tree]

        ## find the nearest quasi-leaf node
        ## definition of quasi-leaf node: 
        ## 1) if x no more than tmp, tmp.left is None
        ## 2) if x more than tmp, tmp.right is None
        while node_stack:
            tmp = node_stack[-1]
            if x[tmp.col_index] <= self.X[tmp.row_index, tmp.col_index]:
                if tmp.left is not None:
                    node_stack.append(tmp.left)
                else:
                    break
            else:
                if tmp.right is not None:
                    node_stack.append(tmp.right)
                else:
                    break

        ## record the moving down path, see usage below
        moving_down_path_nodes = node_stack[:]

        ## traceback
        dis_pqueue = queue.PriorityQueue(maxsize=self.n)
        reciprocal_of_max_distance = float('inf')
        while node_stack:
            tmp = node_stack[-1]

            ## calculate the distance between tmp and x, and try to put it in the pqueue
            dis_from_node = self._dis_between_two_dots(tmp.row_index, x)

            ## 1. upgrade the pqueue of distance
            ## if the pqueue is not full, put tmp in directly
            if not dis_pqueue.full():
                dis_pqueue.put((1 / dis_from_node, tmp))
                reciprocal_of_max_distance = min(reciprocal_of_max_distance, 1 / dis_from_node)

            ## if the pqueue is full and current distance is less than max_distance
            ## pull out the max-distance node and put tmp in the stack
            elif 1 / dis_from_node > reciprocal_of_max_distance:
                dis_pqueue.get()
                dis_pqueue.put((1 / dis_from_node, tmp))

                ## upgrade max distance
                ## use the stupid way: get and put because the PriorityQueue does not has the method `peak()`
                curr_reciprocal_of_max_dis, curr_max_node = dis_pqueue.get()
                reciprocal_of_max_distance = max(reciprocal_of_max_distance, curr_reciprocal_of_max_dis)
                dis_pqueue.put((curr_reciprocal_of_max_dis, curr_max_node))

            ## 2. decide the next tmp mode
            node_stack.pop()
            if tmp.left is not None or tmp.right is not None:
                
                ## if the pqueue is not full or the distance from the boundary is not further, we need to check the other side
                if not dis_pqueue.full() or 1 / self._dis_from_boundary(tmp.row_index, tmp.col_index, x) >= reciprocal_of_max_distance:

                    ## if tmp in the path of moving down, we need to put child on the other side into shack
                    ## if not, put child on this side into stack
                    ## of course, the child must exist
                    if tmp in moving_down_path_nodes:
                        if x[tmp.col_index] <= self.X[tmp.row_index, tmp.col_index] and tmp.right is not None:
                            node_stack.append(tmp.right)
                        elif x[tmp.col_index] > self.X[tmp.row_index, tmp.col_index] and tmp.left is not None:
                            node_stack.append(tmp.left)                            
                    else:
                        tmp.another_side_is_checked = True
                        if x[tmp.col_index] <= self.X[tmp.row_index, tmp.col_index] and tmp.left is not None:
                            node_stack.append(tmp.left)
                        elif x[tmp.col_index] > self.X[tmp.row_index, tmp.col_index] and tmp.right is not None:
                            node_stack.append(tmp.right)

        ## get the most frequently label
        label_list = []
        while not dis_pqueue.empty():
            label_list.append(dis_pqueue.get()[1].label)
        return self.__class__._get_most_arg(label_list)

    ## public attributes
    @property
    def is_fitted(self) -> bool:
        '''Judge the model is fitted or not'''

        return hasattr(self, 'algorithm')

    ## public methods
    def fit(self, X, y):
        self._remap_params()
        self.X = X
        self.y = y
        self.nrow, self.ncol = tuple(self.X.shape)

        ## set subtree first, then set attribute
        ## because we use the attribute `algorithm` to check the model is fitted or not
        if self._params['algorithm'] == 'kd_tree':
            self.kd_tree = _KDTreeNode(upper_class=self, input_row_index=np.arange(len(self.X)), depth=0)
            self.algorithm = self._params['algorithm']
        elif self._params['algorithm'] == 'ball_tree':
            self.ball_tree = NotImplementedError
            self.algorithm = self._params['algorithm']
        elif self._params['algorithm'] == 'brute':
            self.brute = NotImplementedError
            self.algorithm = self._params['algorithm']
        elif self._params['algorithm'] == 'auto':

            ## if n < N / 2: use kdtree; if not, use brute method is better
            if self.n < self.nrow / 2:
                self.kd_tree = _KDTreeNode(upper_class=self, input_row_index=np.arange(len(self.X)), depth=0)
                self.algorithm = 'kd_tree'
            else:
                self.brute = NotImplementedError
                self.algorithm = 'brute'
        else:
            raise ValueError('`algorithm` must be one of kd_tree/ball_tree/brute/auto!')

    def predict(self, X):
        return np.apply_along_axis(lambda x: self._single_predict(x), axis=1, arr=X)