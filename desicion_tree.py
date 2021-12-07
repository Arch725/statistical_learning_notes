import collections
import copy

from base import _Core, _Machine


class _TreeNode(_Core):
    '''A Desicion Tree'''

    def __init__(self, upper_class, X, y, depth, features):
        super().__init__(upper_class=upper_class, X=X, y=y, depth=depth, features=features)

    ## attributes
    @property
    def is_leaf(self):
        return self.num_children == 0

    @property
    def branches(self):
        if self.is_leaf:
            return iter(())
        for child in self.children:
            for branch in child.branches:
                yield branch
        yield self

    @property
    def num_leaves(self):
        if self.is_leaf:
            return 1
        return sum(child.num_leaves for child in self.children)

    @property
    def leaves(self):
        if self.is_leaf:
            yield self
        for child in self.children:
            for leaf in child.leaves:
                yield leaf

    @property
    def size(self):
        return len(self.y)

    ## methods
    def __len__(self):
        return sum(len(child) for child in self.children) + 1

    def __repr__(self):
        if not hasattr(self, 'output_name'):
            self.output_name = self.__class__.__name__[1:].replace('_', '.')
        return self.output_name


class _ID3TreeNode(_TreeNode):
    '''ID3 desicion tree'''

    def __init__(self, upper_class, X, y, depth, features):
        super().__init__(upper_class=upper_class, X=X, y=y, depth=depth, features=features)
        self.label = None
        self.entropy = None
        self.split_col_index = None
        self._children = {}
        self.pruning_checked = False

    @property
    def num_children(self):
        return len(self._children)

    @property
    def children(self):
        for child in self._children.values():
            yield child

    def __repr__(self):
        return f'{super().__repr__()}(entropy={self.entropy}, label={self.label})'

    def built(self):

        ## calculate unconditional entropy
        self.entropy = _Tools.entropy(self.y)

        ## check the restrict: max_depth and min_samples
        if self.depth == self.upper_class.max_depth or self.size <= self.upper_class.min_samples:
            self.label = _Tools.cal_most_freq(self.y)
            return

        ## find min conditional entropy(max info gain)
        min_cond_entropy, best_col_index = self.entropy, None
        for col_index in range(self.X.shape[1]):
            x = self.X[:, col_index]
            cond_entropy = _Tools.cond_entropy(x, self.y)
            if cond_entropy <= min_cond_entropy:
                min_cond_entropy, best_col_index = cond_entropy, col_index

        ## check the restrict: min_impurity_decrease
        if self.entropy - min_cond_entropy <= self.upper_class.min_impurity_decrease:
            self.label = _Tools.cal_most_freq(self.y)
            return

        ## split children nodes
        self.split_col_index = best_col_index
        best_col = self.X[:, self.split_col_index]
        for value, size in collections.Counter(best_col).items():

            ## check min_split_samples
            if size < self.upper_class.min_split_samples:
                self._children = {}
                return

            ## choose several columns
            children_features = self.upper_class._choice_features(data=np.arange(self.X.shape[1]), except_elem=self.split_col_index)
            child_node = self.__class__(
                upper_class=self.upper_class, 
                X=self.X[best_col == value][:, children_features], 
                y=self.y[best_col == value], 
                depth=self.depth + 1, 
                features=children_features
            )
            self._children[value] = child_node

    def empirial_risk_diff(self):
        leaves_empirial_risk = 0
        for leaf in self.leaves:
            leaves_empirial_risk += leaf.entropy * leaf.size / self.size
        return leaves_empirial_risk - self.entropy + self.upper_class.ccp_alpha * (self.num_leaves - 1)

    def pruned(self):
        self.label = _Tools.cal_most_freq(self.y)
        self.split_col_index = None
        self._children = {}

    def get_child(self, x):

        ## if the value of x not in self.children, return a children node randomly
        if x.ndim == 1:
            return x[self.features], self._children.get(
                x[self.features][self.split_col_index],
                self.upper_class._choice_children(list(self._children.values()))[0]
            )
        return x[self.features], self._children.get(
            x[self.features],
            self.upper_class._choice_children(list(self._children.values()))[0]
        )


class _C4_5TreeNode(_TreeNode):
    '''C4.5 desicion tree'''

    def __init__(self, upper_class, X, y, depth, features):
        super().__init__(upper_class=upper_class, X=X, y=y, depth=depth, features=features)
        self.label = None
        self.entropy = None
        self.split_col_index = None
        self._children = {}
        self.pruning_checked = False

    @property
    def num_children(self):
        return len(self._children)

    @property
    def children(self):
        for child in self._children.values():
            yield child

    def __repr__(self):
        return f'{super().__repr__()}(entropy={self.entropy}, label={self.label})'

    def built(self):

        ## calculate unconditional entropy
        self.entropy = _Tools.entropy(self.y)

        ## same as ID3
        if self.depth == self.upper_class.max_depth or self.size <= self.upper_class.min_samples:
            self.label = _Tools.cal_most_freq(self.y)
            return

        ## find conditional entropy(max info gain), which is not less than the average
        cond_entropies = np.zeros(self.X.shape[1])
        for col_index in range(self.X.shape[1]):
            x = self.X[:, col_index]
            cond_entropies[col_index] = _Tools.cond_entropy(x, self.y)
        avg_cond_entropy = np.mean(cond_entropies)

        ## find max info gain ratio
        info_gain_ratios = {}
        for col_index in range(self.X.shape[1]):
            if cond_entropies[col_index] <= avg_cond_entropy:
                x = self.X[:, col_index]
                info_gain_ratios[col_index] = (self.entropy - cond_entropies[col_index]) / _Tools.inherent_value(x)
        best_col_index, max_info_gain_ratio = max(info_gain_ratios.items(), key=lambda x: x[1])
        
        ## check the restrict: min_impurity_decrease
        if self.entropy - cond_entropies[best_col_index] <= self.upper_class.min_impurity_decrease:
            self.label = _Tools.cal_most_freq(self.y)
            return

        ## split children nodes
        self.split_col_index = best_col_index
        best_col = self.X[:, self.split_col_index]
        for value, size in collections.Counter(best_col).items():

            ## check min_split_samples
            if size < self.upper_class.min_split_samples:
                self._children = {}
                return

            ## choose several columns
            children_features = self.upper_class._choice_features(data=np.arange(self.X.shape[1]), except_elem=self.split_col_index)
            child_node = self.__class__(
                upper_class=self.upper_class, 
                X=self.X[best_col == value][:, children_features], 
                y=self.y[best_col == value], 
                depth=self.depth + 1, 
                features=children_features
            )
            self._children[value] = child_node

    def empirial_risk_diff(self):
        leaves_empirial_risk = 0
        for leaf in self.leaves:
            leaves_empirial_risk += leaf.entropy * leaf.size / self.size
        return leaves_empirial_risk - self.entropy + self.upper_class.ccp_alpha * (self.num_leaves - 1)

    def pruned(self):
        self.label = _Tools.cal_most_freq(self.y)
        self.split_col_index = None
        self._children = {}

    def get_child(self, x):

        ## same as ID3
        if x.ndim == 1:
            return x[self.features], self._children.get(
                x[self.features][self.split_col_index],
                self.upper_class._choice_children(list(self._children.values()))[0]
            )
        return x[self.features], self._children.get(
            x[self.features],
            self.upper_class._choice_children(list(self._children.values()))[0]
        )


class _CartCTreeNode(_TreeNode):
    '''Cart classify desicion tree'''

    def __init__(self, upper_class, X, y, depth, features):
        super().__init__(upper_class=upper_class, X=X, y=y, depth=depth, features=features)
        self.label = None
        self.gini = None
        self.split_col_index = None
        self.split_value = None
        self.left = None
        self.right = None

    @property
    def num_children(self):
        if self.left is None and self.right is None:
            return 0
        return 2

    @property
    def children(self):
        if self.is_leaf:
            return iter(())
        yield self.left
        yield self.right

    def __repr__(self):
        return f'{super().__repr__()}(gini={self.gini}, label={self.label})'

    def built(self):

        ## calculate unconditional gini
        self.gini = _Tools.gini(self.y)

        ## same as ID3
        if self.depth == self.upper_class.max_depth or self.size <= self.upper_class.min_samples:
            self.label = _Tools.cal_most_freq(self.y)
            return

        ## find min conditional gini
        min_cond_gini, best_col_index, best_value = self.gini, None, None
        for col_index in range(self.X.shape[1]):
            x = self.X[:, col_index]
            for value in np.unique(x):
                left_filter = x == value
                right_filter = x != value
                cond_gini = _Tools.cond_gini(left_filter, right_filter, self.y)
                if cond_gini <= min_cond_gini:
                    min_cond_gini, best_col_index, best_value = cond_gini, col_index, value

        ## check the restrict: min_impurity_decrease
        if self.gini - min_cond_gini <= self.upper_class.min_impurity_decrease:
            self.label = _Tools.cal_most_freq(self.y)
            return

        ## split children nodes
        self.split_col_index = best_col_index
        self.split_value = best_value
        best_col = self.X[:, self.split_col_index]
        for child_name in ['left', 'right']:
            if child_name == 'left':
                row_filter = best_col == self.split_value
            else:
                row_filter = best_col != self.split_value

            ## check min_split_samples
            size = np.sum(row_filter)
            if size < self.upper_class.min_split_samples:
                self.left = None
                return

            ## choose several columns
            children_features = self.upper_class._choice_features(data=np.arange(self.X.shape[1]))
            child_node = self.__class__(
                upper_class=self.upper_class, 
                X=self.X[row_filter][:, children_features], 
                y=self.y[row_filter], 
                depth=self.depth + 1, 
                features=children_features
            )
            setattr(self, child_name, child_node)

    def critical_alpha(self):
        leaves_empirial_risk = 0
        for leaf in self.leaves:
            leaves_empirial_risk += leaf.gini * leaf.size / self.size
        return (self.gini - leaves_empirial_risk) / (self.num_leaves - 1)

    def pruned(self):
        self.label = _Tools.cal_most_freq(self.y)
        self.split_col_index = None
        self.split_value = None
        self.left = None
        self.right = None

    def get_child(self, x):
        featured_x = x[self.features]
        if featured_x.ndim == 1:
            if features_x[self.split_col_index] == self.split_value:
                return featured_x, self.left
            return featured_x, self.right
        else:
            if features_x == self.split_value:
                return featured_x, self.left
            return featured_x, self.right

class _CartRTreeNode(_TreeNode):
    '''Cart regression desicion tree'''

    def __init__(self, upper_class, X, y, depth, features):
        super().__init__(upper_class=upper_class, X=X, y=y, depth=depth, features=features)
        self.pred = None
        self.square = None
        self.split_col_index = None
        self.split_value = None
        self.left = None
        self.right = None

    @property
    def num_children(self):
        if self.left is None and self.right is None:
            return 0
        return 2

    @property
    def children(self):
        if self.is_leaf:
            return iter(())
        yield self.left
        yield self.right

    def __repr__(self):
        return f'{super().__repr__()}(square={self.square}, pred={self.pred})'

    def built(self):

        ## calculate unconditional square
        self.square = _Tools.square(self.y)

        ## same as ID3
        if self.depth == self.upper_class.max_depth or self.size <= self.upper_class.min_samples:
            self.pred = _Tools.cal_mean(self.y)
            return

        ## find min conditional square
        min_cond_square, best_col_index, best_value = self.square, None, None
        for col_index in range(self.X.shape[1]):
            x = np.sort(self.X[:, col_index])

            ## calculate n - 1 median value of x(len(x) must >= 2)
            for left_value, right_value in zip(x[:-1], x[1:]):
                value = (left_value + right_value) / 2
                left_filter = x <= value
                right_filter = x > value
                cond_square = _Tools.cond_square(left_filter, right_filter, self.y)
                if cond_square <= min_cond_square:
                    min_cond_square, best_col_index, best_value = cond_square, col_index, value

        ## check the restrict: min_impurity_decrease
        if self.square - min_cond_square <= self.upper_class.min_impurity_decrease:
            self.pred = _Tools.cal_mean(self.y)
            return

        ## split children nodes
        self.split_col_index = best_col_index
        self.split_value = best_value
        best_col = self.X[:, self.split_col_index]
        for child_name in ['left', 'right']:
            if child_name == 'left':
                row_filter = best_col <= self.split_value
            else:
                row_filter = best_col > self.split_value

            ## check min_split_samples
            size = np.sum(row_filter)
            if size < self.upper_class.min_split_samples:
                self.left = None
                return

            ## choose several columns
            children_features = self.upper_class._choice_features(data=np.arange(self.X.shape[1]))
            child_node = self.__class__(
                upper_class=self.upper_class, 
                X=self.X[row_filter][:, children_features], 
                y=self.y[row_filter], 
                depth=self.depth + 1, 
                features=children_features
            )
            setattr(self, child_name, child_node)

    def critical_alpha(self):
        leaves_empirial_risk = 0
        for leaf in self.leaves:
            leaves_empirial_risk += leaf.square
        return (self.square - leaves_empirial_risk) / (self.num_leaves - 1)

    def pruned(self):
        self.pred = _Tools.cal_mean(self.y)
        self.split_col_index = None
        self.split_value = None
        self.left = None
        self.right = None

    def get_child(self, x):
        featured_x = x[self.features]
        if featured_x.ndim == 1:
            if features_x[self.split_col_index] <= self.split_value:
                return featured_x, self.left
            return featured_x, self.right
        else:
            if features_x <= self.split_value:
                return featured_x, self.left
            return featured_x, self.right


class _Tools:
    '''A toolbox to do math calculates'''

    @staticmethod
    def cal_most_freq(y):
        value_counts = collections.Counter(y)
        return max(value_counts.items(), key=lambda x: x[1])[0]

    @staticmethod
    def cal_mean(y):
        return np.mean(y)

    @staticmethod
    def entropy(y):
        value_counts = collections.Counter(y)
        return -sum(map(lambda x: x * np.log2(x / len(y)), value_counts.values()))

    @staticmethod
    def cond_entropy(x, y):
        cond_entropy = 0
        for value, size in collections.Counter(x).items():
            row_filter = x == value
            cond_entropy += sum(row_filter) / size * _Tools.entropy(y[row_filter])
        return cond_entropy

    @staticmethod
    def inherent_value(x):
        return _Tools.entropy(x)

    @staticmethod
    def gini(y):
        value_counts = collections.Counter(y)
        return 1 - sum(map(lambda x: np.power(x / len(y), 2), value_counts.values()))

    @staticmethod
    def cond_gini(left_filter, right_filter, y):
        left_size, right_size = sum(left_filter), sum(right_filter)
        size = left_size + right_size
        return left_size / size * _Tools.gini(y[left_filter])\
        + right_size / size * _Tools.gini(y[right_filter])

    @staticmethod
    def square(y):
        return np.sum(np.power(y, 2)) - len(y) * np.power(np.mean(y), 2)

    @staticmethod
    def cond_square(left_filter, right_filter, y):
        return _Tools.square(y[left_filter]) + _Tools.square(y[right_filter])


class MyDecisionTree(_Machine):
    def __init__(
        self, tree_type='CART', mode='classifier', max_depth=None, ccp_alpha=0.0,
        min_split_samples=1, min_samples=2, max_features=None, random_state=None, 
        min_impurity_decrease=0.0, max_num_leaves=None, valid_size=0.3
    ):
        super().__init__(
            tree_type=tree_type, mode=mode, max_depth=max_depth, ccp_alpha=ccp_alpha, 
            min_split_samples=min_split_samples, min_samples=min_samples, 
            max_features=max_features, random_state=random_state, 
            min_impurity_decrease=min_impurity_decrease, max_num_leaves=max_num_leaves, 
            valid_size=valid_size
        )

    ## public property
    @property
    def is_fitted(self) -> bool:
        '''Judge the model is fitted or not'''

        if hasattr(self, '_tree'):
            return True
        raise warnings.warn('The model is not fitted.')
        return False

    @property
    def params(self):
        return self._params

    ## private methods
    def _remap_params(self):

        ## max_depth
        if self._params['max_depth'] is None:
            self.max_depth = float('inf')

        ## max_num_leaves
        if self._params['max_num_leaves'] is None:
            self.max_num_leaves = float('inf')

        ## do not save max_features directly, but save a mapping
        ## because features will decrease in ID3 and C4.5 trees
        if self._params['max_features'] is None:
            self._get_max_features = lambda x: x
        elif self._params['max_features'] == 'auto' or self._params['max_features'] == 'sqrt':
            self._get_max_features = np.sqrt
        elif self._params['max_features'] == 'log2':
            self._get_max_features = np.log2
        else:
            self._get_max_features = lambda x: min(x, self._params['max_features'])

        ## random_state
        if self._params['random_state'] is None:
            self._rng = np.random.RandomState()
        elif hasattr(self._params['random_state'], rand):
            self._rng = self._params['random_state']
        else:
            self._rng = np.random.RandomState(self._params['random_state'])
        
        ## set other params
        self.max_depth = self._params['max_depth']
        self.ccp_alpha = self._params['ccp_alpha']
        self.min_split_samples = self._params['min_split_samples']
        self.min_samples = self._params['min_samples']
        self.min_impurity_decrease = self._params['min_impurity_decrease']
        self.valid_size = self._params['valid_size']
  
    def _choice_features(self, data, except_elem=None):
        if except_elem is not None:
            data = [item for item in data if item != except_elem]
        return self._rng.choice(data, size=self._get_max_features(len(data)), replace=False)

    def _choice_children(self, children_list):
        return self._rng.choice(children_list, size=1)

    def _build(self):
        curr_num_leaves = 0
        build_queue = [self._tree]
        while build_queue:
            tmp = build_queue.pop(0)

            ## since we do not know how many children the tmp are before let it built,
            ## we build the node first, if num of leaves is too large, prune it later
            tmp.built()
            new_num_leaves = tmp.num_leaves

            ## if num of children more than limit, prune it and let next node trying to be built
            if curr_num_leaves + new_num_leaves > self.max_num_leaves:
                tmp.pruned()

            ## if equals to limit, the building procedure will stop
            elif curr_num_leaves + new_num_leaves == self.max_num_leaves:
                curr_num_leaves += new_num_leaves
                break

            ## if more than limit, the building procedure will continue
            else:
                curr_num_leaves += new_num_leaves

                for tmp_child in tmp.children:
                    build_queue.append(tmp_child)

    def _prune(self):
        '''Prune the tree'''
        
        ## if the tree is leaf, end the pruning procedure
        if self._tree.is_leaf:
            return

        ## ID3 and C4.5 tree need ccp_alpha
        if hasattr(self._tree, '_children'):
            node_stack = [self._tree]
            while node_stack:
                tmp = node_stack[-1]

                ## two type of nodes is needed to check immediately:
                ## 1. all children are leaves
                ## 2. all children are checked
                all_children_are_leaves = True
                all_children_are_pruning_checked = True
                for child in tmp.children:
                    if not child.is_leaf and not child.pruning_checked:
                        node_stack.append(child)
                        all_children_are_leaves = False
                        all_children_are_pruning_checked = False

                ## if all the children nodes are leaves
                ## the tmp node will be pruned or safe(will not be checked again)
                ## so pop it anyway
                ## if all the children nodes are checked, time to check this node
                if all_children_are_pruning_checked or all_children_are_leaves:
                    node_stack.pop()

                    ## if empirical risk diff >= 0
                    ## means there is no need to split this node, prune them
                    if tmp.empirial_risk_diff() >= 0:
                        tmp.pruned()
                    tmp.pruning_checked = True

                ## if some children nodes are not leaves
                ## the tmp node may be checked later(if its children nodes are all pruned, it also may be pruned)
                ## so it can not be popped from the stack
                else:
                    pass

        ## the strategy of CART tree is different
        ## do not need `ccp_alpha`, but return a series of sub pruned trees
        ## let the outer GridCV to decide which is the best
        else:
            nested_sub_trees = []
            alpha = 0.0
            root = self._tree
            while not root.is_leaf:

                ## get every non-leaf node and its critical alpha, ascending sorted them
                ## some node may have same critical alpha
                min_critical_alpha = float('inf')
                pruned_branches = []
                for node in root.branches:
                    critical_alpha = node.critical_alpha()
                    if critical_alpha < min_critical_alpha:
                        pruned_branches = [node]
                        min_critical_alpha = critical_alpha
                    elif critical_alpha == min_critical_alpha:
                        pruned_branches.append(node)

                ## prune them
                ## return a nested sub trees: {sub_tree: (min_alpha, max_alpha)}
                ## the interval is left closed and right open: [min_alpha, max_alpha)
                for node in pruned_branches:
                    node.pruned()
                nested_sub_trees.append((copy.copy(root), (alpha, min_critical_alpha)))
                alpha = min_critical_alpha

            ## now the root is only a leaf node
            nested_sub_trees.append((copy.copy(root), (alpha, float('inf'))))

            ## find the best tree
            best_tree = None

            ## for the classifier tree, we need the argmax accuracy
            if hasattr(self._tree, 'label'):
                max_accuracy = 0
                for tree, (min_alpha, max_alpha) in nested_sub_trees:
                    accuracy = self.score(self._X_valid, self._y_valid, score_type='accuracy')
                    if accuracy >= max_accuracy:
                        best_tree, best_alphas, max_accuracy = tree, (min_alpha, max_alpha), accuracy

            ## for the regressor tree, we need the argmin square
            else:
                min_square = float('inf')
                for tree, (min_alpha, max_alpha) in nested_sub_trees:
                    square = self.score(self._X_valid, self._y_valid, score_type='square')
                    if square <= min_square:
                        best_tree, best_alphas, min_square = tree, (min_alpha, max_alpha), square
            self._tree = best_tree
            self._best_alphas = best_alphas

    def _single_predict(self, x):

        ## copy x, because rows and columns of X is changing during spliting tree
        copy_x, tmp = x.copy(), self._tree
        while not tmp.is_leaf:
            copy_x, tmp = tmp.get_child(copy_x)
        if hasattr(self._tree, 'label'):
            return tmp.label
        else:
            return tmp.pred
        
    ## public methods
    def fit(self, X, y):

        ## remap params
        self._remap_params()

        ## set tree
        features = self._choice_features(data=np.arange(X.shape[1]))
        if self._params['tree_type'] == 'ID3':
            self._tree = _ID3TreeNode(upper_class=self, X=X, y=y, depth=0, features=features)
        elif self._params['tree_type'] == 'C4.5':
            self._tree = _C4_5TreeNode(upper_class=self, X=X, y=y, depth=0, features=features)
        elif self._params['tree_type'] == 'CART':
            X_train, X_valid, y_train, y_valid = train_test_split(
                X, y, test_size=self._params['valid_size'], random_state=self._rng, stratify=y
            )
            self._X_valid, self._y_valid = X_valid, y_valid
            if self._params['mode'] == 'classifier':
                self._tree = _CartCTreeNode(upper_class=self, X=X_train, y=y_train, depth=0, features=features)
            elif self._params['mode'] == 'regressor':
                self._tree = _CartRTreeNode(upper_class=self, X=X_train, y=y_train, depth=0, features=features)
            else:
                raise ValueError(f'tree_type must be classifier/regressor!')
        else:
            raise ValueError(f'tree_type must be ID3/C4.5/CART!')

        ## building the tree
        self._build()
        
        ## pruning the tree
        self._prune()

    def predict(self, X):
        return np.apply_along_axis(lambda x: self._single_predict(x), axis=1, arr=X)

    def score(self, X, y, score_type=None):
        if score_type is None:
            if hasattr(self._tree, 'label'):
                score_type = 'accuracy'
            else:
                score_type = 'square'
        y_pred = self.predict(X)
        if score_type == 'accuracy':
            return np.sum(y_pred == y) / len(y)
        elif score_type == 'square':
            return np.var(y_pred - y) * len(y)