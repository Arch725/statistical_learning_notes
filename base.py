class _Core:
    '''A template for the core component(s) of the classifier/regressor 
    class(hereinafter referred to as "upper class"), the component can
    (e.g. a two classes classifier in a One-VS-Rest multiclassifier) or
    can not (e.g. a tree in a decision tree) do the same works(`fit()`, 
    `predict()`, ..., etc) like the upper class.
    '''

    def __init__(self, **kwargs):
        '''**kwargs contain:
        1. upper class itself. Pass it to the core class because the core class will use 
            its methods or attributes as public toolbox, it will be too redundant if the 
            upper class pass many same methods or attributes repeatly to each core class,
            let the core class itself to use the public toolbox by codes like 
            `self.upper_class.public_tool()`.
        2. data: X, y, which are passed by `fit()` of the upper class.
        3. parameters: Users pass them to the upper class, and the upper class passes them here.
        '''

        ## iter `locals()` to avoid enumerating each argument
        for arg_name, arg_val in locals()['kwargs'].items():
            setattr(self, arg_name, arg_val)


class _Machine():
    '''A template for the classifier/regressor class'''

    def __init__(self, **kwargs):

        ## save **kwargs to a dict for remapping later
        self._params = locals()['kwargs']

    ## public attributes
    @property
    def is_fitted(self) -> bool:
        '''Judge the model is fitted or not'''

        raise NotImplementedError

    @property
    def params(self) -> dict:
        return self._params 

    ## public method
    def fit(self, X, y):
        '''Fit the model with given X and y'''

        ## set one or more lower class to work actually using class `_Core`

        raise NotImplementedError

    def predict(self, X):
        '''Predict with given X'''

        raise NotImplementedError
