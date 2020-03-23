from sklearn import metrics
from sklearn.model_selection import GroupKFold, GridSearchCV, TimeSeriesSplit
import pandas as pd
import numpy as np

    
class CrossValidation:

    """
    Setup:
    >>> X, y = [], param_grid = {}
    >>> evaler = CrossValidation(estimator, X, y)
    >>> evaler = CrossValidation(estimator, X, y, param_grid)
    Get Splits:
    >>> groups = []
    >>> splits = evaler.get_grouped_cv_splits(n_splits, groups)
    >>> splits = evaler.get_time_ts_cv_splits(n_splits, date_col = "date")
    Grid Search:
    >>> evaler.grid_search()
    >>> evaler.grid_search(3)
    >>> evaler.grid_search(splits)
    """

    def __init__(self, estimator, X, y, param_grid={}, **kwargs):
        """
        Arguments:
            estimator sklearn.estimator -- model that hasn't yet been fit
            X {nxm np array or pandas df} -- features
            y {nxp np array or pandas df} -- targets
        
        Keyword Arguments:
            param_grid {dict} -- [if {} just normal cv, otherwise same as param_grid input to gridsearch skleans] (default: {{}})
        """
        self.estimator = estimator
        self.X = X
        self.y = y
        self.param_grid = param_grid
        self.splits = None


    def get_grouped_cv_splits(self, n_splits, groups):
        """Perform cross val with GroupKFold
        
        Arguments:
            n_splits {int}
            groups: array-like with shape (n_samples,) 
                - number of distinct groups has to be at least equal to the number of folds
        
        Returns:
           Indices of training and test set
        """
        
        assert (n_splits >= 2)
        assert len(groups) == len(self.y)
        assert len(set(groups)) >= n_splits, "The number of distinct groups has to be at least equal to the number of folds"

        self.splits = list(GroupKFold(n_splits=n_splits).split(self.X, self.y, groups))
        return self.splits


    def get_time_ts_cv_splits(self, n_splits, date_col = "date"):
        """Perform cross val with TimeSeriesSplit on time/date based data.
        Garentees that the time/date for validation data comes after time/date for training data
        REQUIRED: X needs to be transformed to be the type pandas.DataFrame
        
        Arguments:
            n_splits {int}
            date_col {string} -- name of the column in the dataframe which we will perform time series split on
        Returns:
           Indices of training and test set respecting the time garantee
        """

        assert isinstance(self.X, pd.DataFrame), f"get_time_ts_cv_splits require training X to be the type <pandas.DataFrame>, got {type(self.X)}"
        X_reset = self.X.copy().reset_index(drop=True)

        u, indices = np.unique(X_reset[date_col], return_index = True)
        split_X = X_reset.loc[indices, [date_col]].reset_index(drop=True)
        time_tscv = list(TimeSeriesSplit(n_splits=n_splits).split(split_X))

        for i, val in enumerate(time_tscv):
            train_index, test_index = val[0], val[1]
            X_for_train, X_for_test = split_X.loc[train_index], split_X.loc[test_index]

            X_for_train["join_guard"] = 1
            train_merged = pd.merge(X_reset, X_for_train, on=date_col, how="left")
            assert(X_reset.shape[0] == train_merged.shape[0])
            train_indicies = (self.X[ train_merged["join_guard"].notna() ].index).to_numpy()

            X_for_test["join_guard"] = 1
            test_merged = pd.merge(X_reset, X_for_test, on=date_col, how="left")
            assert(X_reset.shape[0] == test_merged.shape[0])
            test_indicies = (self.X[ test_merged["join_guard"].notna() ].index).to_numpy()

            time_tscv[i] = (train_indicies, test_indicies)

        self.splits = time_tscv
        return time_tscv


    def grid_search(self, cv=None, verbose=0, path=None):
        """Perform Grid Search 
        
        Arguments:
            cv {int or sklearn.cross_val} -- if int, will perform normal cv
                                             otherwise can be outputs from get_grouped_cv_splits or get_ts_cv_splits
        
        Returns:
            sklearn grid search object -- see https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
        """
        if cv is None and self.splits is not None:
            self.print('Inheriting splits from self.splits')
            cv = self.splits
        grid_search = GridSearchCV(
            estimator=self.estimator,
            param_grid=self.param_grid,
            cv=cv,
            verbose=verbose,
            n_jobs=-1,
            refit=True,
            return_train_score=True
        )
        print('Running grid search...')
        result = grid_search.fit(self.X, self.y)
        df_results = pd.DataFrame(result.cv_results_)
        df_results.sort_values(by=['mean_test_score'], ascending=False, inplace=True)
        df_results.reset_index(inplace=True)
        if path is not None:
            df_results.to_csv(path)

        return grid_search.best_estimator_, result.best_estimator_.feature_importances_,grid_search.best_estimator_.score