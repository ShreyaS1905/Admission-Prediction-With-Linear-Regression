import numpy as np

class KFold:

    def __init__(self,n_splits):
        self.n_splits = n_splits

    def split(self, X):
        """Splits the given numpy array into n_split parts

        Args:
            X (ndarray): Input Numpy array

        Returns:
            list: list of indices containing train and test split
        """
        sample_size = len(X)
        train_house = [a for a in range(sample_size)]
        fold_size = sample_size//self.n_splits
        test_inds = []
        train_splits = []
        test_splits = []
        for x in range(0,self.n_splits):
            temp = [x*fold_size,x*fold_size + fold_size]
            test_inds.append(temp)
        for x in test_inds:
            temp_test = [a for a in range(x[0],x[1])]
            temp_train = list(set(train_house)^set(temp_test))
            train_splits.append(np.array(temp_train))
            test_splits.append(np.array(temp_test))
        op = []
        for i in range(0,self.n_splits):
            op.append((train_splits[i],test_splits[i]))
        
        return op