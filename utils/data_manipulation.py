import numpy as np
import math


def shuffle_data(X,y,seed=None):
    if seed:
        np.random.seed(seed)
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    return X[idx],y[idx]

def batch_iterator(X,y=None,batch_size=32):
    """Simple batch generator"""
    n_samples = X.shape[0]
    for i in np.arange(0,n_samples,batch_size):
        begin,end = i,min(i+batch_size,n_samples)
        if y is not None:
            yield X[begin:end],y[begin:end]
        else:
            yield X[begin:end]

def train_test_split(X,y,test_size=0.2,shuffle=True,seed=36):
    if shuffle:
        X,y = shuffle_data(X,y,seed)
    split_i = len(y) - int(len(y) // (1 / test_size))
    X_train,X_test = X[:split_i],X[split_i:]
    y_train,y_test = y[:split_i],y[split_i:]
    return np.array(X_train),np.array(X_test),np.array(y_train),np.array(y_test)

def to_categorical(x,n_col=None):
    """ One-hot encoding of nominal values """
    if not n_col:
        n_col = np.amax(x) + 1
    one_hot = np.zeros((x.shape[0],n_col))
    one_hot[np.arange(x.shape[0]),x] = 1
    return one_hot

def to_nominal(x):
    """ Conversion from one-hot encoding to nominal"""
    return np.argmax(x,axis=1)

def make_diagonal(x):
    m = np.zeros((len(x),len(x)))
    for i in range(len(m[0])):
        m[i,i] = x[i]
    return m