import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split

class MNIST(object):
    def __init__(self):
        self.x, self.y = self.download_data()
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
                        self.x, self.y, test_size=0.25, random_state=0)
        self.split_data()

        self.data_n = 0 # size of training data set
        for digit in self.digits:
            self.data_n += len(getattr(self, digit + '_train'))
        
    def split_data(self):
        self.digits = ['zeros', 'ones', 'twos', 'threes', 'fours', 'fives', 'six', 'sevens', 'eights', 'nines']
        for i in range(10):
            setattr(self, self.digits[i], [self.x[j] for j in range(len(self.y)) if self.y[j] == i])
            x_train_, x_test_, _, _ = train_test_split(
                        getattr(self, self.digits[i]), getattr(self, self.digits[i]),
                                test_size=0.25, random_state=0)
            setattr(self, self.digits[i] + '_train', x_train_)
            setattr(self, self.digits[i] + '_test', x_test_)
    
    def download_data(self):
        mnist = fetch_mldata('MNIST original')
        return mnist['data'], mnist['target']
    
    def chose_same_number(self, phase, one_shot=False):
        if not one_shot:
            digit = np.random.choice(self.digits)
        else:
            digit = np.random.choice(self.digits[:8])
        digit_indexes =  np.random.choice(len(getattr(self, digit + '_' + phase)), size=2, replace=False)
        return tuple(getattr(self, digit + '_' + phase)[i] for i in digit_indexes)
    
    def chose_different_numbers(self, phase, one_shot=False):
        if not one_shot:
            digits = np.random.choice(self.digits, size=2, replace=False)
        else:
            digits = np.random.choice(self.digits[:8], size=2, replace=False)
        digit_indexes =  [np.random.choice(len(getattr(self, i + '_' + phase))) for i in digits]
        return tuple(getattr(self, digit + '_' + phase)[index] for digit, index in zip(digits, digit_indexes))
    
    def get_next_batch(self, batch, phase='train', one_shot=False):
        """
        Args:
            batch: an `integer` representing the size of the batch.
            phase: a `string` in `['train', 'test']`. Indicates which
                data to retrieve.
            one_shot: a `boolean`. If True, data of eights and nines
                are dismissed during training and testing.

        Returns:
            x1_: a `numpy array` of shape `(batch, 28, 28)` containing
                images for the first network.
            x2_: a `numpy array` similar to x1_ containing images
                for the second network.
            y_: a `numpy array` of shape `(batch)` containing the labels.
        """
        x1_ = []
        x2_ = []
        y_ = []
        for _ in range(batch):
            if np.random.uniform() <= .5: # we chose two different numbers
                x1_tmp, x2_tmp = self.chose_different_numbers(phase, one_shot=one_shot)
                x1_.append(x1_tmp)
                x2_.append(x2_tmp)
                y_.append(0)
            else: # we chose two similar numbers
                x1_tmp, x2_tmp = self.chose_same_number(phase, one_shot=one_shot)
                x1_.append(x1_tmp)
                x2_.append(x2_tmp)
                y_.append(1)

        x1_ = np.asarray([x_.reshape((28, 28)) for x_ in x1_])
        x2_ = np.asarray([x_.reshape((28, 28)) for x_ in x2_])
        y_ = np.asarray(y_)
        return x1_, x2_, y_
