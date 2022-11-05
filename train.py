import os
import random
import numpy as np
from sklearn.utils.fixes import sklearn
from dataset import get_data
from model import lstm_model, fast_text_svm

if __name__ == '__main__':
    SEED = 1988

    random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    np.random.seed(SEED)
    sklearn.random.seed(SEED)

    train, test, target = get_data()

    # lstm_model(train, test, target)

    fast_text_svm(train, test, target)
