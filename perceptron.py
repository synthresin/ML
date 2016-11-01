import numpy as np


class Perceptron(object):
    """Perceptron classifier.a

    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
        n_iter : int
        Passes over the training dataset.

    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting.

    errors_ : list
        Number of misclassifications in every epoch.
    """
    def __init__(self, eta=0.01, n_iter=10):
      self.eta = eta
      self.n_iter = n_iter

    def fit(self, X, y):
        """Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features
            n개의 피쳐를 가지는 샘플(벡터)이 n개 있다.
        y : array-like, shape = [n_samples]
            Target values. (아마도 true class labels of sample in X)


        Returns
        -------
        self : obejct
        """

        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []
        # w_ is basically a weight vector.
        # length of a weight vector is 1 higher than number of features.
        # X.shape[1] is column, so It's number of featuers.
        # initial weight vector has elements which is zero.

        for _ in range(self.n_iter):
          ## 지정된 반복회차 만큼...
          ## 샘플들을 반복 러닝 시키겠지.
          errors = 0
          for xi, target in zip(X, y):
            ## X는 기본적으로 sample 갯수만큼의 어레이이니깐...
            ## y는 각 샘플의 true class 니까, 길이가 샘플갯수지.
            ## 두개가 렝스가 딱 맞어
            ## xi 는 각 샘플의 피쳐를 가지고 있는 어레이이고,
            ## target 은 true class label 이지
            ## 여기서 해야하는 것은
            update = self.eta * (target - self.predict(xi))
            #이건 배운대로의 거시기고
            self.w_[1:] += update * xi
            # weight 벡터의 2번째 부터 끝 element에 (그럼 피쳐 갯수 만큼이다)
            # 해당 샘플의 각 피쳐 값 * update 보정치를 더해줌
            # [0,0,0,0,0,0]
            #+  [3,4,5,6,7] 이런 느낌?
            self.w_[0] = update
            # 0번째 피쳐는 1 이니까 update 에 뭐 곱할게 없어요.
            errors += int(update != 0.0)
            # 한 회차(모든 샘플 싹다 돌림)에 에러 몇개인지 계산
            self.errors_.append(errors)
            # 한회차에 에러 몇개 나왔나 쭉 기록.
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)












