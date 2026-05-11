import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel, polynomial_kernel
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import gc
from scipy.sparse.linalg import cg
from scipy.linalg import cho_factor, cho_solve

def smart_solve(A, b, spd=False, gpu=False):
    if gpu:
        import cupy as cp
        return cp.asnumpy(cp.linalg.solve(cp.asarray(A), cp.asarray(b)))

    if spd:
        c, lower = cho_factor(A, check_finite=False)
        return cho_solve((c, lower), b, check_finite=False)

    if spd==False and gpu==False:
        x, info = cg(A, b, rtol=1e-5)
        if info != 0:
            raise RuntimeError("CG did not converge")
        return x


def _get_kernel(X, Y=None, kernel="rbf", gamma=None, degree=3, coef0=1.0):
    if kernel == "linear":
        return linear_kernel(X, Y)
    elif kernel == "rbf":
        return rbf_kernel(X, Y, gamma=gamma)
    elif kernel == "poly":
        return polynomial_kernel(X, Y, degree=degree, gamma=gamma, coef0=coef0)
    elif callable(kernel):
        return kernel(X, Y)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")


class _BaseLSSVM(BaseEstimator):
    def __init__(self, C=1.0, kernel="rbf", gamma=None,
                 degree=3, coef0=1.0):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0

    def _fit(self, X, y):
        X, y = check_X_y(X, y, dtype=np.float64)
        self.X_train_ = X
        self.y_train_ = y

        n = X.shape[0]

        K = _get_kernel(
            X, X,
            kernel=self.kernel,
            gamma=self.gamma,
            degree=self.degree,
            coef0=self.coef0
        )

        # Regularized kernel
        Omega = K + np.eye(n) / self.C

        # Build linear system
        A = np.zeros((n + 1, n + 1))
        A[0, 1:] = 1
        A[1:, 0] = 1
        A[1:, 1:] = Omega

        B = np.zeros(n + 1)
        B[1:] = y

        #Fast solver for sparse matrices
        solution = smart_solve(A, B, spd=False, gpu=False)

        self.b_ = solution[0]
        self.alpha_ = solution[1:]

        del Omega, A, B
        gc.collect()

        return self

    def decision_function(self, X):
        check_is_fitted(self, ["alpha_", "b_", "X_train_"])
        X = check_array(X, dtype=np.float64)

        K = _get_kernel(
            X, self.X_train_,
            kernel=self.kernel,
            gamma=self.gamma,
            degree=self.degree,
            coef0=self.coef0
        )

        return K @ self.alpha_ + self.b_

class LSSVMClassifier(_BaseLSSVM, ClassifierMixin):
    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)

        if len(self.classes_) != 2:
            raise ValueError("LSSVMClassifier supports binary classification only")

        y_bin = np.where(y == self.classes_[0], -1.0, 1.0)
        return self._fit(X, y_bin)

    def predict(self, X):
        scores = self.decision_function(X)
        return np.where(scores >= 0, self.classes_[1], self.classes_[0])

class LSSVMRegressor(_BaseLSSVM, RegressorMixin):
    def fit(self, X, y):
        return self._fit(X, y)

    def predict(self, X):
        return self.decision_function(X)
