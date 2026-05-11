import io
import os
import pickle
import numpy as np
from sklearn import metrics
from sklearn import utils
from sklearn import model_selection
from sklearn.model_selection import KFold
import scipy
import torch
#import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import make_scorer, get_scorer


class _CpuUnpickler(pickle.Unpickler):
    """Remap CUDA tensors to CPU for models serialised on a GPU node."""
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
        return super().find_class(module, name)


try:
    import sklearn.metrics._scorer as _scorer
except ImportError:
    # fallback placeholder to allow unpickling old models
    class _DummyScorer:
        class _PredictScorer:
            def __init__(self, *args, **kwargs):
                pass
    _scorer = _DummyScorer()

def save_model(model, filename):

    outpath = os.path.join("../Models/", filename)

    with open(outpath, "wb") as f:
        pickle.dump(model, f)

    print("Saved model to file: %s" % (outpath))


def load_model(path):
    try:
        # provide compatibility alias for older sklearn pickles that reference _PredictScorer
        if not hasattr(_scorer, "_PredictScorer"):
            class _PredictScorer:
                def __init__(self, *args, **kwargs):
                    pass
            _scorer._PredictScorer = _PredictScorer
    except Exception:
        pass

    with open(path, "rb") as f:
        try:
            model = pickle.load(f)
        except RuntimeError as exc:
            if "CUDA" not in str(exc):
                raise
            # Model was saved on a GPU node; remap tensors to CPU on reload.
            # TabPFN will move them back to GPU automatically at predict time
            # if CUDA becomes available in the process.
            f.seek(0)
            model = _CpuUnpickler(f).load()
    return model


def regression_results(model, y_true, y_pred):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    print("-" * 80)
    print("Model: %s" % (model))
    print("-" * 80)
    results = []
    for metric in [metrics.mean_squared_error, metrics.mean_squared_log_error, metrics.mean_absolute_error,
                   metrics.explained_variance_score, metrics.median_absolute_error, metrics.r2_score]:

        res = metric(y_true, y_pred)
        results.append(res)
        print("%s: %.3f" % (metric.__name__, res))
    res = scipy.stats.pearsonr(np.array(y_true),np.array(y_pred))[0]
    results.append(res)
    print("Pearson R: %.3f" %(res))

    print("=" * 80)
    return results


def calculate_regression_metrics(labels, predictions):
    # Mean Absolute Error
    mae_val = round(metrics.mean_absolute_error(labels, predictions), 3)

    # Root Mean Squared Error (compatible with all sklearn versions)
    try:
        rmse_val = round(metrics.mean_squared_error(labels, predictions, squared=False), 3)
    except TypeError:
        rmse_val = round(np.sqrt(metrics.mean_squared_error(labels, predictions)), 3)

    # R², Pearson, Spearman
    pearson_r = scipy.stats.pearsonr(np.array(labels).flatten(), np.array(predictions).flatten())[0]
    r2_val = round(np.power(pearson_r, 2), 3)
    pearson_val = round(pearson_r, 3)
    spearman_val = round(scipy.stats.spearmanr(np.array(labels).flatten(), np.array(predictions).flatten())[0], 3)

    return float(mae_val), float(rmse_val), float(r2_val), float(pearson_val), float(spearman_val)


def get_CV_results (model, X_train, Y_train, n_splits):
    kf = KFold(n_splits=n_splits)
    mae_list, rmse_list, r2_list, pr_list, sr_list = [],[],[],[],[]
    for train_index, test_index in kf.split(Y_train):
        y_pred = model.best_estimator_.predict(X_train.iloc[test_index,:])
        results=calculate_regression_metrics(Y_train[test_index],y_pred)
        mae_list.append(results[0])
        rmse_list.append(results[1])
        r2_list.append(results[2])
        pr_list.append(results[3])
        sr_list.append(results[4])
    print(mae_list,rmse_list,r2_list)
    mean_mae, sd_mae = round(np.mean(mae_list),3), round(np.std(mae_list),3)
    mean_rmse, sd_rmse = round(np.mean(rmse_list),3), round(np.std(rmse_list),3)
    mean_r2, sd_r2 = round(np.mean(r2_list),3), round(np.std(r2_list),3)
    mean_pr, sd_pr = round(np.mean(pr_list),3), round(np.std(pr_list),3)
    mean_sr, sd_sr = round(np.mean(sr_list),3), round(np.std(sr_list),3)
    return(mean_mae, sd_mae, mean_rmse, sd_rmse, mean_r2, sd_r2, mean_pr, sd_pr, mean_sr, sd_sr)


def supervised_learning_steps(method, scoring, data_type, task, model, params, X_train, y_train, n_iter, n_splits = 5):
    
    gs = grid_search_cv(model, params, X_train, y_train, scoring=scoring, n_iter = n_iter, n_splits = n_splits)

    y_pred = gs.predict(X_train)
    y_pred[y_pred < 0] = 0

    if task:
        results=calculate_classification_metrics(y_train, y_pred)
        print("Acc: %.3f, F1: %.3f, AUC: %.3f, AUPR: %.3f" % (results[0], results[1], results[2], results[3]))
    else:
        results=calculate_regression_metrics(y_train,y_pred)
        print("MAE: %.3f, MSE: %.3f, R2: %.3f, Pearson R: %.3f, Spearman R: %.3f" % (results[0], results[1], results[2], results[3], results[4]))
   
    print('Parameters')
    print('----------')
    for p,v in gs.best_estimator_.get_params().items():
        print(p, ":", v)
    print('-' * 80)

    if task:
        save_model(gs, "%s_models/%s_%s_classifier_gs.pk" % (method,method,data_type))
        save_model(gs.best_estimator_, "%s_models/%s_%s_classifier_best_estimator.pk" %(method,method,data_type))
    else:
        save_model(gs, "%s_models/%s_%s_regressor_gs.pk" % (method,method,data_type))
        save_model(gs.best_estimator_, "%s_models/%s_%s_regressor_best_estimator.pk" %(method,method,data_type))
    return(gs)


def grid_search_cv(model, parameters, X_train, y_train, n_splits=5, n_iter=1000, n_jobs=42, scoring="r2", stratified=False):
    """
        Tries all possible values of parameters and returns the best regressor/classifier.
        Cross Validation done is stratified.
        See scoring options at https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    """

    # Stratified n_splits Folds. Shuffle is not needed as X and Y were already shuffled before.
    if stratified:
        cv = model_selection.StratifiedKFold(n_splits=n_splits, shuffle=False, random_state=42)
    else:
        cv = n_splits

    rev_model = model_selection.RandomizedSearchCV(estimator=model, param_distributions=parameters, cv=cv, scoring=scoring, n_iter=n_iter, n_jobs=42, random_state=0, verbose=2)
    if (model=="xgb"):
        xgbtrain = xgb.DMatrix(X_train, Y_train)
        output = rev_model.fit(xgbtrain)
        rm(xgbtrain)
        gc()
        return output    
    else:
        return rev_model.fit(X_train, y_train)


