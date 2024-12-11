import numpy as np
import optuna
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

from simpleaml.models import ModelCollection
from simpleaml.models import BasicModels


def _hp_optimization(model, hps, X, y, n_trials, n_folds, stratified,metric,random_seed):
    def objective(trial):
        hp_trial_dict = {}
        
        for hp in hps:
            if hp[1] == "categorical":
                hp_trial_dict[hp[0]] = trial.suggest_categorical(hp[0], hp[2])
            elif hp[1] == "int":
                hp_trial_dict[hp[0]] = trial.suggest_int(hp[0], hp[2][0], hp[2][1])
            elif hp[1] == "float":
                hp_trial_dict[hp[0]] = trial.suggest_float(hp[0], hp[2][0], hp[2][1])
            elif hp[1] == "fixed":
                hp_trial_dict[hp[0]] = hp[2]
        
        cur_model = model(**hp_trial_dict)
        
        if stratified:
            kf = StratifiedKFold(n_splits=n_folds, random_state=random_seed, shuffle=True)
        else:
            kf = KFold(n_splits=n_folds, random_state=random_seed, shuffle=True)
        acc = []
        auc = []
        for i, (train_index, test_index) in enumerate(kf.split(X, y)):
            cur_model.fit(X[train_index], y[train_index])
            y_pred = cur_model.predict(X[test_index])
            if metric == "acc":
                cur_acc = accuracy_score(y[test_index], y_pred)
                acc.append(cur_acc)
            else:
                cur_auc = roc_auc_score(y[test_index], y_pred)
                auc.append(cur_auc)
            
        if metric == "acc":
            return np.mean(acc)
        else:
            return np.mean(auc)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    return study.best_value ,study.best_params



class AutoMLClassification:
    def __init__(self, models: ModelCollection=BasicModels, metric: str="acc",n_trials: int=10, n_folds: int=3, stratified: bool=True,random_seed: int=42):
        self.models = models
        self.n_trials = n_trials
        self.n_folds = n_folds
        self.stratified = stratified
        self.metric = metric
        self.random_seed = random_seed
        self.results_metric = {}
        self.results_hps = {}
        self.best_model = None
    
    def fit(self, X, y):
        for name, model,hps in self.models:
            if hps == []:
                n_trials = 1
            else:
                n_trials = self.n_trials
            print(f"Optimizing {name}")
            self.results_metric[name], self.results_hps[name] = _hp_optimization(model, hps, X, y, n_trials, self.n_folds, self.stratified,self.metric,self.random_seed)
        best_index = max(self.results_metric, key=self.results_metric.get)
        self.best_model = self.models[best_index][1](**self.results_hps[best_index])
        self.best_model.fit(X, y)
        
    def predict(self, X):
        return self.best_model.predict(X)
    
    def predict_proba(self, X):
        return self.best_model.predict_proba(X)
    
    def save_tuning(self, path, filename):
        d = {}
        for model_name,_,_ in self.models:
            d[model_name] = {"metric":self.results_metric[model_name], "hps":self.results_hps[model_name]}
        with open(f"{path}/{filename}.json", "w") as file:
            json.dump(d,file,indent=4)
        
        
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)
    
    aml = AutoMLClassification(BasicModels,metric="acc",n_trials=50)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    aml.fit(X_train,y_train)
    y_pred = aml.predict(X_test)
    print(accuracy_score(y_test,y_pred))