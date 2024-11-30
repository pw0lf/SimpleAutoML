import numpy as np
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

from models import ModelCollection
from models import BasicModels


def _hp_optimization(model, hps, X, y, n_trials, n_folds, stratified,metric,random_seed):
    def objective(trial):
        print(f"Trial: {trial.number}")
        
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
            cur_acc = accuracy_score(y[test_index], y_pred)
            cur_auc = roc_auc_score(y[test_index], y_pred)
            auc.append(cur_auc)
            acc.append(cur_acc)
            
        if metric == "acc":
            return np.mean(acc)
        else:
            return np.mean(auc)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    return study.best_value ,study.best_params



class AutoMLClassification:
    def __init__(self, models: ModelCollection, metric: str="acc",n_trials: int=10, n_folds: int=3, stratified: bool=True,random_seed: int=42):
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
            print(f"Optimizing {name}")
            self.results_metric[name], self.results_hps[name] = _hp_optimization(model, hps, X, y, self.n_trials, self.n_folds, self.stratified,self.metric,self.random_seed)
        print(self.results_metric)
        print(self.results_hps)
        best_index = max(self.results_metric, key=self.results_metric.get)
        print(self.models[best_index])
        self.best_model = self.models[best_index][1](**self.results_hps[best_index])
        self.best_model.fit(X, y)
        
    def predict(self, X):
        return self.best_model.predict(X)
    
    def predict_proba(self, X):
        return self.best_model.predict_proba(X)
        
        
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    X, y = make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1)
    X += 2 * np.random.uniform(size=X.shape)
    
    aml = AutoMLClassification(BasicModels,metric="auc")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    aml.fit(X_train,y_train)
    y_pred = aml.predict(X_test)
    print(accuracy_score(y_test,y_pred))