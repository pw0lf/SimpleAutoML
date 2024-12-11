from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

class ModelCollection:
    def __init__(self, models: list, model_names:list, hyperparameters: list):
        self.models = models
        self.model_names = model_names
        self.hps = hyperparameters
        
        self.iter_index = 0

    def __getitem__(self, key):
        #TODO: special case key is integer and key is string
        if isinstance(key, int):
            return self.model_names[key], self.models[key], self.hps[key]
        elif isinstance(key, str):
            if key in self.model_names:
                return key, self.models[self.model_names.index(key)], self.hps[self.model_names.index(key)]
            else:
                raise KeyError(f"Model {key} not found in ModelCollection")
        else:
            raise KeyError(f"{key} not found in ModelCollection")

    def __len__(self):
        return len(self.models)

    def __iter__(self):
        self.iter_index = 0
        return self
    
    def __next__(self):
        if self.iter_index < len(self.models):
            result = self.model_names[self.iter_index], self.models[self.iter_index], self.hps[self.iter_index]
            self.iter_index += 1
            return result
        else:
            raise StopIteration

    def __str__(self):
        return f"ModelCollection({self.model_names})"

    def __repr__(self):
        return str(self)
    
logreg_hps = [("C","float",(0.0,1.0)),("penalty","fixed","l2"),("solver","fixed","lbfgs"),("fit_intercept","categorical",[True,False])]
rf_hps = [("n_estimators","int",(10,100)),("max_depth","int",(1,10)),("min_samples_split","int",(2,10)),("min_samples_leaf","int",(1,10))]
svc_hps = [("C","float",(0.0,1.0)),("kernel","categorical",["linear","rbf","poly","sigmoid"])]
BasicModels = ModelCollection([LogisticRegression,LogisticRegression,RandomForestClassifier,RandomForestClassifier,SVC,SVC], 
                              ["LogisticRegression_plain","LogisticRegression_tuned","RandomForest_plain","RandomForest_tuned","SVC_plain","SVC_tuned"], 
                              [[],logreg_hps,[],rf_hps,[],svc_hps])