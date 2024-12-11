# SimpleAutoML

A framework for simple AutoML in the sense of keeping an overview of the steps the framework takes and having full control.

## Installation

1. Download the package by cloning this repo.
2. In your python environment run the following command but replace the path:
```
pip install path/to/SimpleAutoML
```

## How to use the package

```
from simpleaml.automl import AutoMLClassification

automl = AutoMLClassification()
automl.fit(X_train,y_train)

y_pred = automl.predict(X_test)
```

## AutoMLClassification API

TODO

## Configure models and hyperparameter-grid

TODO