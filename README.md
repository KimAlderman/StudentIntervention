# StudentIntervention
Machine Learning Nanodegree at Udacity - Student Intervention

The goal of this project was to build a model to predict which students might need intervention before they fail to graduate.

Tools and techniques used in this project include:

* data exploration through summary statistics and data visualization
* pandas, numpy, sklearn and matplotlib libraries for Python
* converting categorical features to dummy variables using pandas get_dummies function
* splitting data into train & test sets, shuffling, and cross-validation
* evaluating pros & cons of different supervised machine learning models and choosing appropriate models for this application
* performance metrics for unbalanced classes, including precision, recall, and F1 score; used equal-weight F1 score for this project
* sklearn implementation of 3 models for comparison: logistic regression, decision tree, and support vector machines (SVM)
* use of gridsearch for SVM hyperparameter tuning and optimization

With additional experience, I would probably make a few changes in this project.

1) Because the data is unbalanced, I would want to preserve the ratio in both the train and test sets using stratify.
```Python
X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        X_all,y_all,test_size=0.25,random_state = 10, stratify = y_all)
```
This could be an alternative too:
```Python
from sklearn.cross_validation import StratifiedShuffleSplit
...
ssscv = StratifiedShuffleSplit( y_train, n_iter=10, test_size=0.1)
grid = GridSearchCV(clf, parameters, cv = ssscv , scoring=f1_scorer) 
grid.fit( X_train, y_train ) 
...
```

2) Simplify the code by using a for loop.
```Python
for clf in [clf_A,clf_B,clf_C]:
    for size in [100,200,300]:
        train_predict(clf, X_train[:size], y_train[:size], X_test, y_test)
        print ""
```

3) Use a differently weighted F1 score to evaluate model until I had the desired number of students to target for intervention. This could fluctuate depending on funding each year.
```Python
f1_scorer = make_scorer(fbeta_score, beta=1, pos_label= "yes")
```
The beta parameter determines the weight of precision in the combined score:
* beta = 1; equal weight to precision and recall, equivalent to F1 score
* beta < 1 lends more weight to precision;
* beta > 1 favors recall
Ref: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html




