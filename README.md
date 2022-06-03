## Keystroke Dynamics Authentication by using Random Forest Classifier (ランダムフォレストを用いたキーストローク認証)

This project creates a random forest classifier by using the keystroke dataset created by Carnegie Mellon University.  
#### Keystroke Dynamics - Benchmark Data Set (https://www.cs.cmu.edu/~keystroke/)
<br/>

For evaluating the classifier, the average equal-error rate is computed so that we can compare this classifier with other anomanly detection algorithms conducted in the CMU research paper (http://www.cs.cmu.edu/~keystroke/KillourhyMaxion09.pdf).

The keystroke data set is contained in the data.csv file, and RandomForest.py trains and tests a random forest classfier by using data.csv for creating the classifier and ERR.py for the testing.
