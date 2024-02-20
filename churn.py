import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
data=pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn (1).csv")
print(data.head(5))
data['class'] = data['Churn'].apply(lambda x : 1 if x == "Yes" else 0)

X = data[['tenure','MonthlyCharges']].copy()
y = data['class'].copy()
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X,y , test_size = 0.2, random_state = 0)


clf = LogisticRegression(fit_intercept=True, max_iter=10000)
clf.fit(X_train, y_train)

# train_preds = clf.predict_proba(X_train)
# test_preds = clf.predict_proba(X_test)
# train_class_preds = clf.predict(X_train)
# test_class_preds = clf.predict(X_test)




from flask import Flask,render_template,request
churn= Flask(__name__)

@churn.route('/')
def prju():
    return render_template('churn.html')

@churn.route('/output',methods=['GET','POST'])
def clty():
    tenure=request.form['tenure']
    MonthlyCharges=request.form['MonthlyCharges']
    arr=np.array([tenure,MonthlyCharges])
    arr=arr.astype(np.float64)
    pred=clf.predict([arr])

    if pred==1:
        result= "churned"
    else:
        result= "not churned"



    return render_template('output.html',pred=result)


if __name__ == '__main__':
    churn.run(debug=True)

