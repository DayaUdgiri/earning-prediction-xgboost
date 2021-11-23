
# importing the necessary dependencies
from flask import Flask, render_template, request,jsonify
from flask_cors import CORS,cross_origin
import pickle
import pandas as pd
#from train import scaler

app = Flask(__name__) # initializing a flask app

@app.route('/',methods=['GET'])  # route to display the home page
@cross_origin()
def homePage():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
@cross_origin()
def index():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            age = float(request.form['age'])
            workclass = str(request.form['workclass'])
            fnlwgt = float(request.form['fnlwgt'])
            education = str(request.form['education'])
            education_num= float(request.form['education_num'])
            marital_status = str(request.form['marital_status'])
            occupation = str(request.form['occupation'])
            relationship = str(request.form['relationship'])
            race = str(request.form['race'])
            sex = str(request.form['sex'])
            capital_gain = float(request.form['capital_gain'])
            capital_loss = float(request.form['capital_loss'])
            hours_per_week = float(request.form['hours_per_week'])
            native_country = str(request.form['native_country'])

            user_input = pd.DataFrame([[age, workclass, fnlwgt, education, education_num, marital_status, occupation,
                                        relationship, race, sex, capital_gain, capital_loss, hours_per_week, native_country]],
                                      columns=['age', 'workclass', 'fnlwgt', 'education', 'education_num',
                                               'marital_status', 'occupation', 'relationship', 'race', 'sex',
                                               'capital_gain', 'capital_loss', 'hours_per_week', 'native_country'])

            #converting the type as "Category" as required by XGboost enable_categorical=True feature
            user_input["workclass"] = user_input["workclass"].astype("category")
            user_input["education"] = user_input["education"].astype("category")
            user_input["marital_status"] = user_input["marital_status"].astype("category")
            user_input["occupation"] = user_input["occupation"].astype("category")
            user_input["relationship"] = user_input["relationship"].astype("category")
            user_input["race"] = user_input["race"].astype("category")
            user_input["sex"] = user_input["sex"].astype("category")
            user_input["native_country"] = user_input["native_country"].astype("category")

            # loading the model file from the storage
            filename = 'XGB_clfModel_earning.pickle'
            loaded_model = pickle.load(open(filename, 'rb'))

            # predictions using the loaded model file
            prediction=loaded_model.predict(user_input)

            if prediction== 0:
                earning_pred='<=50K'
            elif prediction==1:
                earning_pred='>50K'
            print('Prediction is', prediction ," i.e. person's earning is " ,earning_pred)
            # showing the prediction results in a UI
            return render_template('results.html',prediction=earning_pred)
        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'
    # return render_template('results.html')
    else:
        return render_template('index.html')



if __name__ == "__main__":
    #app.run(host='127.0.0.1', port=8001, debug=True)
	app.run(debug=True) # running the app