import pickle

from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
pipeline = pickle.load(open("preprocessing_pipeline.pkl", "rb"))


@app.route("/")
def hello_world():
    return render_template("prediction_page.html")


@app.route('/predict', methods=['POST', 'GET'])
def xyz():
    if request.method == 'POST':
        data2 = request.form['Age']
        data3 = request.form['TypeofContact']
        data4 = request.form['CityTier']
        data5 = request.form['DurationOfPitch']
        data6 = request.form['Occupation']
        data7 = request.form['Gender']
        data8 = request.form['NumberOfPersonVisited']
        data9 = request.form['NumberOfFollowups']
        data10 = request.form['PreferredPropertyStar']
        data11 = request.form['ProductPitched']
        data12 = request.form['MaritalStatus']
        data13 = request.form['NumberOfTrips']
        data14 = request.form['Passport']
        data15 = request.form['PitchSatisfactionScore']
        data16 = request.form['OwnCar']
        data17 = request.form['NumberOfChildrenVisited']
        data18 = request.form['Designation']
        data19 = request.form['MonthlyIncome']

        # Create a dictionary with the user's input
        user_input = {'Age': data2,
                      'TypeofContact': data3,
                      'CityTier': data4,
                      'DurationOfPitch': data5,
                      'Occupation': data6,
                      'Gender': data7,
                      'NumberOfPersonVisited': data8,
                      'NumberOfFollowups': data9,
                      'PreferredPropertyStar': data10,
                      'ProductPitched': data11,
                      'MaritalStatus': data12,
                      'NumberOfTrips': data13,
                      'Passport': data14,
                      'PitchSatisfactionScore': data15,
                      'OwnCar': data16,
                      'NumberOfChildrenVisited': data17,
                      'Designation': data18,
                      'MonthlyIncome': data19,
                      }

        # Convert user input to a DataFrame for transformation
        input_df = pd.DataFrame([user_input])

        # Transform user input using the preprocessing pipeline
        transformed_input = pipeline.transform(input_df)

        transformed_input = pd.DataFrame(transformed_input.toarray())

        # Make the prediction using your model
        value = model.predict(transformed_input)
        if value[0] == 1:
            pred = "will"
        else:
            pred = "will not"
        # Taking the P_value as % probability for the purchase
        p_value = model.predict_proba(transformed_input)
        p_value_perc = (p_value[:, 1][0]) * 100
        p_value_class_1 = round(p_value_perc, 2)

        return render_template("result2.html", prediction=pred, probability=p_value_class_1)
    else:
        return render_template("result2.html")


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
