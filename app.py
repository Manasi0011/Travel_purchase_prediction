import pickle

from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))


@app.route("/")
def hello_world():
    return render_template("prediction_page.html")


@app.route('/xyz', methods=['POST', 'GET'])
def xyz():
    if request.method == 'POST':
        data2 = request.form['Age']
        data3 = request.form['Type of Contact']
        data4 = request.form['City Tier']
        data5 = request.form['Duration Of Pitch']
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

        list1 = np.array([[data2, data3, data4, data5,
                           data6, data7, data8, data9, data10,
                           data11, data12, data13, data14, data15,
                           data16, data17, data18, data19]], dtype='float64')

        value = model.predict(list1)
        return render_template("After.html", data=value)
    else:
        return render_template("After.html")


if __name__ == '__main__':
    app.run(debug=True)
