# Importing essential libraries
from flask import Flask, render_template, request

import numpy as np
import pickle

# Load the Random Forest CLassifier model
lr_regressor = pickle.load(open('lr-model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
 return render_template('index-1.html')

@app.route('/predict', methods=['POST'])
def predict():
    temp_array = list()
    
    if request.method == 'POST':
        
        batting_team = request.form['batting-team']
        if batting_team == 'Gyanganga':
            temp_array = temp_array + [1,0,0,0,0,0,0,0]
        elif batting_team == 'NH Goel':
            temp_array = temp_array + [0,1,0,0,0,0,0,0]
        elif batting_team == 'Holy Cross':
            temp_array = temp_array + [0,0,1,0,0,0,0,0]
        elif batting_team == 'KPS':
            temp_array = temp_array + [0,0,0,1,0,0,0,0]
        elif batting_team == 'KV':
            temp_array = temp_array + [0,0,0,0,1,0,0,0]
        elif batting_team == 'Holy Heart':
            temp_array = temp_array + [0,0,0,0,0,1,0,0]
        elif batting_team == 'DPS':
            temp_array = temp_array + [0,0,0,0,0,0,1,0]
        elif batting_team == 'Aadarsh':
            temp_array = temp_array + [0,0,0,0,0,0,0,1]
            
            
        bowling_team = request.form['bowling-team']
        if bowling_team == 'Gyanganga':
            temp_array = temp_array + [1,0,0,0,0,0,0,0]
        elif bowling_team == 'NH Goel':
            temp_array = temp_array + [0,1,0,0,0,0,0,0]
        elif bowling_team == 'Holy Cross':
            temp_array = temp_array + [0,0,1,0,0,0,0,0]
        elif bowling_team == 'KPS':
            temp_array = temp_array + [0,0,0,1,0,0,0,0]
        elif bowling_team == 'KV':
            temp_array = temp_array + [0,0,0,0,1,0,0,0]
        elif bowling_team == 'Holy Heart':
            temp_array = temp_array + [0,0,0,0,0,1,0,0]
        elif bowling_team == 'DPS':
            temp_array = temp_array + [0,0,0,0,0,0,1,0]
        elif bowling_team == 'Aadarsh':
            temp_array = temp_array + [0,0,0,0,0,0,0,1]
            
            
        overs = float(request.form['overs'])
        runs = int(request.form['runs'])
        wickets = int(request.form['wickets'])
        runs_in_prev_5 = int(request.form['runs_in_prev_5'])
        wickets_in_prev_5 = int(request.form['wickets_in_prev_5'])
        
        temp_array = temp_array + [overs, runs, wickets, runs_in_prev_5, wickets_in_prev_5]
        
        data = np.array([temp_array]).reshape(1, -1)
        my_prediction = int(lr_regressor.predict(data)[0])
              
        return render_template('result-1.html', lower_limit = my_prediction-10, upper_limit = my_prediction+5)



if __name__ == '__main__':
	app.run(debug=True)

