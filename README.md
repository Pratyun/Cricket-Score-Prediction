🏏 Cricket Score Prediction using Machine Learning
This project focuses on predicting the final score of an IPL (Indian Premier League) cricket innings using Machine Learning models like Linear Regression and Random Forest Regressor.

We trained the models on real match data to forecast the final score based on live in-match parameters such as runs, wickets, overs, and batsmen's stats.

<img width="1913" height="990" alt="image" src="https://github.com/user-attachments/assets/3a3baf11-61c5-4aba-a2d1-7dba6f3f8d43" />
<img width="1919" height="943" alt="image" src="https://github.com/user-attachments/assets/41413981-36c9-4b95-84ca-c4f1b3ebd8a3" />


🚀 Project Features
Predicts final innings score based on:

Current runs

Wickets fallen

Overs completed

Striker and non-striker performance

Trained with:

Linear Regression

Random Forest Regression

Feature importance visualization to analyze the model focus

Accuracy checked using:

R² score

Custom accuracy threshold (±10 runs)

Beautiful matplotlib plots to visualize predictions vs actual scores

📊 Technologies & Libraries
Python 🐍

pandas, NumPy

scikit-learn (LinearRegression, RandomForestRegressor)

matplotlib

Flask (for deployment – optional)

IPL match dataset (CSV format)

📁 Dataset Info
Used dataset columns:

runs

wickets

overs

striker

non-striker

total (label: actual final score)

⚙️ How it Works
Data Preprocessing

Selected key input features

Applied train-test split

Scaled data using StandardScaler

Model Training

Trained on 75% of data

Tested on remaining 25%

Evaluation

Calculated accuracy

Plotted Actual vs Predicted results

Prediction

You can input live match values and predict the final score.

📈 Output Example
makefile
Copy
Edit
Input: [140 runs, 5 wickets, 10 overs, 50 striker, 50 non-striker]
Output: Predicted Final Score: 176 ± 3 runs
📍 Future Improvements
Integrate more features like venue, bowling team, current bowler stats

Deploy using Flask or Streamlit as a web app

Try advanced models like XGBoost, LSTM (for time-series)

🤝 Contributing
Feel free to fork this repo, create new features or enhance the model! Contributions are welcome.

