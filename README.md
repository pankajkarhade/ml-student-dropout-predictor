🎓 Student Dropout Prediction using Machine Learning

This project is a Machine Learning based prediction system designed to identify whether a student is at risk of dropping out from their academic program. Early prediction helps institutions take preventive actions and improve student retention.

The model is built using Logistic Regression and trained on student-related features such as demographic details, academic performance, internet access, scholarship status, and other important factors.

🧠 Machine Learning Workflow

The project follows a complete end-to-end ML pipeline:

1️⃣ Data Preprocessing

Handling missing values

Encoding categorical variables

Feature transformation

2️⃣ Feature Engineering

Label Encoding

Ordinal Encoding

Feature scaling using StandardScaler

3️⃣ Model Training

Logistic Regression model

Handling class imbalance using class weights

4️⃣ Model Evaluation

Accuracy

Precision

Recall

F1 Score

Confusion Matrix

5️⃣ Threshold Tuning

Precision-Recall analysis

Optimization of classification performance

🚀 Deployment

The trained model is deployed using Streamlit, allowing users to interact with the system through a simple web interface.

Users can enter student information such as:

Gender

Internet Access

Part-Time Job

Scholarship Status

Semester

Parental Education

Study Hours

Attendance

The system then predicts:

✅ Student likely to continue
⚠️ Student at risk of dropout

The app also displays the probability of dropout, helping educators make informed decisions.

🛠 Technologies Used

🐍 Python

📊 Pandas, NumPy

🤖 Scikit-learn

📈 Matplotlib, Seaborn

🌐 Streamlit

💾 Pickle for model serialization

📌 Key Features

✔ End-to-end Machine Learning pipeline
✔ Data preprocessing and feature encoding
✔ Model evaluation and performance analysis
✔ Streamlit interactive dashboard
✔ Real-time student dropout prediction

🎯 Goal of the Project

The main objective is to assist educational institutions in identifying students at risk of dropping out, enabling early intervention strategies to improve student success and retention rates.
