Fake News Detection Using NLP

An NLP-based Fake News Detection system using TF-IDF and Logistic Regression with a Streamlit web interface.

Features

Fake and Real News Classification

TF-IDF Text Vectorization

Logistic Regression Model

Streamlit Web Interface

FastAPI Prediction Endpoint

Prediction Score and Explanation

Technologies Used

Python

Pandas

Scikit-learn

NLTK

Streamlit

FastAPI

Installation

git clone https://github.com/vaibhavbatham13-code/fake-news-detection-using-nlp.git
cd fake-news-detection-using-nlp
pip install -r requirements.txt

Run the Application

streamlit run app.py

Model Training

Place the Kaggle Fake.csv and True.csv files inside the data folder, then run:

python train_model.py

Disclaimer

This model identifies statistical writing patterns. Its predictions are not proof that an article is true or false. Always verify news using trusted sources.

Author

Created by Vaibhav Batham.