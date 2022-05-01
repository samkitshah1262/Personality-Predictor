from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
from dep import Lemmatizer
from bs4 import BeautifulSoup
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    text=request.form.values()
    regex = re.compile('[%s]' % re.escape('|'))
    words = str(text).split()
    words = [i.lower() + " " for i in words]
    words = [i for i in words if not "http" in i]
    words = " ".join(words)
    words = words.translate(words.maketrans('', '', string.punctuation))
    vectorizer=TfidfVectorizer(max_features=5000,stop_words='english',tokenizer=Lemmatizer())
    raw=vectorizer.fit(words).toarray()
    output=model.predict(raw)
    return render_template('index.html', prediction_text='Your Personality Type is : {}'.format(output))

# @app.route('/predict_api',methods=['POST'])
# def predict_api():
#     '''
#     For direct API calls trought request
#     '''
#     data = request.get_json(force=True)
#     prediction = model.predict([np.array(list(data.values()))])

#     output = prediction[0]
#     return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)