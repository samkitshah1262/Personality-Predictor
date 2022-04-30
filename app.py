from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
from dep import Lemmatizer
from bs4 import BeautifulSoup
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)
# model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    # int_features = [int(x) for x in request.form.values()]
    # final_features = [np.array(int_features)]
    # prediction = model.predict(final_features)
    text=request.form.values()
    regex = re.compile('[%s]' % re.escape('|'))
    # text = BeautifulSoup(text, "lxml").text
    # text = re.sub(r'\|\|\|', r' ', text) 
    # text = re.sub(r'http\S+', r'<URL>', text)
    # text = regex.sub(" ", text)
    words = str(text).split()
    words = [i.lower() + " " for i in words]
    words = [i for i in words if not "http" in i]
    words = " ".join(words)
    words = words.translate(words.maketrans('', '', string.punctuation))
    # output = round(prediction[0], 2)
    vectorizer=TfidfVectorizer(max_features=5000,stop_words='english',tokenizer=Lemmatizer())
    # raw=vectorizer.fit(words).toarray()
    # output=model.predict(raw)
    output='success'
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