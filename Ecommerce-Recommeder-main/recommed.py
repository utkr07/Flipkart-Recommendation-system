import pickle as pkl
import json
from flask import Flask, request, jsonify
import pandas as pd
import requests
import random;
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask
from flask_cors import CORS, cross_origin
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

   
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    data=data['products']
    # print(data)
    def recommend(data,top_n=20):
        avg_feature = similarity[data].mean(axis=0)
        similarities = cosine_similarity(avg_feature.reshape(1, -1), similarity)
        similar_product_indices = similarities.argsort()[0][::-1][:top_n]
        return similar_product_indices.tolist()
      
    questions_dict=pkl.load(open("questions.pkl","rb"))
    questions_dict=pd.DataFrame(questions_dict)
    similarity=pkl.load(open("similarity.pkl","rb"))  
    prediction = recommend(data)
    return jsonify({"indices": prediction},)

if __name__ == "__main__":
    app.run()
