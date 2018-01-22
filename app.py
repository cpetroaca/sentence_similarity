from flask import Flask
from flask import request
from flask import jsonify
from flask import g
from test import Tester
app = Flask(__name__)
neural_net = Tester()

@app.route("/similarity")
def compute_similarity():
    text1 = request.args.get('text1');
    text2 = request.args.get('text2');
    similarity = neural_net.compute_similarity(text1, text2)
    
    return jsonify(similarity.item(0))

if __name__ == '__main__':
    app.run(debug=False)