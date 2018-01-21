from flask import Flask
from flask import request
from flask import jsonify
from flask import g
from test import Tester
app = Flask(__name__)

def get_neural_net():
    if not hasattr(g, 'neural_net'):
        g.neural_net = Tester()
    
    return g.neural_net

@app.route("/similarity")
def calculate_similarity():
    text1 = request.args.get('text1');
    text2 = request.args.get('text2');
    tester = get_neural_net()
    similarity = tester.compute_similarity(text1, text2)
    
    return jsonify(similarity.item(0))

if __name__ == '__main__':
    app.run(debug=True)