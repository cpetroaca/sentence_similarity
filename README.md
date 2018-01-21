# sentence_similarity
Performs sentence similarity

First download all necessary data. See data/README for more details.

Then run train.py. This will create the tensorflow model model/ folder.

Then run app.py. This will start a rest api service at http://localhost:5000/similarity?text1='Text1'&text2='Text2'. text1 and text2 are te 2 texts which will be compared. The Api returns a float number between 0 and 1, o being totally different and 1 being identical.
