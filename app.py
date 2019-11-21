# Dependencies
from flask import Flask, request, jsonify
from sklearn.externals import joblib
from keras import backend as K
import traceback
import pandas as pd
import numpy as np

# Your API definition
app = Flask(__name__)

@app.route('/', methods=['POST'])
def predict():

	# tf.keras.backend.clear_session()
	K.clear_session()
	lr = joblib.load("model.pkl") # Load "model.pkl"

	model_columns = joblib.load("model_columns.pkl") # Load "model_columns.pkl"

	if lr:
		try:
			#

			json_ = request.json
			# print(json_)
			query = pd.DataFrame(json_, index=[0])
			query = query.reindex(columns=model_columns, fill_value=0)

			print(query)
			prediction = lr.predict(query).round().astype(int)

			print(prediction)


			return jsonify({'prediction': str(prediction[0][0])})
		except:
			return jsonify({'trace': traceback.format_exc()})
			#K.clear_session()	
	else:
		print ('Train the model first')
		return ('No model here to use')


if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345

    app.run(port=port, debug=True)

