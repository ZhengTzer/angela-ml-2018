# USAGE
# Start the server:
# 	python app.py

# import the necessary packages
from lifetimes.utils import summary_data_from_transaction_data
from lifetimes.utils import calibration_and_holdout_data
from lifetimes import GammaGammaFitter

import numpy
import flask
import json
import pandas
import datetime

baseURL = "http://ancient-river-10489.herokuapp.com"

# initialize our Flask application
app = flask.Flask(__name__)


@app.route("/")
def index():
    return "hello world"
@app.route("/predictSpending/<customerId>", methods=["GET"])
def predictSpending(customerId):
	# initialize the data dictionary that will be returned 
	data = {"success": False, "result": {"customerId": "", "y":0.0} }
	
	# ensure the customer ID was properly uploaded to our endpoint
	if customerId:		
		print("* get data")
		data = pandas.read_csv("sample_transactions.csv")
		#data = pandas.read_json(baseURL + "/api/transactions")
		#data = data.drop(columns="_id")
		
		print("* prepare data")
		# prepare and shaping the data
		# columns -
		#   customerId
		# 	frequency : number of repeat purchase transactions
		#	recency: time (in days) between first purchase and latest purchase 
		#	T: time (in days) between first purchase and end of the period under study
		#	monetary_value: average transactions amount
		today = pandas.to_datetime(datetime.date.today())
		summaryData = summary_data_from_transaction_data(data, 
						"customerId", "transactionDate", 
						monetary_value_col="transactionAmount", 
						observation_period_end=today)
		# filter the customer data that has no transaction	
		analysisData = summaryData[summaryData["frequency"]>0]
		
		# get the stat of the particular customer
		customer = analysisData.loc[customerId]
		
		# load model
		ggf_loaded = GammaGammaFitter()
		ggf_loaded.load_model('ggf.pkl')
		
		# estimate the average transaction amount
		predict = ggf_loaded.conditional_expected_average_profit(customer["frequency"], customer['monetary_value'])
		
		# add the input and predicted output to the return data
		data = {"success": True, "result": {"customerId": customerId, "y":predict} }

	# return the data dictionary as a JSON response
	return flask.jsonify(data)

		
# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print("* Loading ML model and Flask starting server...")
	print("* Please wait until server has fully started...")

	# get data
	# columns -
	# 	customerId: customer unique identifier
	#	transactionDate: date of transaction
	#	transactionAmount: amount spent
	
	print("* get data")
	data = pandas.read_csv("sample_transactions.csv")
	#data = pandas.read_json(baseURL + "/api/transactions")
	#data = data.drop(columns="_id")
	
	print("* prepare data")
	# prepare and shaping the data
	# columns -
	#   customerId
	# 	frequency : number of repeat purchase transactions
	#	recency: time (in days) between first purchase and latest purchase 
	#	T: time (in days) between first purchase and end of the period under study
	#	monetary_value: average transactions amount
	today = pandas.to_datetime(datetime.date.today())
	summaryData = summary_data_from_transaction_data(data, 
					"customerId", "transactionDate", 
					monetary_value_col="transactionAmount", 
					observation_period_end=today)
	# filter the customer data that has no transaction
	
	analysisData = summaryData[summaryData["frequency"]>0]
	
	print("* train model")
	# using lifetimes - Gamma-Gamma submodel
	# train the model using purchase frequency (correlated to monetary value) 
	ggf = GammaGammaFitter(penalizer_coef = 0)
	ggf.fit(analysisData["frequency"],analysisData["monetary_value"])
		
	# save model
	ggf.save_model('ggf.pkl')
	
	# run the app
	app.run()	
	