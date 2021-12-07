"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os

# Data dependencies
import pandas as pd

# Vectorizer
news_vectorizer = open("resources/tfidf_vectorizer.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Tweet Classifer")
	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Prediction", "Information","Sentiment Analysis","About The App"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Information" page
	if selection == "Information":
		page = """	
				* Home
				* Read about the project
				* Use 'choose option' to change page
				"""
		st.sidebar.markdown(page)
		st.info(" “Changing minds, not the climate” ")
		# You can read a markdown file from supporting resources folder
		st.markdown("The impact of climate change in the environment has been an extremely debated issue throughout various platforms. Social Media has given individuals of different backgrounds a forum where they can share their thoughts and opinions. This presentation analyses the twitter climate change sentiments and how this can assist UNESCO Climate Change Initiative.")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with ML Models")
		

		options = ["Logistics Regression", "Random Forest", "XGB Classifier"]
		selection = st.selectbox("Choose Model", options)

        # Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")

		st.info(selection)
			# Now, let’s create a progress bar:
			# Add a placeholder
		""" computing... please wait""" 
# 			latest_iteration = st.empty()
		bar = st.progress(0) 


		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			prediction= ""

			if selection == "Logistics Regression":
				predictor = joblib.load(open(os.path.join("resources/lr_model.pkl"),"rb"))
				prediction = predictor.predict(vect_text)
			
			if selection == "Random Forest":
				predictor = joblib.load(open(os.path.join("resources/rf_model.pkl"),"rb"))
				prediction = predictor.predict(vect_text)

			#if selection == "XGB Classifier":
			#	predictor = joblib.load(open(os.path.join("resources/xgb_model.pkl"),"rb"))
			#	prediction = predictor.predict(vect_text)
	
			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Successful...")

			if prediction == 0:
				st.markdown("Nuetral: the tweet neither supports nor refutes the belief of man-made climate change")

			if prediction == 1:
				st.markdown("Positive: the tweet supports the belief of man-made climate change")

			if prediction == 2:
				st.markdown("News: the tweet links to factual news about climate change")
			
			if prediction == -1:
				st.markdown("Negative: the tweet does not believe in man-made climate change")

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
