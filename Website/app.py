import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
import data_creation_v3 as d
from keras.models import load_model
import h5py
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, model_from_json
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import PorterStemmer
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize

order = ['bodyLength', 'bscr', 'dse', 'dsr', 'entropy', 'hasHttp', 'hasHttps',
       'has_ip', 'numDigits', 'numImages', 'numLinks', 'numParams',
       'numTitles', 'num_%20', 'num_@', 'sbr', 'scriptLength', 'specialChars',
       'sscr', 'urlIsLive', 'urlLength']

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer( min_df=2, max_df=0.7, stop_words=stopwords.words('english'))
# posts = vectorizer.fit_transform(df['Sentence'].values.astype('U')).toarray()

def convert_to_ascii(sentence):
    sentence_ascii=[]
    for i in sentence:
        if(ord(i)<8222):      # ” has ASCII of 8221
            
            if(ord(i)==8217): # ’  :  8217
                sentence_ascii.append(134)
            if(ord(i)==8221): # ”  :  8221
                sentence_ascii.append(129)
            if(ord(i)==8220): # “  :  8220
                sentence_ascii.append(130)
            if(ord(i)==8216): # ‘  :  8216
                sentence_ascii.append(131)
            if(ord(i)==8217): # ’  :  8217
                sentence_ascii.append(132)
            if(ord(i)==8211): # –  :  8211
                sentence_ascii.append(133)
            if (ord(i)<=128):
                    sentence_ascii.append(ord(i))
            else:
                    pass
    zer=np.zeros((10000))
    for i in range(len(sentence_ascii)):
        zer[i]=sentence_ascii[i]

    zer.shape=(100, 100)
    return zer

def clean_data(input_val):

    input_val=input_val.replace('\n', '')
    input_val=input_val.replace('%20', ' ')
    input_val=input_val.replace('=', ' = ')
    input_val=input_val.replace('((', ' (( ')
    input_val=input_val.replace('))', ' )) ')
    input_val=input_val.replace('(', ' ( ')
    input_val=input_val.replace(')', ' ) ')
    input_val=input_val.replace('1 ', 'numeric')
    input_val=input_val.replace(' 1', 'numeric')
    input_val=input_val.replace("'1 ", "'numeric ")
    input_val=input_val.replace(" 1'", " numeric'")
    input_val=input_val.replace('1,', 'numeric,')
    input_val=input_val.replace(" 2 ", " numeric ")
    input_val=input_val.replace(' 3 ', ' numeric ')
    input_val=input_val.replace(' 3--', ' numeric--')
    input_val=input_val.replace(" 4 ", ' numeric ')
    input_val=input_val.replace(" 5 ", ' numeric ')
    input_val=input_val.replace(' 6 ', ' numeric ')
    input_val=input_val.replace(" 7 ", ' numeric ')
    input_val=input_val.replace(" 8 ", ' numeric ')
    input_val=input_val.replace('1234', ' numeric ')
    input_val=input_val.replace("22", ' numeric ')
    input_val=input_val.replace(" 8 ", ' numeric ')
    input_val=input_val.replace(" 200 ", ' numeric ')
    input_val=input_val.replace("23 ", ' numeric ')
    input_val=input_val.replace('"1', '"numeric')
    input_val=input_val.replace('1"', '"numeric')
    input_val=input_val.replace("7659", 'numeric')
    input_val=input_val.replace(" 37 ", ' numeric ')
    input_val=input_val.replace(" 45 ", ' numeric ')

    return input_val



if __name__ == '__main__':
	st.title("Group 26: Ensemble-based XSS and SQLi Detection using CNN")
	
	st.text("")
	st.write("_______________")
	st.text(""" 
        """)
	st.sidebar.text("--- Developed By ---")
	st.sidebar.text(""" Group 26:
	Jai Vardhan 
	Kothapalli Sai Swetha 
	Pangoth Santhosh Kumar 


        """)

	user_input = st.text_input("Enter SQL Query:")

	# with open('vectorizer_cnn', 'wb') as fin:
	# 	pickle.dump(vectorizer, fin) 
	# load json and create model
	json_file = open('model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights("SQLmodel.h5")
	# print("Loaded model from disk")
	mymodel = loaded_model
	myvectorizer = pickle.load(open("vectorizer_cnn", 'rb'))

	input_val=clean_data(user_input)
	input_val=[input_val]
	input_val=myvectorizer.transform(input_val).toarray()
	# input_val.shape=(1,64,64,1)
	result=mymodel.predict(input_val)

	# a = d.UrlFeaturizer(user_input).run()
	# test = []
	# for i in order:
	#     test.append(a[i])

	# encoder = LabelEncoder()
	# encoder.classes_ = np.load('lblenc_v1.npy',allow_pickle=True)
	# scalerfile = 'scaler.sav'
	# scaler = pickle.load(open(scalerfile, 'rb'))
	# model = load_model("SQLmodel.h5")#, custom_objects={'f1_m':f1_m,"precision_m":precision_m, "recall_m":recall_m})
	# test = pd.DataFrame(test).replace(True,1).replace(False,0).to_numpy().reshape(1,-1)
	# predicted = np.argmax(model.predict(scaler.transform(test)),axis=1)

	ben = [1.        , 1.        , 1.        , 1.        , 0.56158211,
       0.        , 1.        , 0.        , 0.58866722, 1.        ,
       1.        , 0.16708727, 1.        , 0.16454762, 1.        ,
       1.        , 1.        , 0.95        , 1.        , 0.        ,
       0.70961575]

	submit = st.button('Predict')
	if (user_input==""):
		st.write("Enter Valid URL")
	else:
		if submit and user_input!="":
			if result>0.5:
				pred = 'Injectable SQL Query'
			# print("ALERT!!!! SQL injection Detected")
		elif result<=0.5:
				pred = "Normal SQL Query"
			# print("It is normal")
		st.header("Type of URL : "+pred)
		st.subheader("What is a "+pred+" URL?")
	

		if (pred=="Normal SQL Query"):
			st.text("These Queries are generally harmless and non-malicious.")
		elif(pred=="Spam"):
			st.write("Spam refers to a broad range of unwanted pop-ups, links, data and emails that we face in our daily interactions on the web. Spam’s namesake is, (now unpopular) luncheon meat that was often unwanted but ever present. Spam can be simply unwanted, but it can also be harmful, misleading and problematic for your website in a number of ways.")
			st.write("Read More: [https://www.goup.co.uk/guides/spam/](https://www.goup.co.uk/guides/spam/)")
		elif(pred=="Defacement"):
			st.write("Web defacement is an attack in which malicious parties penetrate a website and replace content on the site with their own messages. The messages can convey a political or religious message, profanity or other inappropriate content that would embarrass website owners, or a notice that the website has been hacked by a specific hacker group.")
			st.write("Read More: [https://www.imperva.com/learn/application-security/website-defacement-attack/](https://www.imperva.com/learn/application-security/website-defacement-attack/)")	
		elif(pred=="Malware"):
			st.write("The majority of website malware contains features which allow attackers to evade detection or gain and maintain unauthorized access to a compromised environment. Some common types of website malware include credit card stealers, injected spam content, malicious redirects, or even website defacements.")
			st.write("Read More: [https://sucuri.net/guides/website-malware/](https://sucuri.net/guides/website-malware/)")	
		else:
			st.write("A phishing website (sometimes called a 'spoofed' site) tries to steal your account password or other confidential information by tricking you into believing you're on a legitimate website. You could even land on a phishing site by mistyping a URL (web address).")
			st.write("Read More: [https://safety.yahoo.com/Security/PHISHING-SITE.html#:~:text=A%20phishing%20website%20(sometimes%20called,a%20URL%20(web%20address).](https://safety.yahoo.com/Security/PHISHING-SITE.html#:~:text=A%20phishing%20website%20(sometimes%20called,a%20URL%20(web%20address).)")	


		# st.write("")
		# st.header("Extracted Features vs Safe URL")
		# st.subheader("Given below are the features extracted from the URL and the values of these features are plotted along x-axis with the features on the y-axis.")
		# plt.figure(figsize=(12,12))
		# plt.plot(scaler.transform(test)[0],order,color='red', marker='>',linewidth=0.65,linestyle=":",alpha=0.5)
		# plt.plot(ben,order,marker='o',linewidth=0.65,linestyle="--",alpha=0.5)
		# plt.legend(["Extracted Features","Avg Safe URL"])
		# plt.title("Variation of features for different types of URLs")
		# plt.ylabel("Features")
		# plt.xlabel("Normalised Mean Values")
		# plt.plot()
		# st.pyplot()

	a = st.text_input("Enter XSS::")
	model = load_model("yes.h5")

	if (a == ""):
		st.write("Enter Valid URL:")
	s1 = st.button('Predict XSS:')

	if s1 and a != "":
		a = np.reshape(a, (-1, 1))
		a = pd.DataFrame(a)
		a.columns = ["a"]
		a = a["a"]
		arr = np.zeros((len(a),100,100))

		for i in range(len(a)):
			image=convert_to_ascii(a[i])
			x=np.asarray(image,dtype='float')
			import cv2
			image =  cv2.resize(x, dsize=(100,100), interpolation=cv2.INTER_CUBIC)
			image/=128
			arr[i]=image
		data = arr.reshape(arr.shape[0], 100, 100, 1)
		A = model.predict(data)
		A = np.around(A, decimals=0)
		if A == 1:
			st.write("Safe")
		if A == 0:
			st.write("Unsafe")
