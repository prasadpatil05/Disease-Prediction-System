import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from itertools import combinations
import requests
from bs4 import BeautifulSoup
from nltk.corpus import wordnet
import warnings




st.image("background.jpg")

# Suppress warnings
warnings.simplefilter("ignore")

# Download NLTK resources
import nltk
#nltk.download('all')

# Load dataset
df_comb = pd.read_csv("Dataset/dis_sym_dataset_comb.csv")
df_norm = pd.read_csv("Dataset/dis_sym_dataset_norm.csv")

X = df_comb.iloc[:, 1:]
Y = df_comb.iloc[:, 0:1]

# Using Logistic Regression (LR) Classifier
lr = LogisticRegression()
lr = lr.fit(X, Y)
X = df_norm.iloc[:, 1:]
Y = df_norm.iloc[:, 0:1]
dataset_symptoms = list(X.columns)

# Function to find synonyms of a symptom
def synonyms(term):
    synonyms = []
    response = requests.get('https://www.thesaurus.com/browse/{}'.format(term))
    soup = BeautifulSoup(response.content,  "html.parser")
    try:
        container = soup.find('section', {'class': 'MainContentContainer'}) 
        row = container.find('div',{'class':'css-191l5o0-ClassicContentCard'})
        row = row.find_all('li')
        for x in row:
            synonyms.append(x.get_text())
    except:
        pass
    for syn in wordnet.synsets(term):
        synonyms += syn.lemma_names()
    return set(synonyms)

# Homepage layout
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from itertools import combinations
import requests
from bs4 import BeautifulSoup
from nltk.corpus import wordnet
import warnings

# Suppress warnings
warnings.simplefilter("ignore")

# Load dataset
df_comb = pd.read_csv("Dataset/dis_sym_dataset_comb.csv")
df_norm = pd.read_csv("Dataset/dis_sym_dataset_norm.csv")

X = df_comb.iloc[:, 1:]
Y = df_comb.iloc[:, 0:1]

# Using Logistic Regression (LR) Classifier
lr = LogisticRegression()
lr = lr.fit(X, Y)
X = df_norm.iloc[:, 1:]
Y = df_norm.iloc[:, 0:1]
dataset_symptoms = list(X.columns)

# Function to find synonyms of a symptom
def synonyms(term):
    synonyms = []
    response = requests.get('https://www.thesaurus.com/browse/{}'.format(term))
    soup = BeautifulSoup(response.content,  "html.parser")
    try:
        container = soup.find('section', {'class': 'MainContentContainer'}) 
        row = container.find('div',{'class':'css-191l5o0-ClassicContentCard'})
        row = row.find_all('li')
        for x in row:
            synonyms.append(x.get_text())
    except:
        pass
    for syn in wordnet.synsets(term):
        synonyms += syn.lemma_names()
    return set(synonyms)






# Homepage layout
def index():
    st.title("Disease Prediction")
    st.write("Enter symptoms separated by commas:")

    user_input = st.text_input("Symptoms")

    if st.button("Predict"):
        # Preprocessing symptoms
        processed_user_symptoms = []
        for sym in user_input.lower().split(','):
            sym = sym.strip()
            sym = sym.replace('-', ' ')
            sym = sym.replace("'", '')
            sym = ' '.join([WordNetLemmatizer().lemmatize(word) for word in RegexpTokenizer(r'\w+').tokenize(sym)])
            processed_user_symptoms.append(sym)

        # Finding synonyms for each symptom
        user_symptoms = []
        for user_sym in processed_user_symptoms:
            user_sym = user_sym.split()
            str_sym = set()
            for comb in range(1, len(user_sym)+1):
                for subset in combinations(user_sym, comb):
                    subset = ' '.join(subset)
                    subset = synonyms(subset)
                    str_sym.update(subset)
            str_sym.add(' '.join(user_sym))
            user_symptoms.append(' '.join(str_sym).replace('_', ' '))

        # Finding related symptoms in dataset
        found_symptoms = set()
        for idx, data_sym in enumerate(dataset_symptoms):
            data_sym_split = data_sym.split()
            for user_sym in user_symptoms:
                count = 0
                for symp in data_sym_split:
                    if symp in user_sym.split():
                        count += 1
                if count / len(data_sym_split) > 0.5:
                    found_symptoms.add(data_sym)

        # Final symptoms  
        sample_x = [0 for _ in range(len(dataset_symptoms))]
        for val in found_symptoms:
            if val in dataset_symptoms:
                sample_x[dataset_symptoms.index(val)] = 1

        # Predict disease
        prediction = lr.predict_proba([sample_x])

        # Showing the list of top k diseases with their prediction probabilities in a table format
        k = 10
        diseases = list(set(Y['label_dis']))
        diseases.sort()
        topk = prediction[0].argsort()[-k:][::-1]

        topk_dict = {}
        for idx, t in enumerate(topk):
            prob = prediction[0][t] * 100
            topk_dict[diseases[t]] = round(prob, 2)

        # Display top predicted diseases in a table format
        st.write("Top Predicted Diseases:")
        df_topk = pd.DataFrame(list(topk_dict.items()), columns=['Disease', 'Probability'])
        st.table(df_topk)



if __name__ == '__main__':
    index()

