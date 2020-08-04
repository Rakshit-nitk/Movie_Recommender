#REF --> https://towardsdatascience.com/how-to-build-from-scratch-a-content-based-movie-recommender-with-natural-language-processing-25ad400eb243

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv('https://query.data.world/s/uikepcpffyo2nhig52xxeevdialfl7')

df = df[['Title','Genre','Director','Actors','Plot']]
df.head()

def func(str):
    l = str.split(",")
    length = len(l)
    for i in range(length):
        l[i] = l[i].replace(' ','')
    return " ".join(l)

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

#df['bag_of_words']=[251]
reviews = [None] * 250
for i in range(0, 250):
    reviews[i] = func(df['Genre'][i]) + " " + func(df['Director'][i]) \
    + " " + func(df['Actors'][i]) + " "
    
    string = df['Plot'][i]
    string = string.split()
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english')) 
    string = [ps.stem(word) for word in string if not word in stop_words]
    string = ' '.join(string)
    
    reviews[i] = reviews[i] + string
    reviews[i] = re.sub('[^a-zA-Z]', ' ', reviews[i])
    reviews[i] = reviews[i].lower()
    
data = df[['Title']]
data['bag_of_words'] = reviews
    
from sklearn.feature_extraction.text import CountVectorizer
count = CountVectorizer()
count_matrix = count.fit_transform(data['bag_of_words'])

# generating the cosine similarity matrix
cosine_sim = cosine_similarity(count_matrix, count_matrix)

indices = pd.Series(data.Title)
indices[0]

def recommendations(Title, cosine_sim = cosine_sim):
    
    # initializing the empty list of recommended movies
    recommended_movies = []
    
    # gettin the index of the movie that matches the title
    idx = indices[indices == Title].index[0]

    # creating a Series with the similarity scores in descending order
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)

    # getting the indexes of the 10 most similar movies
    top_10_indexes = list(score_series.iloc[1:11].index)
    
    # populating the list with the titles of the best 10 matching movies
    for i in top_10_indexes:
        recommended_movies.append(list(data.Title)[i])
        
    return recommended_movies


recommended_movies = recommendations('Fargo')

