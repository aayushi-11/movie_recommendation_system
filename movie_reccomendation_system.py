#!/usr/bin/env python
# coding: utf-8

# # Movie Recommendation System

# In[3]:


import pandas as pd
import numpy as np


# In[4]:


movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')


# In[5]:


movies.head(1)
movies.shape


# In[6]:


credits.shape


# In[7]:


movies=movies.merge(credits,on='title')


# In[8]:


movies.head(1)


# In[9]:


movies.info()


# In[10]:


#genres
#id 
#cast 
#crew
#title
#overview
#keywords

movies=movies[['movie_id','title','genres','overview','cast','crew','keywords']]
movies.head(1)
    


# In[11]:


movies.isnull().sum()


# In[12]:


movies.dropna(inplace=True)
movies.isnull().sum()


# In[13]:


movies.duplicated().sum()


# In[14]:


movies.iloc[0].genres


# In[15]:


def convert(obj):
    list=[]
    for i in ast.literal_eval(obj):
        list.append(i['name']);
    return list;


# In[16]:


import ast
movies['genres']=movies['genres'].apply(convert)


# In[17]:


movies['keywords']=movies['keywords'].apply(convert)


# In[18]:


def convert3(obj):
    list=[]
    counter=0
    for i in ast.literal_eval(obj):
        if counter!=3:
            list.append(i['name']);
            counter+=1
        else:    
            break
    return list;


# In[19]:


movies['cast']=movies['cast'].apply(convert3)


# In[20]:


movies


# In[21]:


movies['crew'][0]


# In[22]:


def director(obj):
    list=[]
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            list.append(i['name']);
            break;
    return list;


# In[23]:


movies['crew']=movies['crew'].apply(director)


# In[24]:


movies['overview'][0]


# In[25]:


#converting into list from string for concatenating

movies['overview']=movies['overview'].apply(lambda x:x.split())


# In[26]:


movies.head(1)


# In[27]:


#removing space between two words

movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","")for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","")for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","")for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","")for i in x])


# In[28]:


#concate columns

movies['tags']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']


# In[29]:


movies=movies[['movie_id','title','tags']]


# In[30]:


#converting list into string

movies['tags']=movies['tags'].apply(lambda x:" ".join(x))
movies.head(1)


# In[31]:


movies['tags'][0]


# In[32]:


#converting data into lowercase

movies['tags'].apply(lambda x:x.lower())


# In[33]:


movies.head(1)


# In[34]:


movies['tags'][0]


# In[35]:


#natural language processing libraray
import nltk


# In[36]:


from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


# In[37]:


def stem(text):
    y=[]
    
    for i in text.split():
        y.append(ps.stem(i))
        
    return" ".join(y)    


# In[38]:


ps.stem('dancing')


# In[39]:


stem('In the 22nd century, a paraplegic Marine is dispatched to the moon Pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization. Action Adventure Fantasy ScienceFiction cultureclash future spacewar spacecolony society spacetravel futuristic romance space alien tribe alienplanet cgi marine soldier battle loveaffair antiwar powerrelations mindandsoul 3d SamWorthington ZoeSaldana SigourneyWeaver JamesCameron')


# In[40]:


movies['tags']=movies['tags'].apply(stem)


# In[41]:


#Convert a collection of text documents to a matrix of token counts.

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000,stop_words="english")


# In[42]:


vectors=cv.fit_transform(movies['tags']).toarray()


# In[43]:


vectors[0]


# In[44]:


cv.get_feature_names_out()


# In[45]:


from sklearn.metrics.pairwise import cosine_similarity


# In[46]:


similarity=cosine_similarity(vectors)


# In[47]:


cosine_similarity(vectors).shape


# In[48]:


#if 1 then similarity maximum and if 0 then least similar
similarity[0]


# In[49]:


def recommend(movie):
    movie_index=movies[movies['title']==movie].index[0]
    distances=similarity[movie_index]
    movies_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    for i in movies_list:
        print(movies.iloc[i[0]].title)
      
         


# In[54]:


recommend('Independence Day')


# In[ ]:




