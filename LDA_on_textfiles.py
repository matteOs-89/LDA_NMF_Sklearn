
import pandas as pd
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


df = pd.read_csv("/Users/eseosa/Desktop/NLP/UPDATED_NLP_COURSE/05-Topic-Modeling/npr.csv")


print(df.head())


count_vect = CountVectorizer(max_df=0.9, min_df=2, stop_words="english")


dtm = count_vect.fit_transform(df["Article"])

print(dtm)


LDA = LatentDirichletAllocation(n_components=5, random_state=42)
LDA.fit(dtm)


"""Now to get the vocabulary of words"""

def get_random_words(count_vect, length):
    
    """this function will randomly select words from 
    get_feature_names  and return the word."""
    
    random_idx = random.randint(0, length)
    random_word = count_vect.get_feature_names()[random_idx]
    
    print(random_word)


length = len(count_vect.get_feature_names())

get_random_words(count_vect, length)


"""Now to visualize the Topic allocations created using LDA"""

print(LDA.components_.shape)


for idx, topic in enumerate(LDA.components_):
    
    print()
    print(f"Top 15 words in {idx} topic")
    print()
    for i in topic.argsort()[-15:]:
        print(count_vect.get_feature_names()[i])


results = LDA.transform(dtm)

print(results.shape)

df["Topic"] = results.argmax(axis=1)

print(df)


