
import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF


df = pd.read_csv("/Users/eseosa/Desktop/NLP/UPDATED_NLP_COURSE/05-Topic-Modeling/npr.csv")
print(df.head())


Tfidf_vect = TfidfVectorizer(max_df=0.9, min_df=2, stop_words="english")


dtm = Tfidf_vect.fit_transform(df["Article"])



nmf = NMF(n_components=5, random_state=101)
nmf.fit(dtm)


for idx, topic in enumerate(nmf.components_):
    
    print()
    print(f"Top 15 words in {idx} topic")
    print()
    for i in topic.argsort()[-15:]:
        print(Tfidf_vect.get_feature_names()[i])


results = nmf.transform(dtm)


df["Topic"] = results.argmax(axis=1)


print(df)


