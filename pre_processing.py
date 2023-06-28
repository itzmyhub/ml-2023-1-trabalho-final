import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize


def pre_processing():

    dataset = pd.read_csv('sentiment-emotion-labelled_Dell_tweets.csv')

    # remover caracteres especiais e números dos textos/deixar todas as letras minúsculas
    dataset['Text'] = dataset['Text'].apply(lambda x: re.sub('[^a-zA-Z]', ' ', x))
    dataset['Text'] = dataset['Text'].str.lower()

    # remover stopwords
    stop_words = set(stopwords.words('english'))
    dataset['Text'] = dataset['Text'].apply(lambda x: ' '.join(word for word in word_tokenize(x) if word not in stop_words))

    # realizar lematização
    lemmatizer = WordNetLemmatizer()
    dataset['Text'] = dataset['Text'].apply(lambda x: ' '.join(lemmatizer.lemmatize(word) for word in word_tokenize(x)))

    # Stemming
    stemmer = PorterStemmer()
    dataset['Text'] = dataset['Text'].apply(lambda x: ' '.join(stemmer.stem(word) for word in word_tokenize(x)))

    return dataset
