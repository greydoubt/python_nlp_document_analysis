'''performs topic modeling on a set of documents, and identifies the primary and secondary topics for each document. Additionally, it analyzes the similarity of discrete parts of the document (paragraph-to-paragraph) to detect if the original text was modified to add a second topic.'''

import numpy as np
import pandas as pd
import spacy
import gensim
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
from sklearn.metrics.pairwise import cosine_similarity

# load the pre-trained English language model
nlp = spacy.load("en_core_web_sm")

# define a function to preprocess the text
def preprocess_text(text):
    # tokenize the text using spaCy
    doc = nlp(text)
    # lemmatize the tokens and remove stop words
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and not token.is_space]
    # remove numbers and short tokens
   
