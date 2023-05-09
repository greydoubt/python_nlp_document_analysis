import numpy as np
import pandas as pd
import spacy
from gensim import corpora, models
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
    tokens = [token for token in tokens if not token.isdigit() and len(token) > 2]
    # join the tokens back into a string
    text = " ".join(tokens)
    return text

# define a function to perform topic modeling and identify primary and secondary topics
def perform_topic_modeling(documents):
    # preprocess the documents
    preprocessed_docs = [preprocess_text(doc) for doc in documents]
    
    # create a dictionary of the preprocessed documents
    dictionary = corpora.Dictionary([doc.split() for doc in preprocessed_docs])
    
    # convert the preprocessed documents into a document-term matrix
    doc_term_matrix = [dictionary.doc2bow(doc.split()) for doc in preprocessed_docs]
    
    # perform LDA topic modeling
    lda_model = models.LdaModel(doc_term_matrix, num_topics=2, id2word=dictionary, passes=10)
    
    # extract the primary and secondary topics
    primary_topic = lda_model.print_topic(0)
    secondary_topic = lda_model.print_topic(1)
    
    return primary_topic, secondary_topic

# define a function to analyze the similarity of discrete parts of the document
def analyze_document_similarity(document):
    # preprocess the document
    preprocessed_doc = preprocess_text(document)
    
    # split the document into paragraphs
    paragraphs = preprocessed_doc.split("\n\n")  # assuming paragraphs are separated by two newlines
    
    # calculate the similarity between paragraph pairs using cosine similarity
    similarity_scores = []
    for i in range(len(paragraphs) - 1):
        for j in range(i+1, len(paragraphs)):
            p1 = paragraphs[i].split()
            p2 = paragraphs[j].split()
            vector1 = nlp(" ".join(p1)).vector.reshape(1, -1)
            vector2 = nlp(" ".join(p2)).vector.reshape(1, -1)
            similarity_score = cosine_similarity(vector1, vector2)[0][0]
            similarity_scores.append(similarity_score)
    
    # calculate the average similarity score
    average_similarity_score = np.mean(similarity_scores)
    
    return average_similarity_score

# sample documents
documents = [
    "This is an informative document about the benefits of product X. It provides detailed information on its features, specifications, and usage.",
    "Introducing the amazing product X! It will revolutionize your life and solve all your problems. Don't miss out on this incredible opportunity!",
    "Here's an overview of product X and how it can improve your daily routine. Get ready for a transformative experience like never before!",
    "Product X is the ultimate solution you've been waiting for. It offers unparalleled benefits for your needs. With product X, you'll achieve remarkable results and outperform your competition. Act now and experience the power of product X!"]

#perform topic modeling

primary_topic, secondary_topic = perform_topic_modeling(documents)
analyze document similarity

average_similarity_score = analyze_document_similarity(documents[1])
#print the results

print("Primary Topic:", primary_topic)
print("Secondary Topic:", secondary_topic)
print("Average Similarity Score:", average_similarity_score)

'''
define the `preprocess_text()` function to tokenize the text, lemmatize the tokens, remove stop words, and perform additional preprocessing steps. The `perform_topic_modeling()` function uses the preprocessed documents to create a dictionary and a document-term matrix, and then applies LDA to extract the primary and secondary topics. The `analyze_document_similarity()` function preprocesses the document, splits it into paragraphs, and calculates the similarity between paragraph pairs using cosine similarity.

The script then applies these functions to the provided sample documents, and prints out the primary topic, secondary topic, and average similarity score. The primary topic and secondary topic represent the main themes identified in the documents, while the average similarity score gives an indication of how similar different parts of the document are, which can help detect modifications or shifts in topics.

'''