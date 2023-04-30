import os
import nltk
import pandas as pd

# Set up NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Set up file paths
path_to_files = '/path/to/files'
filenames = os.listdir(path_to_files)

# Read in the documents
documents = {}
for filename in filenames:
    with open(os.path.join(path_to_files, filename), 'r') as f:
        documents[filename] = f.read()

# Tokenize the documents
tokenized_documents = {}
for filename, document in documents.items():
    tokens = nltk.word_tokenize(document.lower())
    tokenized_documents[filename] = tokens

# Filter out stop words
stop_words = set(nltk.corpus.stopwords.words('english'))
filtered_documents = {}
for filename, tokens in tokenized_documents.items():
    filtered_tokens = [token for token in tokens if token not in stop_words]
    filtered_documents[filename] = filtered_tokens

# Calculate document similarity
similarity_scores = {}
for filename1, tokens1 in filtered_documents.items():
    for filename2, tokens2 in filtered_documents.items():
        if filename1 == filename2:
            continue
        similarity = nltk.jaccard_distance(set(tokens1), set(tokens2))
        if filename1 not in similarity_scores:
            similarity_scores[filename1] = {}
        similarity_scores[filename1][filename2] = similarity

# Convert similarity scores to a DataFrame
similarity_df = pd.DataFrame.from_dict(similarity_scores, orient='index')

# Group documents by author
authors = {}
for filename in filenames:
    author = filename.split('_')[0]
    if author not in authors:
        authors[author] = []
    authors[author].append(filename)

# Filter out imposters
imposters = set()
for author, files in authors.items():
    if len(files) <= 1:
        continue
    author_df = similarity_df.loc[files, files]
    mean_similarity = author_df.mean().mean()
    for file1, row in author_df.iterrows():
        for file2, similarity in row.iteritems():
            if file1 == file2:
                continue
            if similarity < mean_similarity:
                imposters.add(file1)
                imposters.add(file2)

# Print the filtered documents
for filename in filenames:
    if filename not in imposters:
        print(filename)
