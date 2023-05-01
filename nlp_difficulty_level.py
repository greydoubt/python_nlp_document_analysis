'''Python script that uses Natural Language Processing to classify text into difficulty levels, and estimate the grade-level required for reading. Additionally, it considers profanity and "triggering subjects" as more advanced, compared to similar documents.'''



import spacy
from textstat import flesch_reading_ease

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

# define a function to classify text into difficulty levels
def classify_text(text):
	# preprocess the text
	text = preprocess_text(text)
	# compute the Flesch reading ease score
	score = flesch_reading_ease(text)
	# classify the text based on the score
	if score > 90:
		return "Very easy (1st grade level)"
	elif score > 80:
		return "Easy (2nd-3rd grade level)"
	elif score > 70:
		return "Fairly easy (4th-5th grade level)"
	elif score > 60:
		return "Standard (6th-8th grade level)"
	elif score > 50:
		return "Fairly difficult (9th-10th grade level)"
	elif score > 30:
		return "Difficult (11th-12th grade level)"
	else:
		return "Very difficult (college level)"

#define a function to classify text with profanity and triggering subjects as more advanced
def advanced_classification(text):
	# preprocess the text
	text = preprocess_text(text)
	# check if the text contains profanity or triggering subjects
		if "profanity_word_1" in text or "profanity_word_2" in text or "trigger_word_1" in text or "trigger_word_2" in 	text:
			# classify the text as advanced
			return "Advanced (college level)"
		else:
	# classify the text using the standard classification function
			return classify_text(text)

#test the function on some sample text
text1 = "The cat sat on the mat. It was a sunny day."
text2 = "The cat sat on the f*cking mat. It was a sunny day."
text3 = "The cat sat on the mat. It was a dark and stormy night. Trigger warning: this story contains graphic violence."

print("Text 1 classification:", classify_text(text1))
print("Text 2 classification:", advanced_classification(text2))
print("Text 3 classification:", advanced_classification(text3))

#preprocess_text() function tokenizes the input text using spaCy,
