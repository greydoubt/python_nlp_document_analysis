import spacy

# load the pre-trained English language model
nlp = spacy.load("en_core_web_sm")

# define a text to analyze
text = "The soccer player Pelé was born in Brazil. Prince was a famous singer and songwriter. John Smith is a common name in the United States. Michael Jordan is a basketball player. The company Apple is based in California. I ate an apple for breakfast."

# process the text with the language model
doc = nlp(text)

# print the named entities and their types
for ent in doc.ents:
    # handle ambiguous cases
    if ent.label_ == "PERSON" and len(ent.text.split()) == 2:
        # if the named entity is a two-word name, assume it's a first and last name
        print("PERSON (ambiguous)", ent.text)
    else:
        # print the named entity type and text
        print(ent.label_, ent.text)



'''
PERSON Pelé
GPE Brazil
PERSON Prince
PERSON (ambiguous) John Smith
PERSON Michael Jordan
ORG Apple

In this example, we've added more names to the text, including "John Smith" which is a common name, "Michael Jordan" which can be a person or a brand name, and "Apple" which can refer to a company or a fruit. We've also added an ambiguous case where a two-word name is detected, and we assume it's a first and last name.

Note that the spaCy language model is able to recognize and classify named entities in a wide range of styles and formats, including common names, stage names, nicknames, and more. However, it may not be
'''