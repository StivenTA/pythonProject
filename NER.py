import spacy

nlp = spacy.load("id_core_web_sm")

text = "What video sharing service did Steve Chen, Chad Hurley, and Jawed Karim create in 2005?"
doc = nlp(text)

print(nlp.pipe_names)