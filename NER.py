# import spacy
#
# nlp = spacy.load("en_core_web_sm")
#
# text = "What video sharing service did Steve Chen, Chad Hurley, and Jawed Karim create in 2005?"
# doc = nlp(text)
#
# print(nlp.pipe_names)
import pandas as pd
dict = {}

data = pd.read_excel("Chat_Intent.xlsx")
message = {data['Index'], data['Message'], data['Intent']}
print(message)
#
# for i in message['Index']:
#     # dict[i] = dict.get(i)
#     print(i)