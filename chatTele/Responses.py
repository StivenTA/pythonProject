from datetime import datetime
import pickle
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from sklearn import feature_extraction
import pandas as pd

loaded_model = pickle.load(open('model_pickle_CV','rb'))
data = pd.read_excel('Chat_Intent.xlsx')

intent = data.groupby("Intent").filter(lambda x: len(x) >= 5)
dataframe = data[(data.Intent.isin(intent['Intent']))].reset_index(drop=True)

factory = StemmerFactory()
stemmer = factory.create_stemmer()
stop = set(stopwords.words('indonesian'))

def sample_responses(input_text):
    user_message = []
    user_message.append(input_text)
    message = dataframe['Message']

    X = []
    for kata in message:
        X.append(stemmer.stem(kata))
    X.append(stemmer.stem(user_message[0]))
    # print(X[len(X)-1])
    vectorize = feature_extraction.text.CountVectorizer(stop_words=stop)
    vectorize.fit(X)
    X = vectorize.fit_transform(X)

    prediction = loaded_model.predict(X)

    if prediction[len(prediction)-1] == 0:
        return "Baik Ibu/ Bapak, kelas akan ditiadakan untuk murid"
    if prediction[len(prediction)-1] == 1:
        return "Intent: Ask Attendence"
    if prediction[len(prediction)-1] == 2:
        return "Intent: Ask Class"
    if prediction[len(prediction)-1] == 3:
        return "Intent: Ask Course"
    if prediction[len(prediction)-1] == 4:
        return "Intent: Ask Finance"
    if prediction[len(prediction)-1] == 5:
        return "Intent: Ask Milestone"
    if prediction[len(prediction)-1] == 6:
        return "Intent: Ask Payment"
    if prediction[len(prediction)-1] == 7:
        return "Intent: Ask Progress"
    if prediction[len(prediction)-1] == 8:
        return "Baik Ibu/ Bapak, proses perubahan kelas akan dilakukan oleh admin"
    if prediction[len(prediction)-1] == 9:
        return "Maaf Bot tidak memahami pertanyaan Ibu/Bapak, dapat melakukan pertanyaan kembali"
    if prediction[len(prediction)-1] == 10:
        return "Senang dapat membantu Ibu/ Bapak"
    if prediction[len(prediction)-1] == 11:
        return "Halo Ibu/Bapak, apa yang bisa saya bantu?"
    if prediction[len(prediction)-1] == 12:
        return "Baik Ibu/ Bapak, dapat langsung memasuki kelas melalui link zoom berikut ini"


    return "Jika ada yang ingin ditanyakan lebih detail, dapat dihubungi admin melalui no hp ()"

