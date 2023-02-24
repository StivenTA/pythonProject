from datetime import datetime
import pickle
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from sklearn import feature_extraction
import pandas as pd

loaded_model = pickle.load(open('model_pickle_CV','rb'))
loaded_data_training = pickle.load(open('training_data_vectorize','rb'))


factory = StemmerFactory()
stemmer = factory.create_stemmer()
stop = set(stopwords.words('indonesian'))

def sample_responses(input_text):
    user_message = []
    user_message.append(stemmer.stem(input_text))
    # print(X[len(X)-1])
    vectorize = loaded_data_training

    user_message_transform = vectorize.transform(user_message)

    prediction = loaded_model.predict(user_message_transform)

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


    return "maaf untuk pertanyaan bapak/ibu masih belum dapat dimengerti oleh chatbot. Atas ketidakmampuan chatbot maka bapak/ibu dapat lansung menghubungi admin melalui no hp (0801241530453)"

