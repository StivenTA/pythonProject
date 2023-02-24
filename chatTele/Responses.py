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
        return "Bapak/Ibu, kami tetap akan beroperasi secara normal dari hari senin hingga sabtu"\
    "secara offline maupun online dan jika anak Bapak/Ibu mengalami keterlambatan atau ingin mengambil izin untuk tidak masuk" \
    "silakan untuk memberikan alasan terhadap ketidakmasukannya."
    if prediction[len(prediction)-1] == 2:
        return "Bapak/Ibu kelas bisa dilakukan secara online maupun offline dan maksimal murid perkelasnya ialah 3 murid"

    if prediction[len(prediction)-1] == 3:
        return "SD Level:\n"\
    "Memperdalam konsep coding, Computational Thinking,"\
    "dan computer science: menggunakan online platforms popular:"\
    "Tynker, Scratch, CodeMonkey, MIT App Inventor, Grok Learning membuat"\
    "animasi, video game, web pages, mobile apps membuat custom Minecraft mod"\
    "latihan computational thinking untuk kompetisi Bebras perkenalan ke dasar"\
    "text-based coding languages (HTML, Python, Java, C++, C, Javascript) dan"\
    "konsep Artificial Intelligent – Maching Learning."\
    "\n"\
    "SMP Level:\n"\
    "Memperdalam konsep coding, Computational Thinking, dan"\
    "computer science untuk level SMP: perkenalan dasar coding dengan"\
    "menggunakan block programming dan dilanjutkan ke text-based coding languages"\
    "(HTML, Python, Java, C++, C, Javascript) membuat animasi, websites,"\
    "mobile apps, games dengan menggunakan programming latihan computational thinking"\
    "untuk kompetisi Bebras perkenalan konsep Artificial Intelligent – Maching Learning"
    "\n"\
    "SMA Level:\n"\
    "Memperdalam konsep programming dan dilanjutkan dengan eksplorasi"\
    "coding di area berikut: Game Development Web Development App Development Computer Science,"\
    "Algorithm, and Math in Coding Data, Database and AI Intro"

    if prediction[len(prediction)-1] == 4:
        return "SD Level:\n"\
        "Green(14 sesi) 2jt 250rb"\
        "Purple(29 sesi) 4jt 500rb" \
               "\n" \
        "SMP Level\n:"\
        "Green(14 sesi) 2jt 550rb"\
        "Purple(29 sesi) 5jt 100rb" \
               "\n" \
        "SMA Level\n:"\
        "Green(14 sesi) 2jt 850rb"\
        "Purple(29 sesi) 5jt 700rb"\

    if prediction[len(prediction)-1] == 5:
        return " Baik Ibu/Bapak, proses pengiriman percapaian Ibu/Bapak akan kami kirimkan secara bentuk non-fisik dan fisik"
    if prediction[len(prediction)-1] == 6:
        return "Bapak/Ibu terhomat pembayaran dapat dikirim kepada nomor rekening BCA berikut ini 0123 016 111"
    if prediction[len(prediction)-1] == 7:
        return "Ibu/Bapak dapat melihat progress anak bapak melalui aplikasi Kodekiddo kami"
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

