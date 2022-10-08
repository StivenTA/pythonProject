# Standard Python Libaries
import urllib.request
import os

# Third Party Modules
import matplotlib.pyplot as plt
import pandas as pd # pip install pandas openpyxl
data = []
with open(r"./WhatsApp_Chat_with_Pop_Aga_13yo_Cbb.txt",encoding="utf-8",mode='r') as f:
    data.append(f.readlines())
with open(r"./WhatsApp_Chat_with_Andini__Amanda.txt",encoding="utf-8",mode='r') as f:
    data.append(f.readlines())
with open(r"./WhatsApp_Chat_with_Rian_davi_10yo.txt",encoding="utf-8",mode='r') as f:
    data.append(f.readlines())
with open(r"./WhatsApp_Chat_with_Billy_Toby_10__AGI_7.txt",encoding="utf-8",mode='r') as f:
    data.append(f.readlines())
with open(r"./WhatsApp_Chat_with_Garin__Abiyasa_14_Yo_CBB.txt",encoding="utf-8",mode='r') as f:
    data.append(f.readlines())
# print(data)
exportdata = pd.DataFrame(columns = ['Date', 'Time', 'Name', 'Message'])
for each in data:
    dataset = each[1:]
    # print(dataset)
    cleaned_data = []
    for line in dataset:
        # Check, whether it is a new line or not
        # If the following characters are in the line -> assumption it is NOT a new line
        if '/' in line and ',' in line and ':' in line and '-' in line:
            # grab the info and cut it out
            date = line.split(",")[0]
            # print(date)
            line2 = line[len(date):]
            time = line2.split("-")[0][2:]
            # print(time)
            line3 = line2[len(time):]
            name = line3.split(":")[0][4:]
            # print(name)
            line4 = line3[len(name):]
            message = line4[6:-1]  # strip newline character
            # print(message)
            cleaned_data.append([date, time, name, message])

        # else, assumption -> new line. Append new line to previous 'message'
        else:
            # print(line)
            new = cleaned_data[-1][-1] + " " + line
            cleaned_data[-1][-1] = new
    df = pd.DataFrame(cleaned_data, columns = ['Date', 'Time', 'Name', 'Message'])
    # print(df)
    frames = [exportdata,df]
    exportdata = pd.concat(frames,sort=False)
print(exportdata)
exportdata.to_excel('chat_history.xlsx', index=False)
#
# df['Time'].value_counts().head(10).plot.barh()
# plt.xlabel('Number of messages')
# plt.ylabel('Time')