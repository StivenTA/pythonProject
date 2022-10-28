import pandas as pd

data = pd.read_excel("Chat_Intent.xlsx")

# print(data["Message"])

dataDict = {}
index = 1
for i in data["Message"]:
    dataDict.setdefault(index, []).append(i)
    index = index + 1
index = 1
for j in data["Intent"]:
    dataDict.setdefault(index, []).append(j)
    # dataDict[index] = dataDict.get(index, i)
    # dataDict[index].append(j)
    index = index + 1
# for i in dataDict:
# print(dataDict)
# print(dataDict)
for i, items in dataDict.items():
    if items[1] == "Farewell":
        print(items[0])
# halo

# kalau mau akses isi Intent saja [i][value]
# value nya beda in aja jadi 1 untuk Intent, 0 untuk Message
myListMessage = [dataDict[i][0] for i in sorted(dataDict.keys())]
myListIntent = [dataDict[i][1] for i in sorted(dataDict.keys())]
#
# print(myListIntent)
# newDict = {}
# index = 1
# for i in range(len(myListMessage)):
#     # print(myList[i])
#     newDict[index] = newDict.get(index, myListMessage[i])
#     # newDict.setdefault(index, []).append(newDict.get(index, myListIntent[i+1]))
#     index+=1
# index = 0
# for i in range(len(myListIntent)):
#     # print(myList[i])
#     newDict.setdefault(index, []).append(i)
#     # newDict.setdefault(index, []).append(newDict.get(index, myListIntent[i+1]))
#     index+=1
# print(type(newDict.setdefault(1)))
# print(newDict.get(index, myList[i]))
print(newDict)
