import pandas as pd
from numpy import numarray as num

# formatted_float = "${:,.2f}".format(1500.2)
# print(formatted_float)

modal = 500000
profit = 0.05
week = input("Please input the week: ")
profit_acc = 1
count = 0
profit_dataframe = []
# print("======================================================================")
for i in range(0, int(week)):
    #   print("----------------------------------------------------------------------")
    #   print("Modal sebelumnya " + str(i+1) + ": " + str("${:,.2f}".format(modal)))
    # if (i + 1) <= 20:
    #     #       print("Kerugian : " + str("${:,.2f}".format(modal * profit)))
    #     modal -= modal * profit
    # elif (i + 1) >= 100 and (i + 1) <= 150:
    #     loss = 0.02
    #     modal -= modal * loss
    # else:
        #       print("Keuntungan : " + str("${:,.2f}".format(modal * profit)))
    modal += modal * profit
    profit_acc += profit_acc * profit
    # print("Modal sekarang    " + str(i + 1) + ": " + str("${:,.2f}".format(modal)))
    # print("Profit Accumulation    " + str(i + 1) + ": " + str(profit_acc))
    profit_dataframe += [modal]
# print("======================================================================")

dataframe = pd.DataFrame(profit_dataframe)
# print(dataframe)
# dataframe.to_csv(r'C:\Users\stiven tri alvin\Desktop\provit1.csv')
