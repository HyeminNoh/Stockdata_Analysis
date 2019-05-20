#Kyungsub code

import pandas

line_counter = 0
line_counter2 = 0
serial_numbering = 1
stockname_numbering_check = ""
header = []
header_numbering = []
stock_list_numbering = []
stock_list = []

with open('stock_history_short.csv') as f:
    while 1:
        data = f.readline()

        if not data: break
        if line_counter == 0:
            header = data.split(",") # 맨 첫 줄은 header로 저장
            header_numbering.append("Serial_Numbering")
            header[7]=header[7].replace("\n", "")
            k = 0
            for i in header:
                header_numbering.append(header[k])
                k += 1
        else:
            stock = data.split(",")
            stock_list.append(stock)

        line_counter += 1

#print(stock_list)
header_numbering.append("cv_diff_value")
header_numbering.append("cv_diff_rate")
header_numbering.append("cv_maN_value")
header_numbering.append("cv_maN_rate")
header_numbering.append("ud_Nd")
header_numbering.append("cvNd_diff_rate\n")

with open('stock_history_short_added.csv', 'w') as f:
    f.write(",".join(header_numbering))
    for stock in stock_list:
        if stockname_numbering_check == stock[1]:
            f.write(str(serial_numbering)+",")
            f.write(",".join(stock))
            stockname_numbering_check = stock[1]
            serial_numbering +=1

        else:
            serial_numbering = 1
            f.write(str(serial_numbering) + ",")
            f.write(",".join(stock))
            stockname_numbering_check = stock[1]
            serial_numbering += 1


with open('stock_history_short_added.csv') as f:
    while 1:
        data = f.readline()
        if not data: break
        if line_counter2 == 0:
            header = data.split(",") # 맨 첫 줄은 header로 저장
        else:
            stock = data.split(",")
            stock_list_numbering.append(stock)

        line_counter2 += 1

"""cv_diff_value 구하는 과정"""
for i in range(len(stock_list_numbering)-1):
    stock_list_numbering[i][8] = stock_list_numbering[i][8].replace('\n',"")
    if int(stock_list_numbering[i][0]) - int(stock_list_numbering[i+1][0]) == -1 :
        stock_list_numbering[i].append(str(int(stock_list_numbering[i][7]) - int(stock_list_numbering[i+1][7])))
    else:
        stock_list_numbering[i].append("0")
stock_list_numbering[len(stock_list_numbering)-1][8] = stock_list_numbering[len(stock_list_numbering)-1][8].replace("\n", "")
stock_list_numbering[len(stock_list_numbering)-1].append("0")

"""cv_diff_rate 구하는 과정"""
for i in range(len(stock_list_numbering)-1):
    if int(stock_list_numbering[i][0]) - int(stock_list_numbering[i+1][0]) == -1 :
        stock_list_numbering[i].append(str(round((int(stock_list_numbering[i][7]) - int(stock_list_numbering[i+1][7])) / int(stock_list_numbering[i+1][7]), 3)))

    else:
        stock_list_numbering[i].append("0")


stock_list_numbering[len(stock_list_numbering)-1].append("0")


with open('stock_history_short_added.csv', 'w') as f:
    f.write(",".join(header_numbering))
    for stock in stock_list_numbering:
        f.write(",".join(stock)+'\n')


print("finish")