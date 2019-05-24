import pandas as pd
import numpy as np

def print_stock_category():
    data = pd.read_csv("stock_history.csv", sep=",", encoding='euc-kr')
    # 종목 이름 전체 종목 확인 -> 중복된것제거
    stockname = data["stockname"]
    stockname = stockname.drop_duplicates()
    stockname = stockname.reset_index(drop=True)
    stockname.to_csv("stock_category.csv", mode='w', encoding='cp949')
    #print(stockname)

def select_stock():
    data = pd.read_csv("stock_history.csv", sep=",", encoding='euc-kr')
    #종목 이름 전체 종목 확인 -> 중복된것제거
    stockname = data["stockname"]
    stockname = stockname.drop_duplicates()
    stockname = stockname.reset_index(drop=True)
    #임의로 5개만 리스트 출력해 냄
    #stocklist=[stockname[0], stockname[1], stockname[2], stockname[3], stockname[4], stockname[5]]
    #print("주식 종목 리스트:"+str(stocklist))
    input_stockname = input('종목이름 입력: ')
    result = (data[data['stockname'].isin([input_stockname])])  #선택한 주식 종목명을 가진 행만 출력
    result = result.reset_index(drop=True)
    return result

def prepare_data(selected_data):
    #날짜 오름차순으로 정렬
    selected_data = selected_data.sort_values(["basic_date"], ascending=[True])
    selected_data = selected_data.reset_index(drop=True)
    #종가 일간 변화량 cv_diff_value열
    close_value = selected_data['close_value']
    cv_diff_value = []
    for index in close_value.index.values:
        if index == 0:
            cv_diff_value.append(0)
        else:
            diff_value = close_value[index]-close_value[index-1]
            #print(index, diff_value)
            cv_diff_value.append(diff_value)
    selected_data["cv_diff_value"] = cv_diff_value

    # 종가 일간 변화율 cv_diff_rate열
    cv_diff_rate = []
    for index in close_value.index.values:
        if index == 0:
            cv_diff_rate.append(0)
        else:
            if close_value[index] == close_value[index-1]:
                diff_rate = 0
            else:
                if close_value[index] > close_value[index-1]:    #증가율
                    diff_rate = (close_value[index]-close_value[index-1])/close_value[index-1]*100
                elif close_value[index] < close_value[index-1]:  #감소율
                    diff_rate = -(close_value[index-1] - close_value[index])/close_value[index-1]*100
            #print(index, diff_rate)
            #소수점둘째자리까지
            cv_diff_rate.append(round(diff_rate, 2))
    selected_data["cv_diff_rate"] = cv_diff_rate
    result_data = selected_data
    return result_data

# cv_maN_value: 종가의 N일 이동평균, (예: N=5)단, 5일이 안되는 기간은 제외 (예:  10(금),13(월), 14(화),15(수),16(목) 종가의 평균은 16일의 cv_ma5_value)
def cv_moveAverage_value(N_days, data):
    moveAverage_value = data['close_value'].rolling(window=N_days).mean()
    data["cv_maN_value"] = moveAverage_value
    result_data = data.fillna(0)
    return result_data

#cv_maN_rate: 종가의 N일 이동평균의 일간 변화율, (예:N=5) cv_ma5_rate, 단, 5일이 안되는 기간은 제외
def cv_moveAverage_rate(N_day, data):
    cv_diff_rate = data['cv_maN_value']
    moveAverage_rate = []
    for index in cv_diff_rate.index.values:
        if index < N_day-1:
            moveAverage_rate.append(0)
        elif index >= N_day-1:
            if cv_diff_rate[index] == cv_diff_rate[index - 1] or cv_diff_rate[index-1] == 0:
                average_rate = 0
            else:
                if cv_diff_rate[index] > cv_diff_rate[index - 1]:
                    average_rate = ((cv_diff_rate[index] - cv_diff_rate[index - 1])/cv_diff_rate[index-1]) * 100
                elif cv_diff_rate[index] < cv_diff_rate[index - 1]:
                    average_rate = (cv_diff_rate[index] - cv_diff_rate[index - 1])/abs(cv_diff_rate[index-1]) * 100
            # print(index, diff_rate)
            # 소수점둘째자리까지
            moveAverage_rate.append(round(average_rate, 2))
    data["cv_maN_rate"] = moveAverage_rate
    result_data = data
    return result_data

'''
    ud_Nd - 
    (a). N일 연속 종가가 상승할 때, (N-1)번째 날의 값은 1, 
    (b) N일 연속 종가가 하락할 때, (N-1)번째 날의 값은 -1, 
    (c) 그렇지 않은 날의 (N-1)번째 날 값은 0
    (예: ud_5d) 13(월), 14(화),15(수),16(목), 17(금) 종가의 전일대비상승하면, 16(목)의 값은 1, (주의) 13일은 10일 종가보다 상승
'''
#수정필요
def set_udNd(N_day, data):
    diff_value = data['cv_diff_value']
    udNd = []
    for index in diff_value.index.values:
        if index > 0 and index < N_day-1:
            udNd.insert(index-1, 0)
        elif index >= N_day-1:
            #5개의 데이터씩 선택 후 복사 복사한 데이터를 검사 후 리스트화
            five_values = diff_value[index-N_day:index].copy()
            five_values[five_values > 0] = 1
            five_values[five_values < 0] = 2
            five_values[five_values == 0] = 3
            list = five_values.values.T.tolist()
            if list.count(1) == N_day:
                udNd.insert(index-1, 1)
            elif list.count(2) == N_day:
                udNd.insert(index-1, -1)
            else:
                udNd.insert(index-1, 0)
    udNd.append('-')
    data["ud_Nd"] = udNd
    result_data = data
    return result_data

#   cvNd_diff_rate: N일간의 종가 상승률을 (N-1)번째 날의 값으로 설정,
#   (예: cv5d_diff_rate = [17(금) 종가 - 10(금)종가] / 10(금)종가] 값을 16(목)에 설정)
def cvNd_diff_rate(N_days, data):
    close_value = data['close_value']
    cvNd_diff_rate = []
    for index in close_value.index.values:
        if index > 0 and index < N_days-1:
            cvNd_diff_rate.insert(index-1, 0)
        elif index >= N_days-1:
            if close_value[index] == close_value[index - N_days+1] or close_value[index - N_days+1] == 0:
                cvNd_rate = 0
            else:
                if close_value[index] > close_value[index - N_days+1]:
                    cvNd_rate = ((close_value[index] - close_value[index - N_days+1])/close_value[index - N_days+1]) * 100
                elif close_value[index] < close_value[index - N_days+1]:
                    cvNd_rate = -((close_value[index - N_days+1] - close_value[index])/close_value[index - N_days+1]) * 100
            # print(index, diff_rate)
            # 소수점셋째자리까지
            cvNd_diff_rate.insert(index-1, round(cvNd_rate, 2))
    cvNd_diff_rate.append('-')
    data["cvNd_diff_rate"] = cvNd_diff_rate
    result_data = data
    return result_data

if __name__ == '__main__':
    #print_stock_category()
    #종목선택 데이터 자르기
    selected_data = select_stock()
    #print(selected_data)
    #데이터 준비, 변수추가
    prepared_data = prepare_data(selected_data)                 #종가 일간 변화량, 변화율 추가
    #print(prepared_data)
    prepared_data = cv_moveAverage_value(3, prepared_data)      #N_day 평균 병화량
    prepared_data = cv_moveAverage_rate(3, prepared_data)
    prepared_data = set_udNd(3, prepared_data)
    prepared_data = cvNd_diff_rate(3, prepared_data)
    #날짜 내림차순으로 재 정렬
    prepared_data = prepared_data.sort_values(["basic_date"], ascending=[False])
    result_data = prepared_data.reset_index(drop=True)
    #print(result_data)
    result_data.to_csv("stock_history_added.csv", mode='w', encoding='cp949')
