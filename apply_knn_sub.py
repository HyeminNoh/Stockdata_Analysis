import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from linear_algebra import distance
from stats import mean
import math, random

def select_stock():
    data = pd.read_csv("stock_history.csv", sep=",", encoding='euc-kr')
    #종목 이름 전체 종목 확인 -> 중복된것제거
    stockname = data["stockname"]
    stockname = stockname.drop_duplicates()
    stockname = stockname.reset_index(drop=True)
    #임의로 5개만 리스트 출력해 냄
    #stocklist=[stockname[0], stockname[1], stockname[2], stockname[3], stockname[4], stockname[5]]
    #print("주식 종목 리스트:"+str(stocklist))
    #input_stockname = input('종목을 입력: ')
    result = (data[data['stockname'].isin(["고려아연"])])  #선택한 주식 종목명을 가진 행만 출력
    result = result.reset_index(drop=True)
    return result

#종가 일간 변화량 과 종가 일간 변화율
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
            cv_diff_rate.append(round(diff_rate,2))
    selected_data["cv_diff_rate"] = cv_diff_rate
    result_data = selected_data
    return result_data

# cv_maN_value: 종가의 N일 이동평균, (예: N=5)단, 5일이 안되는 기간은 제외 (예:  10(금),13(월), 14(화),15(수),16(목) 종가의 평균은 16일의     cv_ma5_value)
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
def set_udNd(N_day, data):
    diff_value = data['cv_diff_value']
    udNd = []
    for index in diff_value.index.values:
        if index > 0 and index <= N_day-1:
            udNd.insert(index-1, 0)

        elif index > N_day-1:
            #5개의 데이터씩 선택 후 복사 복사한 데이터를 검사 후 리스트화
            five_values = diff_value[index-(N_day-1):index+1].copy()
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
    udNd.append(0)
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
    cvNd_diff_rate.append(0)
    data["cvNd_diff_rate"] = cvNd_diff_rate
    result_data = data
    return result_data


def knn_classify(k, labeled_points, new_point):
    """매개변수설명
        k : 어느정도 가까운 것들을 찾는가
        labeled_points : 분류에 사용 될 데이터목록들
        new_point : 분류하고 싶은 데이터
                    1. 분류에 사용될 데이터들을 분류 될 데이터와 거리 순으로 정렬한다.
                    2. 정렬된 데이터 중에서 k 거리 이내에 있는 데이터 목록만 따로 majority_vote에 넘겨서
                    k 거리이내의 데이터들 중에 가장 많이 포함되 있는 라벨을 찾는다.
    """
    """each labeled point should be a pair (point, label)"""

    # order the labeled points from nearest to farthest
    by_distance = sorted(labeled_points,
                         key=lambda point_label: distance(point_label[0], new_point))

    # find the labels for the k closest
    k_nearest_labels = [label for _, label in by_distance[:k]]
    # and let them vote
    return majority_vote(k_nearest_labels)

def majority_vote(labels):
    """assumes that labels are ordered from nearest to farthest"""
    # Counter는 사전(dict) 클래스의 하위 클래스로 리스트나 튜플에서 각 데이터가 등장한 횟수를 사전 형식으로 돌려준다
    vote_counts = Counter(labels)
    #Counter 클래스의 most_common(n) 메쏘드는 등장한 횟수를 내림차순으로 정리 n은 상위 n개
    winner, winner_count = vote_counts.most_common(1)[0]
    num_winners = len([count
                       for count in vote_counts.values()
                       if count == winner_count])

    if num_winners == 1:
        return winner  # unique winner, so return it
    else:
        return majority_vote(labels[:-1])  # try again without the farthest

def classify_data(data):
    total_index = len(data.index)
    # 30% 테스트 데이터, 70% 학습데이터
    #올림 적용 정수형태로 개수 반환
    test_num = math.ceil(total_index*0.3)

    test_data = data.iloc[:test_num, 1:]
    training_data = data.iloc[test_num:, 1:]
    test_data = test_data.reset_index(drop=True)
    training_data = training_data.reset_index(drop=True)
    return (test_data, training_data)

def data_cook(data):
    cv_diff_rate = data['cv_diff_rate']
    cv_maN_rate = data['cv_maN_rate']
    cvNd_diff_rate = data['cvNd_diff_rate']

    udNd = data['ud_Nd']
    data_cook = []
    for index in cv_maN_rate.index.values:
        data_cook.append((cv_diff_rate[index] ,cv_maN_rate[index], str(udNd[index])))

    data_cook = [([diffrate, maNrate], UDND) for diffrate, maNrate, UDND in data_cook]


    return data_cook

def total_prepared_data():
    # 종목선택 데이터 자르기
    selected_data = select_stock()
    # print(selected_data)
    # 데이터 준비, 변수추가
    prepared_data = prepare_data(selected_data)  # 종가 일간 변화량, 변화율 추가
    # print(prepared_data)
    prepared_data = cv_moveAverage_value(3, prepared_data)  # N_day 평균 병화량
    prepared_data = cv_moveAverage_rate(3, prepared_data)
    prepared_data = set_udNd(3, prepared_data)
    prepared_data = cvNd_diff_rate(3, prepared_data)

    # 날짜 내림차순으로 재 정렬
    prepared_data = prepared_data.sort_values(["basic_date"], ascending=[False])
    result_data = prepared_data.reset_index(drop=True)

    result_data.to_csv("stock_history_added.csv", mode='w', encoding='cp949')

    return result_data


def k_change_plot(k, data):
    test_data = classify_data(data)[0]
    training_data = classify_data(data)[1]

    testdata = data_cook(test_data)
    trainingdata = data_cook(training_data)
    accuracy_plot = []
    k_plot = []
    for i in range(1, k+2, 2):
        num_correct = 0

        for rate, actual_udNd in testdata:
            predicted_language = knn_classify(i, trainingdata, rate)
            if predicted_language == actual_udNd:
                num_correct += 1

        accuracy = num_correct / len(testdata) * 100

        for index in test_data.index.values:
            k_plot.append(i)
            accuracy_plot.append(accuracy)

        print(i, "neighbor[s]:", num_correct, "correct out of", len(testdata), accuracy, "%")

    plt.plot(k_plot, accuracy_plot, marker='o')
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.xticks(np.arange(1,k+2,2))
    plt.show()

def show_accuracy():
    input_data = pd.read_csv("stock_history_added.csv", sep=",", encoding='cp949')
    test_data = classify_data(input_data)[0]
    training_data = classify_data(input_data)[1]

    testdata = data_cook(test_data)
    trainingdata = data_cook(training_data)

    num_correct = 0

    for rate, actual_udNd in testdata:
        predicted_language = knn_classify(11, trainingdata, rate)
        if predicted_language == actual_udNd:
            num_correct += 1

    accuracy = num_correct / len(testdata) * 100

    accuracy_data = []
    k_data = []
    for index in test_data.index.values:
        k_data.append(11)
        accuracy_data.append(accuracy)

    test_data["K"] = k_data
    test_data["k_udnd"] = accuracy_data
    test_data.to_csv("stock_history_K.csv", mode='w', encoding='cp949')


if __name__ == '__main__':

    #데이터 준비 1번 과정
    total_prepared_data()

    #K값에 따른 정확도 제시, 인자는 보고싶은 K값의 범위
    k_change_plot(21, total_prepared_data())


    #가장 정확도가 높은 K값에 따른 예측값이 포함된 데이터 파일 생성, 이 함수에선 K는 11로 지정되어있음
    show_accuracy()