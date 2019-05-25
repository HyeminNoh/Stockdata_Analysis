import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from linear_algebra import distance
import math

def classify_data(data):
    total_index = len(data.index)
    # 30% 테스트 데이터, 70% 학습데이터
    # 올림 적용 정수형태로 개수 반환
    test_num = math.ceil(total_index*0.3)

    test_data = data.iloc[:test_num, 1:]
    training_data = data.iloc[test_num:, 1:]
    test_data = test_data.reset_index(drop=True)
    training_data = training_data.reset_index(drop=True)
    return (test_data, training_data)

def plot_state_borders(plt, color='0.8'):
    pass

def plot_udNd(data):
    # key is language, value is pair (longitudes, latitudes)
    plots = {"1": ([], []), "-1": ([], []), "0": ([], [])}

    # we want each language to have a different marker and color
    markers = {"1": "o", "-1": "s", "0": "^"}
    colors = {"1": "r", "-1": "b", "0": "g"}

    cv_diff_rate = data['cv_diff_rate']
    cv_maN_rate = data['cv_maN_rate']
    udNd = data['ud_Nd']
    udndData = []
    for index in cv_diff_rate.index.values:
        udndData.append((cv_diff_rate[index], cv_maN_rate[index], str(udNd[index])))

    udndData = [([diffrate, maNrate], UDND) for diffrate, maNrate, UDND in udndData]

    for (diffrate, maNrate), UDND in udndData:
        plots[UDND][0].append(diffrate)
        plots[UDND][1].append(maNrate)

    # create a scatter series for each udnd
    for udnd, (x, y) in plots.items():
        plt.scatter(x, y, color=colors[udnd], marker=markers[udnd],
                    label=udnd, zorder=10)

    plot_state_borders(plt)  # assume we have a function that does this

    plt.legend(loc=0)  # let matplotlib choose the location
    plt.title("ud_Nd")
    plt.axis([-12, 8, -5, 4])  # set the axes

    # X축이 cv_diff_rate
    # Y축이 cv_maN_rate
    plt.show()

    return udndData


def knn_classify(k, labeled_points, new_point):
    """매개변수설명
        k : 어느정도 가까운 것들을 찾는가
        labeled_points : 분류에 사용 될 데이터목록들
        new_point : 분류하고 싶은 데이터
                    1. 분류에 사용될 데이터들을 분류 될 데이터와 거리 순으로 정렬한다.
                    2. 정렬된 데이터 중에서 k 거리 이내에 있는 데이터 목록만 따로 majority_vote에 넘겨서
                    k 거리이내의 데이터들 중에 가장 많이 포함되 있는 라벨을 찾는다.
    """
    # each labeled point should be a pair (point, label)

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

def classify_and_plot_grid(k,data):
    plots = {"1": ([], []), "-1": ([], []), "0": ([], [])}

    # we want each values to have a different marker and color
    markers = {"1": "o", "-1": "s", "0": "^"}
    colors = {"1": "r", "-1": "b", "0": "g"}

    for longitude in np.arange(-12, 8, 0.25):
        for latitude in np.arange(-5, 4, 0.25):
            predicted_language = knn_classify(k, data, [longitude, latitude])
            plots[predicted_language][0].append(longitude)
            plots[predicted_language][1].append(latitude)


    # create a scatter series for each values
    for language, (x, y) in plots.items():
        plt.scatter(x, y, color=colors[language], marker=markers[language],
                    label=language, zorder=0)

    plot_state_borders(plt, color='black')  # assume we have a function that does this

    plt.legend(loc=0)  # let matplotlib choose the location
    plt.axis([-12, 8, -5, 4])  # set the axes
    plt.title(str(k) + "-Nearest Neighbor Programming Languages")
    plt.show()

def data_cook(data, x_value, y_value):
    x_value = data[x_value]
    y_value = data[y_value]
    udNd = data['ud_Nd']
    data_cook = []
    for index in x_value.index.values:
        data_cook.append((x_value[index] , y_value[index], str(udNd[index])))

    data_cook = [([x_value, y_value], UDND) for x_value, y_value, UDND in data_cook]


    return data_cook

'''
코드 작성 및 테스트 시 사용했던 메인

if __name__ == '__main__':
    input_data = pd.read_csv("stock_history_added.csv", sep=",", encoding='cp949')
    test_data = classify_data(input_data)[0]
    training_data = classify_data(input_data)[1]

    data = plot_udNd(input_data)
    # data_cook 파라미터 값 ( data, x_value, y_value)
    testdata = data_cook(test_data, "cv_diff_value", "cv_maN_value")
    trainingdata = data_cook(training_data, "cv_diff_value", "cv_maN_value")

    for k in range(1, 31, 2):
        num_correct = 0
        k_value_ = []
        column_name = "k_value_" + str(k)
        for rate, actual_udNd in testdata:
            # other_cities는 학습, location은 테스트
            predicted_udNd = knn_classify(k, trainingdata, rate)
            k_value_.append(predicted_udNd)
            if predicted_udNd == actual_udNd:
                num_correct += 1
        test_data[column_name] = k_value_
        print(k, "neighbor[s]:", num_correct, "correct out of", len(data), num_correct / len(data) * 100, "%")

    test_data.to_csv("stock_history_K.csv", mode='w', encoding='cp949')
    # 3. classfy and plot grid with k = 1, 3, 5
    # classify_and_plot_grid(3)
    # 첫번째 인자는 k 값
    # classify_and_plot_grid(3, data)
'''