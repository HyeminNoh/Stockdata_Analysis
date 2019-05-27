import apply_knn as knn
import sorting_data as prepareData
import matplotlib.pyplot as plt
import numpy as np

def total_prepared_data():
    # 종목선택 데이터 자르기
    selected_data = prepareData.select_stock()
    # print(selected_data)
    # 데이터 준비, 변수추가
    prepared_data = prepareData.prepare_data(selected_data)  # 종가 일간 변화량, 변화율 추가
    # print(prepared_data)
    prepared_data = prepareData.cv_moveAverage_value(3, prepared_data)  # N_day 평균 병화량
    prepared_data = prepareData.cv_moveAverage_rate(3, prepared_data)
    prepared_data = prepareData.set_udNd(3, prepared_data)
    prepared_data = prepareData.cvNd_diff_rate(3, prepared_data)

    # 날짜 내림차순으로 재 정렬
    prepared_data = prepared_data.sort_values(["basic_date"], ascending=[False])
    result_data = prepared_data.reset_index(drop=True)

    result_data.to_csv("stock_history_added.csv", mode='w', encoding='cp949')
    return result_data

def k_change_plot(k, data):
    test_data = knn.classify_data(data)[0]
    training_data = knn.classify_data(data)[1]

    testdata = knn.data_cook(test_data, "cv_diff_rate", "cv_maN_rate")
    trainingdata = knn.data_cook(training_data, "cv_diff_rate", "cv_maN_rate")
    accuracy_plot = []
    k_plot = []
    for i in range(1, k+2, 2):
        num_correct = 0

        for rate, actual_udNd in testdata:
            predicted_udNd = knn.knn_classify(i, trainingdata, rate)
            if predicted_udNd == actual_udNd:
                num_correct += 1

        accuracy = num_correct / len(testdata) * 100
        k_plot.append(i)
        accuracy_plot.append(accuracy)

    max_accaracy = max(accuracy_plot)
    count = 0
    for index in accuracy_plot:

        if max_accaracy == index:
            break
        else:
            count = count+1
    max_index = 1+count*2
    plt.plot(k_plot, accuracy_plot, marker='o')
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.xticks(np.arange(1, k+2, 2))
    plt.show()
    return max_index

def make_result(k_num, input_data):
    test_data = knn.classify_data(input_data)[0]
    training_data = knn.classify_data(input_data)[1]

    # data_cook 파라미터 값 ( data, x_value, y_value)
    testdata = knn.data_cook(test_data, "cv_diff_rate", "cv_maN_rate")
    trainingdata = knn.data_cook(training_data, "cv_diff_rate", "cv_maN_rate")
    num_correct = 0
    k_value = []
    for rate, actual_udNd in testdata:
        predicted_udNd = knn.knn_classify(k_num, trainingdata, rate)
        k_value.append(predicted_udNd)
        if predicted_udNd == actual_udNd:
            num_correct += 1

    accuracy = num_correct / len(testdata) * 100
    print("K값 : ",k_num,"정확도: "+str(round(accuracy, 2))+"%")
    test_data["K"] = k_num
    test_data["k_udnd"] = k_value
    test_data.to_csv("stock_history_K.csv", mode='w', encoding='cp949')

    return test_data

if __name__ == '__main__':
    # 데이터 준비 1번 과정
    prepared_data = total_prepared_data()

    # K값에 따른 정확도 제시, 첫번째 파라미터 K값 범위 지정
    # 정확도가 가장 높았던 k값을 반환
    max_k = k_change_plot(21, prepared_data)

    # 가장 정확도가 높은 K값에 따른 예측값이 포함된 데이터 파일 생성
    predict_data = make_result(max_k, prepared_data)

    # test데이터의 기존 udNd 분포도, 예측 udNd 분포도
    knn.plot_udNd(knn.classify_data(prepared_data)[0], predict_data)

