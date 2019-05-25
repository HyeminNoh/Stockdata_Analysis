import sorting_data
import apply_knn as knn

if __name__ == '__main__':
    # 종목선택 데이터 자르기
    selected_data = sorting_data.select_stock()
    # print(selected_data)
    # 데이터 준비, 변수추가
    prepared_data = sorting_data.prepare_data(selected_data)  # 종가 일간 변화량, 변화율 추가
    # print(prepared_data)
    prepared_data = sorting_data.cv_moveAverage_value(3, prepared_data)  # N_day 평균 병화량
    prepared_data = sorting_data.cv_moveAverage_rate(3, prepared_data)
    prepared_data = sorting_data.set_udNd(3, prepared_data)
    prepared_data = sorting_data.cvNd_diff_rate(3, prepared_data)

    # 날짜 내림차순으로 재 정렬
    prepared_data = prepared_data.sort_values(["basic_date"], ascending=[False])
    result_data = prepared_data.reset_index(drop=True)

    # stock_history_added파일 생성
    result_data.to_csv("stock_history_added.csv", mode='w', encoding='cp949')

    # input_data = pd.read_csv("stock_history_added.csv", sep=",", encoding='cp949')
    test_data = knn.classify_data(result_data)[0]
    training_data = knn.classify_data(result_data)[1]

    data = knn.plot_udNd(result_data)
    # data_cook 파라미터 값 ( data, x_value, y_value)
    testdata = knn.data_cook(test_data, "cv_diff_value", "cv_maN_value")
    trainingdata = knn.data_cook(training_data, "cv_diff_value", "cv_maN_value")

    for k in range(1, 31, 2):
        num_correct = 0
        k_value_ = []
        column_name = "k_value_" + str(k)
        for rate, actual_udNd in testdata:
            # other_cities는 학습, location은 테스트
            predicted_udNd = knn.knn_classify(k, trainingdata, rate)
            k_value_.append(predicted_udNd)
            if predicted_udNd == actual_udNd:
                num_correct += 1
        test_data[column_name] = k_value_
        print(k, "neighbor[s]:", num_correct, "correct out of", len(data), num_correct / len(data) * 100, "%")

    test_data.to_csv("stock_history_K.csv", mode='w', encoding='cp949')