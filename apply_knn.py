import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from linear_algebra import distance
from stats import mean
import math, random

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

if __name__ == '__main__':
    input_data = pd.read_csv("stock_history_added.csv", sep=",", encoding='cp949')
    test_data = classify_data(input_data)[0]
    training_data = classify_data(input_data)[1]
    #print(test_data)
    #print(training_data)