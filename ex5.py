import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import cv2

numbers = []

numbers.append(cv2.imread('zero.png', cv2.IMREAD_GRAYSCALE))
numbers.append(cv2.imread('one.png', cv2.IMREAD_GRAYSCALE))
numbers.append(cv2.imread('two.png', cv2.IMREAD_GRAYSCALE))
numbers.append(cv2.imread('three.png', cv2.IMREAD_GRAYSCALE))
numbers.append(cv2.imread('four.png', cv2.IMREAD_GRAYSCALE))
numbers.append(cv2.imread('five.png', cv2.IMREAD_GRAYSCALE))
numbers.append(cv2.imread('six.png', cv2.IMREAD_GRAYSCALE))
numbers.append(cv2.imread('seven.png', cv2.IMREAD_GRAYSCALE))
numbers.append(cv2.imread('eight.png', cv2.IMREAD_GRAYSCALE))
numbers.append(cv2.imread('nine.png', cv2.IMREAD_GRAYSCALE))

for i in range(10):
  # numbers[i]を(8, 8)にリサイズ
  numbers[i] = cv2.resize(numbers[i], (8, 8))

  # numbers[i]を色反転（digits.csvは白と黒の値が逆のため）
  numbers[i] = cv2.bitwise_not(numbers[i])

  # numbers[i]を1次元配列に
  numbers[i] = numbers[i].reshape(numbers[i].size)

  # numbers[i]を0〜16あたりの範囲にスケーリング
  numbers[i] = numbers[i] // 16
  
df = pd.DataFrame(numbers)
df['number'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

X = df.drop("number", axis=1)
y = df["number"]

model = KNeighborsClassifier(n_neighbors=1)
model.fit(X, y)

df_answer = pd.read_csv("digits.csv")
X_ans = df_answer.drop("number", axis=1)
y_ans = df_answer["number"]

y_pred = model.predict(X_ans)

accuracy = accuracy_score(y_ans, y_pred)
print(accuracy)