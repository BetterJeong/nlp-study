import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import urllib.request
from konlpy.tag import Okt
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def below_threshold_len(max_len, nested_list):
    count = 0
    for s in nested_list:
        if len(s) <= max_len:
            count = count + 1
    print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s' % (max_len, (count / len(nested_list)) * 100))


# --------------------- 데이터 로드 ---------------------
print("데이터 로딩중...")
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt",
                           filename="ratings_train.txt")
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt",
                           filename="ratings_test.txt")
print("데이터 로드 완료\n")

train_data = pd.read_table('ratings_train.txt')
test_data = pd.read_table('ratings_test.txt')
print("훈련용 리뷰 개수:", len(train_data))
print("테스트용 리뷰 개수:", len(test_data), "\n")

# --------------------- 전처리 ---------------------
# 한글, 공백 제거
train_data.drop_duplicates(subset=['document'], inplace=True)
train_data['document'] = train_data['document'].str.replace('[^ㄱ-ㅎㅏ-ㅣ가-힣 ]', "", regex=True)
train_data['document'] = train_data['document'].str.replace('^ +', "", regex=True)
train_data['document'].replace('', np.nan, inplace=True)
train_data = train_data.dropna(how='any')

test_data.drop_duplicates(subset=['document'], inplace=True)
test_data['document'] = test_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "", regex=True)
test_data['document'] = test_data['document'].str.replace('^ +', "", regex=True)
test_data['document'].replace('', np.nan, inplace=True)
test_data = test_data.dropna(how='any')

print("전처리 후 훈련용 샘플의 수:", len(train_data))
print('전처리 후 테스트용 샘플의 개수:', len(test_data))

# --------------------- 토큰화 ---------------------
stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']
okt = Okt()

X_train = []
for sentence in tqdm(train_data['document']):
    tokenized_sentence = okt.morphs(sentence, stem=True)
    stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords]
    X_train.append(stopwords_removed_sentence)

X_test = []
for sentence in tqdm(test_data['document']):
    tokenized_sentence = okt.morphs(sentence, stem=True)
    stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords]
    X_test.append(stopwords_removed_sentence)

# --------------------- 정수 인코딩 ---------------------
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
print(tokenizer.word_index, "\n")

# 등장 빈도 수가 3 미만인 단어 카운팅
threshold = 3
total_cnt = len(tokenizer.word_index)
rare_cnt = 0
total_freq = 0
rare_freq = 0

for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value

    if value < threshold:
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

print("단어 집합(vocabulary)의 크기:", total_cnt)
print("등장 빈도가 %s번 이하인 희귀 단어의 수:%s" % (threshold - 1, rare_cnt))
print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt) * 100)
print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq) * 100)

# 등장 빈도 수가 3 미만인 단어 제외한 단어 수를 단어 집합의 최대 크기로 제한
vocab_size = total_cnt - rare_cnt + 1
print("단어 집합의 크기:", vocab_size)

# 텍스트 시퀀스 -> 정수 시퀀스
tokenizer = Tokenizer(vocab_size)
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
print(X_train[:3])

y_train = np.array(train_data['label'])
y_test = np.array(test_data['label'])

# --------------------- 빈 샘플 제거 ---------------------
drop_train = [index for index, sentence in enumerate(X_train) if len(sentence) < 1]
X_train = np.delete(X_train, drop_train, axis=0)
y_train = np.delete(y_train, drop_train, axis=0)
print(len(X_train))
print(len(y_train))

# --------------------- 패딩 ---------------------
print("리뷰의 최대 길이:", max(len(review) for review in X_train))
print("리뷰의 평균 길이:", sum(map(len, X_train)) / len(X_train))
plt.hist([len(review) for review in X_train], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()

max_len = 30
below_threshold_len(max_len, X_train)

X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)
