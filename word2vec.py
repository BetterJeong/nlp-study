import tensorflow.compat.v1 as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
tf.disable_v2_behavior()

# 불용어 제거
def remove_stop_words(corpus):
    stop_words = ['is', 'a', 'will', 'be']
    results = []

    for text in corpus:
        tmp = text.split(' ')
        for stop_word in stop_words:
            if stop_word in tmp:
                tmp.remove(stop_word)
        results.append(" ".join(tmp))

    return results


# 데이터
corpus = ['king is a strong man',
          'queen is a wise woman',
          'boy is a young man',
          'girl is a young woman',
          'prince is a young king',
          'princess is a young queen',
          'man is strong',
          'woman is pretty',
          'prince is a boy will be king',
          'princess is a girl will be queen']
corpus = remove_stop_words(corpus)
words = []
word2int = {}
sentences = []
WINDOW_SIZE = 2
EMBEDDING_DIM = 2
data = []

# 단어 분리
for text in corpus:
    for word in text.split(' '):
        words.append(word)
words = set(words)

# 단어별 인덱스 매핑 테이블 만들기
for i, word in enumerate(words):
    word2int[word] = i

# 레이블 생성 (스킵 그램)
for sentence in corpus:
    sentences.append(sentence.split())

for sentence in sentences:
    for idx, word in enumerate(sentence):
        for neighbor in sentence[ \
                        max(idx - WINDOW_SIZE, 0): \
                                min(idx + WINDOW_SIZE, len(sentence)) + 1]:
            if neighbor != word:
                data.append([word, neighbor])

df = pd.DataFrame(data, columns=['input', 'label'])
df.head(10)

# 텐서플로로 word2vec 모델 구현
ONE_HOT_DIM = len(words)


# 숫자 -> 원 핫 인코딩
def to_one_hot_encoding(data_point_index):
    on_hot_encoding = np.zeros(ONE_HOT_DIM)
    on_hot_encoding[data_point_index] = 1
    return on_hot_encoding


X = []  # input
Y = []  # label

for x, y in zip(df['input'], df['label']):
    X.append(to_one_hot_encoding(word2int[x]))
    Y.append(to_one_hot_encoding(word2int[y]))

# 딥러닝 모델 입력값으로 전환
X_train = np.asarray(X)
Y_train = np.asarray(Y)

# 입력값 및 레이블을 받기 위한 placeholder 설정
x = tf.placeholder(tf.float32, shape=(None, ONE_HOT_DIM))
y_label = tf.placeholder(tf.float32, shape=(None, ONE_HOT_DIM))

# 히든 레이어
W1 = tf.Variable(tf.truncated_normal(
    [ONE_HOT_DIM, EMBEDDING_DIM], stddev=0.1))
hidden_layer = tf.matmul(x, W1)

# 출력 레이어
W2 = tf.Variable(tf.random_normal([EMBEDDING_DIM, ONE_HOT_DIM]))
b2 = tf.Variable(tf.random_normal([1]))
prediction = tf.nn.softmax(tf.add(tf.matmul(hidden_layer, W2), b2))

# 손실함수
loss = tf.reduce_mean(
    -tf.reduce_sum(y_label * tf.log(prediction), axis=[1]))

# 최적화
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

# 학습
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

iteration = 20000

for i in range(iteration):
    sess.run(train_op, feed_dict={x: X_train, y_label: Y_train})
    if i % 3000 == 0:
        print('iteration ' + str(i) + ' loss is : ', \
              sess.run(loss, feed_dict={x: X_train, y_label: Y_train}))

# 히든 레이어 값 저장
vectors = sess.run(W1)
print(vectors)

# 팬더스 데이터프레임에 word2vec 좌푯값 옮기기
w2v_df = pd.DataFrame(vectors, columns=['x1', 'x2'])
w2v_df['word'] = list(words)
w2v_df = w2v_df[['word', 'x1', 'x2']]

# 2차원 공간에 word2vec 임베딩 시각화
fig, ax = plt.subplots()
for word, x1, x2 in zip(w2v_df['word'], w2v_df['x1'], w2v_df['x2']):
    ax.annotate(word, (x1, x2))

PADDING = 1.0
x_axis_min = np.amin(vectors, axis=0)[0] - PADDING
y_axis_min = np.amin(vectors, axis=0)[1] - PADDING
x_axis_max = np.amax(vectors, axis=0)[0] + PADDING
y_axis_max = np.amax(vectors, axis=0)[1] + PADDING

plt.xlim(x_axis_min, x_axis_max)
plt.ylim(y_axis_min, y_axis_max)
plt.rcParams["figure.figsize"] = (10, 10)

plt.show()