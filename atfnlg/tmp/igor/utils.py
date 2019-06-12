import numpy as np
import tensorflow as tf
import nltk
from nltk.corpus import stopwords
import string


nltk.download('stopwords')


def rand_gen(batch_size, seq_len):
    return np.random.randint(0, high=seq_len, size=(batch_size, seq_len))
    mat = np.random.normal(0.0, 1.0, (batch_size, seq_len))
    # print(mat)
    mat = (mat * 100)
    # print(mat)
    mat = mat.astype(int)
    # print(mat)
    mat = np.remainder(mat, seq_len)
    return mat


def batch_gen(batch_size, data):
    data_len = data.shape[0]
    # shuffle = np.random.permutation(data_len)
    start = 0
    #     from IPython.core.debugger import Tracer; Tracer()()
    # data = data[shuffle]
    while start + batch_size <= data_len:
        yield data[start:start + batch_size]
        start += batch_size


def tokenize(file_text):
    # firstly let's apply nltk tokenization
    tokens = nltk.word_tokenize(file_text)

    # let's delete punctuation symbols
    tokens = [i for i in tokens if (i not in string.punctuation)]

    # deleting stop_words
    stop_words = stopwords.words('english')
    # stop_words.extend(['что', 'это', 'так', 'вот', 'быть', 'как', 'в', '—', 'к', 'на'])
    tokens = [i for i in tokens if (i not in stop_words)]

    # cleaning words
    tokens = [i.replace("«", "").replace("»", "") for i in tokens]

    return tokens


def data_preparing(params):
    print("Prepare data...")
    f = open(params["file"])
    data = f.read().lower().replace("\n", " ")
    words = tokenize(data)
    # create vocabluary
    u_characters = set(" ".join(words))
    len_vocab = len(u_characters)
    print("Length vocab: ", len_vocab)
    # create char to num and num to char lookup tables
    char2num = dict(zip(u_characters, range(len(u_characters))))
    num2char = dict(zip(range(len(u_characters)), u_characters))
    char2num["<PAD>"] = len(u_characters)
    num2char[len(u_characters)] = "<PAD>"
    # split text by words

    # seq_len = max([len(list(x)) for x in words])
    print("Sequence len: ", params["seq_len"])
    # padding sequences
    words = map(lambda x: ["<PAD>"] * (params['seq_len'] - len(x)) + list(x), words)
    # transform chars to nums
    seq_data = [[char2num[ch] for ch in wd] for wd in words]
    return np.array(seq_data), num2char


def text2tensor(string):
    return tf.convert_to_tensor(string, dtype=tf.string)


def create_gradients_board(grads):
    for i in grads:
        var = i[1]
        if isinstance(var, tf.Variable):
            tf.summary.histogram(var.name, var)

'''
f = open('./nitshe.txt')
data = f.read().lower().replace("\n", " ")

# create vocabluary
u_characters = set(data)
len_vocab = len(u_characters)
print("Length vocab: ", len_vocab)
# create char to num and num to char lookup tables
char2num = dict(zip(u_characters, range(len(u_characters))))
num2char = dict(zip(range(len(u_characters)), u_characters))
char2num["<PAD>"] = len(u_characters)
num2char[len(u_characters)] = "<PAD>"
# split text by words
words = data.split(' ')
max_seq_len = max([len(list(x)) for x in words])
# padding sequences
words = map(lambda x: ["<PAD>"]*(max_seq_len-len(x)) + list(x), words)
# transform chars to nums
seq_data = [[char2num[ch] for ch in wd] for wd in words]

print(len(seq_data[0]))

sess = tf.InteractiveSession()
print("seq_data: ", seq_data[50])
trans = tf.one_hot(seq_data, len(char2num), 1.0, 0.0, axis=2, dtype=tf.float32)
np.set_printoptions(threshold=np.nan)
print(trans.eval()[50, :, :])

seq_data = np.array(seq_data)
# for i in batch_gen(seq_data, 128):
#    print(i.shape)
'''


'''
print("Prepare data...")
f = open(self.params["file"])
data = f.read().lower().replace("\n", " ")
words = tokenize(data)
# create vocabluary
u_characters = set(" ".join(words))
len_vocab = len(u_characters)
print("Length vocab: ", len_vocab)
# create char to num and num to char lookup tables
char2num = dict(zip(u_characters, range(len(u_characters))))
num2char = dict(zip(range(len(u_characters)), u_characters))
char2num["<PAD>"] = len(u_characters)
num2char[len(u_characters)] = "<PAD>"
# split text by words

# seq_len = max([len(list(x)) for x in words])
print("Sequence len: ", self.params["seq_len"])
# padding sequences
words = map(lambda x: ["<PAD>"] * (self.params['seq_len'] - len(x)) + list(x), words)
# transform chars to nums
seq_data = [[char2num[ch] for ch in wd] for wd in words]
seq_data = np.array(seq_data)
'''