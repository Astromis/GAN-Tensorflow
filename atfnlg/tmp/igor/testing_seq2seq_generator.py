from faker import Faker
import babel
from babel.dates import format_date
import numpy as np
import random
import time
import tensorflow as tf
from sklearn.model_selection import train_test_split
from atfnlg.tmp.igor.generator import Generator


def create_date():
    """
        Creates some fake dates
        :returns: tuple containing
                  1. human formatted string
                  2. machine formatted string
                  3. date object.
    """
    dt = fake.date_object()

    # wrapping this in a try catch because
    # the locale 'vo' and format 'full' will fail
    try:
        human = format_date(dt,
                            format=random.choice(FORMATS),
                            locale=random.choice(LOCALES))

        case_change = random.randint(0, 3)  # 1/2 chance of case change
        if case_change == 1:
            human = human.upper()
        elif case_change == 2:
            human = human.lower()

        machine = dt.isoformat()
    except AttributeError as e:
        return None, None, None

    return human, machine  # , dt


def batch_data(x, y, batch_size):
    shuffle = np.random.permutation(len(x))
    start = 0
    # from IPython.core.debugger import Tracer; Tracer()()
    x = x[shuffle]
    y = y[shuffle]
    while start + batch_size <= len(x):
        yield x[start:start+batch_size], y[start:start+batch_size]
        start += batch_size


fake = Faker()
fake.seed(42)
random.seed(42)

FORMATS = ['short',
           'medium',
           'long',
           'full',
           'd MMM YYY',
           'd MMMM YYY',
           'dd MMM YYY',
           'd MMM, YYY',
           'd MMMM, YYY',
           'dd, MMM YYY',
           'd MM YY',
           'd MMMM YYY',
           'MMMM d YYY',
           'MMMM d, YYY',
           'dd.MM.YY',
           ]

# change this if you want it to work with only a single language
LOCALES = babel.localedata.locale_identifiers()
LOCALES = [lang for lang in LOCALES if 'en' in str(lang)]

data = [create_date() for _ in range(50000)]

# print(data[:5])

x = [x_ for x_, y_ in data]
y = [y_ for x_, y_ in data]

u_characters = set(' '.join(x))
char2numX = dict(zip(u_characters, range(len(u_characters))))

u_characters = set(' '.join(y))
char2numY = dict(zip(u_characters, range(len(u_characters))))

char2numX['<PAD>'] = len(char2numX)
num2charX = dict(zip(char2numX.values(), char2numX.keys()))
max_len = max([len(date) for date in x])

x = [[char2numX['<PAD>']]*(max_len - len(date)) +[char2numX[x_] for x_ in date] for date in x]
print(''.join([num2charX[x_] for x_ in x[4]]))
x = np.array(x)

char2numY['<GO>'] = len(char2numY)
num2charY = dict(zip(char2numY.values(), char2numY.keys()))

y = [[char2numY['<GO>']] + [char2numY[y_] for y_ in date] for date in y]
print(''.join([num2charY[y_] for y_ in y[4]]))
y = np.array(y)

x_seq_length = len(x[0])
y_seq_length = len(y[0]) - 1

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
##############################


epochs = 5
sess = tf.InteractiveSession()
batch_size = 128

hparams = {'i_voc': len(char2numX), 'o_voc': len(char2numY), 'x_seq_len': x_seq_length, 'y_seq_len': y_seq_length,
           'batch_size': batch_size, 'num_hidden': 32, "opt": "rmsprop", 'learning_rate': 0.1, 'cell_type': 'lstm'}

model = Generator(sess, hparams)

sess.run(tf.global_variables_initializer())
for epoch_i in range(epochs):
    start_time = time.time()
    for batch_i, (source_batch, target_batch) in enumerate(batch_data(X_train, y_train, batch_size)):
        batch_loss, batch_logits = model.train(source_batch, target_batch)
    accuracy = np.mean(batch_logits.argmax(axis=-1) == target_batch[: , 1:])
    print('Epoch {:3} Loss: {:>6.3f} Accuracy: {:>6.4f} Epoch duration: {:>6.3f}s'.format(epoch_i, batch_loss,
                                                                     accuracy, time.time() - start_time))
    print("Sampling sequence: ", model.sampling(source_batch, y_seq_length, num2charY, char2numY))

