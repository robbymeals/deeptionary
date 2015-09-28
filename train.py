import random
import spacy
import numpy as np
import time
import sys
from spacy.en import English
from string import lowercase
from collections import defaultdict, Counter
from itertools import chain
from nltk.corpus import wordnet as wn
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report 
from keras.layers.core import RepeatVector
from keras.layers.recurrent import JZS1, JZS2, JZS3
nlp = English()


def iter_sample_fast(iterable, samplesize):
    results = []
    iterator = iter(iterable)
    # Fill in the first samplesize elements:
    try:
        for _ in xrange(samplesize):
            results.append(iterator.next())
    except StopIteration:
        raise ValueError("Sample larger than population.")
    random.shuffle(results)  # Randomize their positions
    for i, v in enumerate(iterator, samplesize):
        r = random.randint(0, i)
        if r < samplesize:
            results[r] = v  # at a decreasing rate, replace random items
    return results


#stops = [
#        'of', 'or', 'for', 'a', 'the', 'as', 'but', 'to',
#        'am', 'an', 'are', 'in', 'with', 'and', 'that', 'is',
#        ]
stops = []
def preprocess_synset(ss, char_ref, pos_ref, char_def_ref):
    if any([c not in lowercase for c in ss.name().split('.')[0]]):
        return "Invalid"
    d = [
            [char_ref[l] for l in list(ss.name().split('.')[0])],
            pos_ref[ss.pos()], 
            [t.orth_.lower() for t in nlp(ss.definition())]
        ]
    d = d + [[w for w in d[2] if w not in stops],]
    d = d + [[char_def_ref[l] for l in list(ss.definition())],]
    return d

def preprocess_defin(d, word_ref):
    d[3] = [word_ref.get(w) for w in d[3]]
    d[3] = [w for w in d[3] if w is not None]
    return d


def label_to_multiclass(label, n_labels):
    y = np.zeros(n_labels)
    y[label] = 1
    return y


## pad word-character and definition-word vectors
def rm_invalid_and_zero_pad(seq, max_len, lpad=True):
    seq = [w for w in seq if w != None]
    if len(seq) >= max_len:
        return seq[0:max_len]
    else:
        if lpad:
            return np.array([0 for _ in range(max_len-len(seq))] + seq)
        else:
            return np.array(seq + [0 for _ in range(max_len-len(seq))])

        
char_ref = defaultdict(lambda: char_ref.__len__() + 1,
        {l:i+1 for i, l in enumerate(list(unicode(lowercase)))})
pos_ref = defaultdict(lambda: pos_ref.__len__())
char_def_ref = defaultdict(lambda: char_def_ref.__len__() + 1)

sample = int(sys.argv[1])

if sample > 0:
    defs = iter_sample_fast(wn.all_synsets('n'), sample)
else:
    defs = wn.all_synsets()

defs = [preprocess_synset(ss, char_ref, pos_ref, char_def_ref) for ss in defs]
defs = [d for d in defs if d != 'Invalid']
print(len(defs))
word_count = Counter(chain(*[d[3] for d in defs]))
word_ref = {w1:i+1 for i, w1 in enumerate((w2 for w2, c in word_count.items()
    if all([c > 1, w2 not in stops])))}
defs = [preprocess_defin(d, word_ref) for d in defs]


## generate feature index to feature mappings
rev_word_ref = {v:k for k,v in word_ref.items()}
rev_char_ref = {v:k for k,v in char_ref.items()}
rev_char_def_ref = {v:k for k,v in char_def_ref.items()}
rev_pos_ref = {v:k for k,v in pos_ref.items()}
char_ref.default_factory = None
char_def_ref.default_factory = None
pos_ref.default_factory = None
word_ref['__PADDING__'] = 0
char_ref['__PADDING__'] = 0
char_def_ref['__PADDING__'] = 0

print(''.join([rev_char_ref[x] for x in defs[0][0] if x != 0]))
print(' '.join([rev_word_ref[x] for x in defs[0][3] if x != 0]))
print(''.join([rev_char_def_ref[x] for x in defs[0][4] if x != 0]))

max_word_len = max([len(x[0]) for x in defs])
max_def_len = max([len(x[3]) for x in defs])
max_def_len_char = max([len(x[4]) for x in defs])
for d in defs:
    d[0] = rm_invalid_and_zero_pad(d[0], max_word_len, lpad=False)
    d[3] = rm_invalid_and_zero_pad(d[3], max_def_len)
    d[4] = rm_invalid_and_zero_pad(d[4], max_def_len_char)

print(''.join([rev_char_ref[x] for x in defs[0][0] if x != 0]))                                                                                                       
print(' '.join([rev_word_ref[x] for x in defs[0][3] if x != 0])) 
print(''.join([rev_char_def_ref[x] for x in defs[0][4] if x != 0]))

random.shuffle(defs)
n_defs = len(defs)
train = defs[0:int(0.9*n_defs)]
test = defs[int(0.9*n_defs):]
X_train = np.vstack([x[3] for x in train])
X_test = np.vstack([x[3] for x in test])
X_train_char = np.vstack([x[4] for x in train])
X_test_char = np.vstack([x[4] for x in test])

if False:
    val = test[0:(len(test)/2)]
    test = test[(len(test)/2):]
    HIDDEN = 512
    ## model 2: multi-class part-of-speech classification based on definition
    y2_train = np.array([label_to_multiclass(x[1], len(pos_ref)) for x in train])
    y2_val = np.array([label_to_multiclass(x[1], len(pos_ref)) for x in val])
    y2_test = np.array([label_to_multiclass(x[1], len(pos_ref)) for x in test])
    checkpointer2 = ModelCheckpoint(filepath="/tmp/weights2.hdf5", verbose=1, save_best_only=True)
    early_stop2 = EarlyStopping(monitor='val_loss', patience=2, verbose=0)
    model2 = Sequential()
    model2.add(Embedding(len(word_ref), HIDDEN, mask_zero=True))
    model2.add(LSTM(HIDDEN, HIDDEN, activation='tanh', inner_activation='hard_sigmoid'))
    model2.add(Dropout(0.5))
    model2.add(Dense(HIDDEN, len(pos_ref)))
    model2.add(Activation('sigmoid'))
    model2.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    model2.fit(X_train, y2_train, batch_size=32, nb_epoch=2, verbose=1,
            validation_data=(X_val, y2_val), callbacks=[checkpointer2, early_stop2])
    model2.fit(X_train, y2_train, batch_size=64, nb_epoch=5, verbose=1,
            validation_data=(X_val, y2_val), callbacks=[checkpointer2, early_stop2])
    model2.fit(X_train, y2_train, batch_size=128, nb_epoch=5, verbose=1,
            validation_data=(X_val, y2_val), callbacks=[checkpointer2, early_stop2])
    model2.fit(X_train, y2_train, batch_size=256, nb_epoch=5, verbose=1,
            validation_data=(X_val, y2_val), callbacks=[checkpointer2, early_stop2])
    model2.fit(X_train, y2_train, batch_size=512, nb_epoch=5, verbose=1,
            validation_data=(X_val, y2_val), callbacks=[checkpointer2, early_stop2])

    ## final score
    score = model2.evaluate(X_test, y2_test, batch_size=2048)
    print(score)
    preds = model2.predict(X_test, batch_size=2048)
    print(classification_report(
        [rev_pos_ref[ya] for ya in y2_test.argmax(1)], 
        [rev_pos_ref[yp] for yp in preds.argmax(1)], 
        ))
    ## best val score
    model2.load_weights('/tmp/weights2.hdf5')
    score = model2.evaluate(X_test, y2_test, batch_size=2048)
    print(score)
    preds = model2.predict(X_test, batch_size=2048)
    print(classification_report(
        [rev_pos_ref[ya] for ya in y2_test.argmax(1)], 
        [rev_pos_ref[yp] for yp in preds.argmax(1)], 
        ))


## model 3: predict character vector
y3_train = np.array([np.array([label_to_multiclass(y, len(char_ref))
    for y in x[0]]) for x in train])
y3_test = np.array([np.array([label_to_multiclass(y, len(char_ref))
    for y in x[0]]) for x in test])

#The Magictionary
#The Imaginictionary 
#The Deeptionary
#The Dranktionary

if True:
    HIDDEN_LAYER = 512
    model3 = Sequential()
    model3.add(Embedding(len(word_ref), HIDDEN_LAYER, mask_zero=True))
    model3.add(JZS3(HIDDEN_LAYER, HIDDEN_LAYER, activation='tanh'))
    model3.add(Dropout(0.1))
    model3.add(RepeatVector(max_word_len))
    model3.add(JZS3(HIDDEN_LAYER, HIDDEN_LAYER, activation='tanh', return_sequences=True))
    model3.add(Dropout(0.1))
    model3.add(JZS3(HIDDEN_LAYER, HIDDEN_LAYER, activation='tanh', return_sequences=True))
    model3.add(Dropout(0.1))
    model3.add(JZS3(HIDDEN_LAYER, HIDDEN_LAYER, activation='tanh', return_sequences=True))
    model3.add(Dropout(0.1))
    model3.add(TimeDistributedDense(HIDDEN_LAYER, len(char_ref)))
    model3.add(Activation('softmax'))
    model3.compile(loss='categorical_crossentropy', optimizer='adam')

    preds = []
    checkpointer3 = ModelCheckpoint(filepath="/tmp/weights3.hdf5", verbose=1, save_best_only=True)
    early_stop3 = EarlyStopping(monitor='val_loss', patience=2, verbose=0)

    iteration = 0 
    BATCH_SIZE = 32 
    for _ in range(40):
        iteration = iteration + 1
        print('-- iteration {} --'.format(iteration))
        model3.fit(X_train, y3_train, batch_size=BATCH_SIZE, nb_epoch=1, verbose=1)
        model3.save_weights('model3p3_iteration_{}.hdf5'.format(iteration), overwrite=True)
        preds = preds + [model3.predict(X_test, batch_size=BATCH_SIZE),]
        for j in [0, X_test.shape[0]/2, X_test.shape[0]-1]:
            for x in range(max(0, len(preds)-5), len(preds)):
                print(
                        ''.join([rev_char_ref[i] for i in preds[x][j].argmax(1) if i != 0]), 
                        ' '.join([rev_word_ref[i] for i in X_test[j] if i != 0])
                        )


if False:
    HIDDEN_LAYER = 512
    model4 = Sequential()
    model4.add(TimeDistributedDense(len(char_ref), HIDDEN_LAYER))
    model4.add(JZS2(HIDDEN_LAYER, HIDDEN_LAYER, activation='tanh'))
    model4.add(Dropout(0.2))
    model4.add(RepeatVector(max_word_len))
    model4.add(JZS2(HIDDEN_LAYER, HIDDEN_LAYER, activation='tanh', return_sequences=True))
    model4.add(Dropout(0.2))
    model4.add(JZS2(HIDDEN_LAYER, HIDDEN_LAYER, activation='tanh', return_sequences=True))
    model4.add(Dropout(0.2))
    model4.add(TimeDistributedDense(HIDDEN_LAYER, len(char_ref)))
    model4.add(Activation('softmax'))
    model4.compile(loss='categorical_crossentropy', optimizer='adam')

    preds = []
    checkpointer4 = ModelCheckpoint(filepath="/tmp/weights4.hdf5", verbose=1, save_best_only=True)
    early_stop4 = EarlyStopping(monitor='val_loss', patience=2, verbose=0)

    iteration = 0
    BATCH_SIZE = 64 
    for _ in range(40):
        iteration = iteration + 1
        print('-- iteration {} --'.format(iteration))
        model4.fit(X_train_char, y3_train, batch_size=BATCH_SIZE, nb_epoch=1, verbose=1)
        model4.save_weights('model4_iteration_{}.hdf5'.format(iteration))
        preds = preds + [model4.predict(X_test_char, batch_size=BATCH_SIZE),]
        for j in [0, X_test_char.shape[0]/2, X_test_char.shape[0]-1]:
            for x in range(max(0, len(preds)-5), len(preds)):
                print(
                    ''.join([rev_char_ref[i] for i in preds[x][j].argmax(1) if i != 0]),
                    ''.join([rev_char_def_ref[i] for i in X_test_char[j] if i != 0])
                    )

