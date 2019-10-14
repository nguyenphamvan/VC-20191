import settings
import os
import numpy as np
import gensim
import pickle
from svm  import  *

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm


def read_stopwords(filename):
    with open(filename,'r',encoding='utf-8') as f:
        stopwords = set([w.strip() for w in f.readlines()])
    return stopwords

def split_words(text):
    try:
        return [x.strip(settings.SPECIAL_CHARACTER).lower() for x in text.split()]
    except TypeError:
        return []

def get_words_feature(text):
    words = split_words(text)
    stopwords = read_stopwords(settings.STOP_WORDS)
    sentence = [word for word in words if word not in stopwords]
    # sentence = [word for word in words if word not in stopwords
    #             and word not in ['.',',','"',':','-','?','&','$','%','#','!','(',')','{','}','[',']','\n']]
    return sentence

def get_data_train(folder_path):
    X = []
    y = []
    file_paths = os.listdir(folder_path)
    # thu tu tra ve file khong tuan tu
    for file in file_paths:
        with open(os.path.join(folder_path, file),'r',encoding='utf8') as f:
            lines = f.readlines()
            for line in lines:
                line = get_words_feature(line)
                line = ' '.join(line)
                X.append(line)
                y.append(file.replace('.txt',''))
    return X,y

def get_data_test(folder_path):
    X = []
    y = []
    listFileName = os.listdir(folder_path)
    for file in listFileName:
        s = open(os.path.join(folder_path,file),'r',encoding='utf-8')
        if file == "data.txt":
            lines = s.readlines()
            for line in lines:
                line = get_words_feature(line)
                line = ' '.join(line)
                X.append(line)
        else:
            labels = s.readlines()
            for label in labels:
                y.append(label.replace('\n',''))
        s.close()
    return X,y

def featureExtraction(X_train, X_test):
    tfidf_vect = TfidfVectorizer(analyzer='word', max_features=30000, min_df=0.005, max_df=0.8)

    X_data_tfidf =  tfidf_vect.fit_transform(X_train)
    X_test_tfidf = tfidf_vect.transform(X_test)

    return X_data_tfidf, X_test_tfidf



def train_model_with_sklearn(classifier, X_data, y_data, X_test, y_test, is_neuralnet=False, n_epochs=3):
    X_train, X_val, y_train, y_val = model_selection.train_test_split(X_data, y_data, test_size=0.2, random_state=42)

    classifier.fit(X_train, y_train)

    train_predictions = classifier.predict(X_train)
    val_predictions = classifier.predict(X_val)
    test_predictions = classifier.predict(X_test)

    # print("train accuracy: ", metrics.accuracy_score(train_predictions, y_train))
    # print("Validation accuracy: ", metrics.accuracy_score(val_predictions, y_val))
    print("Test accuracy with sklearn : ", metrics.accuracy_score(test_predictions, y_test))




# X_train, y_train = get_data_train(os.path.join(settings.DATA_TRAIN_PATH))
# X_test, y_test= get_data_test(os.path.join(settings.DATA_TEST_PATH))
#
# X_train_tfidf , X_test_tfidf= featureExtraction(X_train, X_test)
# y_train = [int(x) for x in y_train]
# y_train = np.array(y_train)
#
# y_test = [int(x) for x in y_test]
# y_test = np.array(y_test)



# pickle.dump(X_train_tfidf, open('data/X_train.pkl', 'wb'))
# pickle.dump(y_train, open('data/y_train.pkl', 'wb'))
#
# pickle.dump(X_test_tfidf, open('data/X_test.pkl', 'wb'))
# pickle.dump(y_test, open('data/y_test.pkl', 'wb'))

X_train_tfidf = pickle.load(open('data/X_train.pkl', 'rb'))
y_train = pickle.load(open('data/y_train.pkl','rb'))

X_test_tfidf = pickle.load(open('data/X_test.pkl', 'rb'))
y_test = pickle.load(open('data/y_test.pkl', 'rb'))



N , C, d = X_train_tfidf.shape[0],len(set(y_train)), X_train_tfidf.shape[1]
reg = 0.1
W = np.random.randn(d, C)
X_train_new = X_train_tfidf.toarray()
X_test_new = X_test_tfidf.toarray()

print(X_train_new.shape)

# sub 1 , label start with label is 0
y_train = y_train - 1
y_test = y_test - 1

model_1 = model_with_svm_naive(X_train_new.T, y_train, X_test_new.T, y_test, W)
model_2 = model_svm_GD_vectorized(X_train_new.T, y_train, X_test_new.T, y_test, W)
model_3 = train_model_with_sklearn(svm.LinearSVC(), X_train_tfidf, y_train, X_test_tfidf, y_test, is_neuralnet=False)







