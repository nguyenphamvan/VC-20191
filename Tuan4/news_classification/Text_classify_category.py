import settings
import os
import numpy as np
import pickle
import pandas as pd
from svm_deploy import *

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm, neighbors

def read_data_csv(file_path):
    content_file = pd.read_csv(file_path)
    X_data = []
    y_data = content_file['label'].values
    for line in content_file['title'].values:
                line = get_words_feature(line)
                line = ' '.join(line)
                X_data.append(line)
    return X_data, y_data


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
    tfidf_vect = TfidfVectorizer(analyzer='word', max_features=30000, min_df=0.005 , ngram_range=(1,2))

    X_data_tfidf =  tfidf_vect.fit_transform(X_train)
    X_test_tfidf = tfidf_vect.transform(X_test)

    return X_data_tfidf, X_test_tfidf



def train_model_with_sklearn(classifier, X_data, y_data, X_test, y_test, is_neuralnet=False, n_epochs=3):
    X_train, X_val, y_train, y_val = model_selection.train_test_split(X_data, y_data, test_size=0.2, random_state=42)

    classifier.fit(X_train, y_train)

    y_train_predictions = classifier.predict(X_train)
    y_val_predictions = classifier.predict(X_val)
    y_test_predictions = classifier.predict(X_test)

    # print("train accuracy: ", metrics.accuracy_score(y_train_predictions, y_train))
    # print("Validation accuracy: ", metrics.accuracy_score(y_val_predictions, y_val))
    # print("Test accuracy with sklearn : ", metrics.accuracy_score(y_test_predictions, y_test))
    print("classification_report \n: ", metrics.classification_report(y_test, y_test_predictions))
    return metrics.confusion_matrix(y_test, y_test_predictions)

def main():
    # X_data, y_data = read_data_csv('/home/nguyenpham/Desktop/card11_alpha_uni_short - card11_alpha_uni_short.csv')
    # y_data = [str(label) for label in y_data]
    #
    # X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=42)
    #
    # X_train_tfidf , X_test_tfidf= featureExtraction(X_train, X_test)
    # X_train_tfidf = X_train_tfidf.toarray()
    # X_test_tfidf = X_test_tfidf.toarray()

    # pickle.dump(X_train_tfidf, open('data_new/X_train.pkl', 'wb'))
    # pickle.dump(y_train, open('data_new/y_train.pkl', 'wb'))
    #
    # pickle.dump(X_test_tfidf, open('data_new/X_test.pkl', 'wb'))
    # pickle.dump(y_test, open('data_new/y_test.pkl', 'wb'))

    X_train_tfidf = pickle.load(open('data_new/X_train.pkl', 'rb'))
    y_train = pickle.load(open('data_new/y_train.pkl','rb'))

    X_test_tfidf = pickle.load(open('data_new/X_test.pkl', 'rb'))
    y_test = pickle.load(open('data_new/y_test.pkl', 'rb'))

    model = train_model_with_sklearn(svm.LinearSVC(penalty='l2',C=0.4 ,loss='squared_hinge', tol=0.001)
                                 , X_train_tfidf, y_train, X_test_tfidf, y_test, is_neuralnet=False)

    mini_confusion_matrix = dict()
    for i in range(model.shape[0]):
        row= list(model[i])
        dict_row = dict()
        for j in range(len(row)):
            if row[j] != 0 :
                dict_row[j] = row[j]
        mini_confusion_matrix[i] = dict_row
    for key, value in mini_confusion_matrix.items():
        print("%s : %s" %(key, value))
    print("\nnumber of labels for class: ")
    for key, value in mini_confusion_matrix.items():
        number_of_labels = sum(list(value.values()))
        print("class %s : %s" %(key, number_of_labels))



if __name__ == '__main__':
    main()

