import nltk
from pyvi import ViTokenizer

file = open("vietnamese-stopwords.txt","r",encoding='utf-8')
file1 = open("document.txt",'r',encoding='utf-8')
stop_word = file.read()
file.close()
def preprocessing_data(doc):
    doc_parsed = doc.lower()
    doc_parsed = doc_parsed.replace('\n','. ')
    doc_parsed = doc_parsed.strip()

    # sentences tokenize
    sentences = nltk.sent_tokenize(doc_parsed)
    print(sentences)

    # words segmentation
    total_words = 0
    words_not_in_stopword = []
    for sentence in sentences:
        sentence_tokenized = ViTokenizer.tokenize(sentence)
        words = sentence_tokenized.split(" ")
        total_words += len(words)
        for word in words:
            if word not in stop_word and word not in ['.',',','"',':','?','!','(',')','{','}','[',']']:
                words_not_in_stopword.append(word)

    return words_not_in_stopword

def calculate_frequency(words):
    # create dict contains frequency of words with key  isfrequency and value is
    frequency_of_words = {}
    word_array = []
    frequency = []
    dictionary = set(words)
    for word in dictionary:
        frequency_of_words[word] = words.count(word)/len(words)
        word_array.append(word)
        frequency.append(words.count(word)/len(words))
    max_frequency = max(frequency)

    return frequency_of_words, word_array[frequency.index(max_frequency)] , max_frequency

def main():
    file = open('document.txt','r',encoding='utf-8')
    doc = file.read()
    print("Câu đã tách :")
    words_segmentation = preprocessing_data(doc)
    print()

    print("Từ đã tách từ và loại bỏ stop words : ")
    print(words_segmentation)
    print()

    frequency_words , word_max_frequency , max_frequency = calculate_frequency(words_segmentation)

    print("Tần suất của các từ (Đã loại bỏ stop word) :")
    print(frequency_words)
    print()

    print("Từ xuất hiện nhiều nhất là  {} : {}".format(word_max_frequency,max_frequency))


if __name__ == '__main__':
    main()
