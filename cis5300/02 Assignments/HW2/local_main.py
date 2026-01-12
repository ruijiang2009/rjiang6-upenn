import gzip
from collections import defaultdict
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt


train_file = "complex_words_training.txt" # 4000 words
dev_file = "complex_words_development.txt" # 1000 words
test_file = "complex_words_test_unlabeled.txt" # 922 words
mini_test_file = 'complex_words_test_mini.txt' # 50 words

def load_ngram_counts(ngram_counts_file = 'ngram_counts.txt.gz'):
    counts = defaultdict(int)
    with gzip.open(ngram_counts_file, 'rt') as f:
        for line in f:
            token, count = line.strip().split('\t')
            if token[0].islower():
                counts[token] = int(count)
    return counts

def load_labeled_file(data_file):
    words = []
    labels = []
    with open(data_file, 'rt', encoding="utf8") as f:
        i = 0
        for line in f:
            if i > 0:
                line_split = line[:-1].split("\t")
                words.append(line_split[0].lower())
                labels.append(int(line_split[1]))
            i += 1
    return words, labels


def case_2_1():
    import matplotlib.pyplot as plt
    precision = [0.418, 0.418, 0.418, 0.4256619144602851, 0.4649321266968326, 0.5270805812417437, 0.6053511705685619, 0.6807095343680709, 0.7346278317152104, 0.7593582887700535, 0.7899159663865546, 0.7647058823529411]
    recall = [1.0, 1.0, 1.0, 1.0, 0.9832535885167464, 0.9545454545454546, 0.8660287081339713, 0.7344497607655502, 0.5430622009569378, 0.3397129186602871, 0.22488038277511962, 0.12440191387559808]
    labels = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]
    # for i in range(12):
    #     # plt.text(precision[i], recall[i], labels[i], fontsize=8)
    #     plt.annotate(labels[i], (precision[i], recall[i]))
    plt.scatter(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.show()


def case_2_3():
    import matplotlib.pyplot as plt
    precision = [0.35555555555555557, 0.35555555555555557, 0.35555555555555557, 0.35555555555555557, 0.35555555555555557, 0.35555555555555557, 0.35555555555555557, 0.34782608695652173, 0.34782608695652173, 0.34782608695652173, 0.34782608695652173, 0.34782608695652173, 0.34782608695652173, 0.34782608695652173, 0.34782608695652173]
    recall = [0.03827751196172249, 0.03827751196172249, 0.03827751196172249, 0.03827751196172249, 0.03827751196172249, 0.03827751196172249, 0.03827751196172249, 0.03827751196172249, 0.03827751196172249, 0.03827751196172249, 0.03827751196172249, 0.03827751196172249, 0.03827751196172249, 0.03827751196172249, 0.03827751196172249]
    labels = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]
    # for i in range(12):
    #     # plt.text(precision[i], recall[i], labels[i], fontsize=8)
    #     plt.annotate(labels[i], (precision[i], recall[i]))
    plt.scatter(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.show()

if __name__ == '__main__':
    # ngram_counts = load_ngram_counts()
    # print(ngram_counts)
    # dev_data = load_labeled_file('Sample_Data/complex_words_development.txt')
    # print(dev_data)
    # case_2_1()
    case_2_3()