import re
import math
from collections import Counter, defaultdict


# ЧАСТЬ I — ПРЕДОБРАБОТКА ТЕКСТА


text = """The news mentioned here is fake. Audience do not encourage fake news.
Fake news is false or misleading"""

stop_words = {"the","is","do","not","or","here"}

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    sentences = [s.strip() for s in text.split('\n') if s.strip()]

    tokenized = []
    for sentence in sentences:
        words = sentence.split()
        words = [w for w in words if w not in stop_words]
        tokenized.append(words)

    return tokenized

tokenized_sentences = preprocess_text(text)
print("\nTokenized sentences:")
print(tokenized_sentences)


# СЛОВАРЬ И ЧАСТОТЫ


vocab = sorted(list(set(word for sent in tokenized_sentences for word in sent)))
print("\nVocabulary:")
print(vocab)

freq = Counter(word for sent in tokenized_sentences for word in sent)
freq_sorted = dict(sorted(freq.items(), key=lambda x: x[1], reverse=True))
print("\nWord frequencies:")
print(freq_sorted)

# BINARY BAG OF WORDS


def binary_bow(sentence, vocab):
    return [1 if word in sentence else 0 for word in vocab]

print("\nBinary BoW:")
for sent in tokenized_sentences:
    print(binary_bow(sent, vocab))


# COUNT BAG OF WORDS


def count_bow(sentence, vocab):
    return [sentence.count(word) for word in vocab]

print("\nCount BoW:")
for sent in tokenized_sentences:
    print(count_bow(sent, vocab))


# TF-IDF


N = len(tokenized_sentences)

df = {}
for word in vocab:
    df[word] = sum(1 for sent in tokenized_sentences if word in sent)

def tfidf(sentence):
    vec = []
    for word in vocab:
        tf = sentence.count(word) / len(sentence)
        idf = math.log(N / df[word])
        vec.append(tf * idf)
    return vec

print("\nTF-IDF:")
for sent in tokenized_sentences:
    print(tfidf(sent))


# ЧАСТЬ II — NAIVE BAYES


class NaiveBayes:
    def __init__(self):
        self.class_probs = {}
        self.word_probs = {}
        self.vocab = set()

    def train(self, data):
        total_docs = len(data)
        class_counts = Counter(label for label, _ in data)

        # P(class)
        for c in class_counts:
            self.class_probs[c] = math.log(class_counts[c] / total_docs)

        word_counts = {}
        total_words = {}

        for label, doc in data:
            word_counts.setdefault(label, Counter())
            total_words.setdefault(label, 0)

            for word, count in doc.items():
                word_counts[label][word] += count
                total_words[label] += count
                self.vocab.add(word)

        # P(word | class)
        for c in word_counts:
            self.word_probs[c] = {}
            for word in self.vocab:
                count = word_counts[c][word] + 1  # Laplace smoothing
                self.word_probs[c][word] = math.log(
                    count / (total_words[c] + len(self.vocab))
                )

    def predict(self, doc):
        scores = {}

        for c in self.class_probs:
            score = self.class_probs[c]
            for word, count in doc.items():
                if word in self.vocab:
                    score += count * self.word_probs[c][word]
            scores[c] = score

        return max(scores, key=scores.get)


# ПРИМЕР ОБУЧЕНИЯ NAIVE BAYES (демо датасет)


train_data = [
    ("sport", {"team":2,"win":1,"game":1}),
    ("sport", {"team":1,"score":1,"win":1}),
    ("tech", {"computer":2,"code":1}),
    ("tech", {"python":1,"code":2})
]

nb = NaiveBayes()
nb.train(train_data)

test_doc = {"team":1,"win":1}
print("\nNaive Bayes prediction for test_doc:", nb.predict(test_doc))


# ДОП ЗАДАНИЕ — REGEX


regex_letters = r'^[A-Za-z]+$'
regex_end_b = r'^[a-z]*b$'
regex_ab = r'^(b|bab)+$'

print("\nRegex:")
print("Only letters:", regex_letters)
print("Lowercase ending with b:", regex_end_b)
print("Each 'a' surrounded by 'b':", regex_ab)