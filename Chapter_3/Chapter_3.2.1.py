sent1 = 'The cat is walking in the bedroom.'
sent2 = 'A dog was running across the kitchen.'

from sklearn.feature_extraction.text import CountVectorizer

count_vec = CountVectorizer()

sentences = [sent1, sent2]

print(count_vec.fit_transform(sentences).toarray())

print(count_vec.get_feature_names())

import nltk

tokens_1 = nltk.word_tokenize(sent1)

print(tokens_1)

tokens_2 = nltk.word_tokenize(sent2)

print(tokens_2)

vocab_1 = sorted(set(tokens_1))
print(vocab_1)

vocab_2 = sorted(set(tokens_2))
print(vocab_2)

stemmer = nltk.stem.PorterStemmer()

stem_1 = [stemmer.stem(t) for t in tokens_1]
print(stem_1)


stem_2 = [stemmer.stem(t) for t in tokens_2]
print(stem_2)


pos_tag_1 = nltk.tag.pos_tag(tokens_1)
print(pos_tag_1)

pos_tag_2 = nltk.tag.pos_tag(tokens_2)
print(pos_tag_2)
