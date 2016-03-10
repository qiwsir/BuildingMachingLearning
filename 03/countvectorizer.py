#!/usr/bin/env python
# coding=utf-8

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy as sp
import sys
import nltk.stem

english_stemmer = nltk.stem.SnowballStemmer('english')
class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))

class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))

#统计词语个数，生成词频向量
#vectorizer = CountVectorizer(min_df=1, stop_words='english')
#vectorizer = StemmedCountVectorizer(min_df=1, stop_words='english')
vectorizer = StemmedTfidfVectorizer(min_df=1, stop_words='english', charset_error='ignore')

#content = ["how to format my hard disk format", "hard disk format problems"]
#content = ["苍 老师 是 好 老师", "老师 是 人类 灵魂 工程师" ]
content = ["This is a toy post about machine learning. Actually, it contains no much interesting stuff.", "Imaging databases can get huge.", "Most imaging databases safe images permanently.", "Imaging databases store images.", "Imaging databases store images. Imaging databases store images. Imaging databases store images."]

x = vectorizer.fit_transform(content)
print vectorizer.get_feature_names()       #得到词汇

print x.toarray().transpose()     # 每个词汇的出现次数，即词频向量

num_samples, num_features = x.shape

print "#samples:{0}, #features:{1}".format(num_samples, num_features)

new_post = "imaging databases"
new_post_vec = vectorizer.transform([new_post])
print new_post_vec
print new_post_vec.toarray()

def dist_raw(v1, v2):    #计算两个向量之间的距离，并返回矩阵范数，最小距离
    v1_normalized = v1 / sp.linalg.norm(v1.toarray())
    v2_normalized = v2 / sp.linalg.norm(v2.toarray())
    delta = v1_normalized - v2_normalized
    return sp.linalg.norm(delta.toarray())

best_doc = None
best_dist = sys.maxint
best_i = None

for i in range(0, num_samples):
    post = content[i]
    if post == new_post:
        continue
    post_vec = x.getrow(i)
    d = dist_raw(post_vec, new_post_vec)
    print "== Post {0} with dist={1}: {2}".format(i, d, post)

    if d<best_dist:
        best_dist = d
        best_i = i

print "Best post is {0} with dist={1}".format(best_i, best_dist)
