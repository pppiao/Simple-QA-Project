# -*- coding = utf-8 -*-
"""
@file name : related.py.py
@author    : tongpiao
@time      : 2020/9/23 18:09
@brief     : 
"""
# Todo: 读取glove词向量字典，计算词向量之间的相似度,获取单词的top10近义词，并写入本地related_words.py文件
# coding: utf-8
import json
import codecs
import numpy as np
import queue as Q


# Todo:加载glove 词向量,字符串转为float类型
def loadEmbeddingDic (filename):
    with codecs.open(filename, "r", encoding="utf-8") as Fin:
        lines = Fin.readlines()
        glove_wordemb = {}
        for line in lines:
            words = line.split(" ")
            glove_wordemb[words[0]] = np.array(words[1:], dtype=float)

    return glove_wordemb


glove_word_embeddings = loadEmbeddingDic('glove.6B.200d_origin.txt')
print("-----------------------------------")
# print(type(glove_word_embeddings["more"]))


# Todo:定义余弦相似度计算
def cosineSimilarity(vec1, vec2):
    # print("定义余弦相似度计算 Done")
    return np.dot(vec1, vec2.T) / (np.sqrt(np.sum(vec1 ** 2)) + np.sqrt(np.sum(vec2 ** 2)))


def get_top_related_words(wordemb):
    print("start get_top_related_words ")
    print("the length of word embedding: ", len(wordemb))
    related_words = {}
    top = 11
    # 优先级队列实现大顶堆Heap, 每次输出都是相似度最大值
    que = Q.PriorityQueue()
    # top_words = []

    for k1, v1 in wordemb.items():

        for k2, v2 in wordemb.items():
            sim_value = cosineSimilarity(v1, v2)
            # print("que*******************", k1, k2, "==", sim_value)
            que.put((-1 * float(sim_value), k2))

        idx = 0
        top_words = []
        while (idx < top and not que.empty()):
            top_words.append(que.get()[1])
            idx = idx + 1
        # print("top_words---", top_words)
        related_words[k1] = top_words
        # top_words.append("\n")

        print("*******top_words************\n", top_words)

    # return top_words
    return related_words


result = get_top_related_words(glove_word_embeddings)
print("------------------------------")
print(result)

# todo: 将结果写入本地txt文件
print("start 将结果写入本地txt文件")
jsObj = json.dumps(result)
fileObject = open("related_words.txt", "w")
fileObject.write(jsObj)
fileObject.close()


