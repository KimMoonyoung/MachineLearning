#!/usr/bin/env python
# -*- coding:utf-8 -*-

from math import log
import operator

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']

    #change to discrete values
    return dataSet, labels

# 섀넌 엔트로피 계산
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}

    for featVec in dataSet: #the the number of unique elements and their occurance
        currentLabel = featVec[-1]

        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0

        labelCounts[currentLabel] += 1

    shannonEnt = 0.0

    # 섀넌 엔트로피 : -sum(p(x)*log2p(x))
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob, 2) #log base 2

    return shannonEnt

# 지니 불순도 계산
def calcGiniImpurity(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}

    for featVec in dataSet: #the the number of unique elements and their occurance
        currentLabel = featVec[-1]

        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0

        labelCounts[currentLabel] += 1

    giniImpurity = 0.0

    # 지니 불순도 : 1-sum(p(x)^2)
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        giniImpurity += pow(prob, 2)

    giniImpurity = 1 - giniImpurity

    return giniImpurity

def splitDataSet(dataSet, axis, value):
    retDataSet = []

    for featVec in dataSet:
        # 리스트 axis 열의 데이터가 value인 것의 나머지 열 데이터 추출
        if featVec[axis] == value:
            reduceFeatVec = featVec[:axis]
            # extend : [1, 2], [3, 4] => [1, 2, 3, 4]
            reduceFeatVec.extend(featVec[axis+1:])
            # append : [1, 2], [3, 4] => [1, 2, [3, 4]]
            retDataSet.append(reduceFeatVec)

    return retDataSet