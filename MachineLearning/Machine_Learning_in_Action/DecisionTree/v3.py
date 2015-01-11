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
# 전체 데이터 중 각 라벨이 나올 확률 복잡도
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

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0])-1
    # 전체 데이터의 라벨 복잡도
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1

    # 데이터 열 개수동안
    for i in range(numFeatures):
        # 해당 열이 가지는 값들
        featList = [example[i] for example in dataSet]
        # 해당 열이 가지는 값들을 중복 제거
        uniqueVals = set(featList)
        newEntropy = 0.0

        # 해당 열이 가지는 uniq 값들동안
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            # 확률 * 섀넌 엔트로피 : 해당 열로 구별했을때의 데이터가 나올 확률 * 구별된 데이터의 라벨 확률 복잡도
            newEntropy += prob * calcShannonEnt(subDataSet)

        # 정보 이득 : 전체 엔트로피에서 해당 열로 데이터 분할했을시 엔트로피 값 차이
        infoGain = baseEntropy - newEntropy

        # 지금까지 최고 정보 이득을 보인 열을 bestFeature로 기억
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i

    return bestFeature