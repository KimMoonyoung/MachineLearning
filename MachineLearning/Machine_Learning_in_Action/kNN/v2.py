#!/usr/bin/env python
# -*- coding:utf-8 -*-

from numpy import *
import operator
from os import listdir

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']

    return group, labels

def classify0(inX, dataSet, labels, k):
	dataSetSize = dataSet.shape[0]

	# tile(x, y) : x 리스트를 y개 만큼 중복 생성
	diffMat = tile(inX, (dataSetSize, 1)) - dataSet

	# 좌표간의 유클리드 거리 계산
	sqDiffMat = diffMat ** 2
	sqDistances = sqDiffMat.sum(axis=1)
	distances = sqDistances ** 0.5

	# 거리를 오름차순으로 정렬
	sortedDistIndicies = distances.argsort()

	# 어떤 라벨이 많이 나왔는지 저장
	classCount = {}

	# k개까지 라벨 확인
	for i in range(k):
		vote1label = labels[sortedDistIndicies[i]]
		# 파이썬 사전 get 함수를 사용하여 현재까지 저장된 개수 받아옴
		classCount[vote1label] = classCount.get(vote1label, 0) + 1

	# 라벨 개수로 내림차순 정렬
	sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)

	return sortedClassCount[0][0]