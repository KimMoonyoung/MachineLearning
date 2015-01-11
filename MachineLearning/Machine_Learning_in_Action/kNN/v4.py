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

def file2matrix(filename):
	labels2int = { "didntLike":1, "smallDoses":2, "largeDoses":3}

	fr = open(filename)

	numberOfLines = len(fr.readlines())

	returnMat = zeros((numberOfLines, 3))
	classLabelVector = []
	fr = open(filename)
	index = 0

	for line in fr.readlines():
		line = line.strip()
		listFromLine = line.split("\t")

		# 테스트 데이터 중 앞 3개 정보, 라벨 저장
		returnMat[index, :] = listFromLine[0:3]
		classLabelVector.append(labels2int[listFromLine[-1]])

		index += 1

	return returnMat, classLabelVector

def autoNorm(dataSet):
	# 각 열의 최소값, 최대값 가져오기
	minVals = dataSet.min(0)
	maxVals = dataSet.max(0)

	# min, max normalize
	ranges = maxVals - minVals
	normDataSet = zeros(shape(dataSet))

	m = dataSet.shape[0]

	normDataSet = dataSet - tile(minVals, (m, 1))
	normDataSet = normDataSet / tile(ranges, (m, 1))

	return normDataSet, ranges, minVals