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

def datingClassTest():
	hoRatio = 0.10

	datingDataMat, datingLabels = file2matrix("datingTestSet.txt")
	normMat, ranges, minVals = autoNorm(datingDataMat)

	# 데이터 총 개수
	m = normMat.shape[0]

	# 데이터 10% 개수
	numTestVecs = int(m * hoRatio)
	errorCount = 0.0

	for i in range(numTestVecs):
		# 앞 10% 데이터를 제외한 나머지 뒷 90% 데이터를 사용
		classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:], datingLabels[numTestVecs:m], 3)

		print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i])

		if(classifierResult != datingLabels[i]):
			errorCount += 1.0

	print "the total error rate is: %lf" % (errorCount/float(numTestVecs))

def classifyPerson():
	resultList = ["not at all", "in small doses", "in largeDoses"]
	percentTats = float(raw_input("percentage of time spent playing video games? "))
	ffMiles = float(raw_input("frequent flier miles earned per year? "))
	iceCream = float(raw_input("liters of ice cream consumed per year? "))

	datingDataMat, datingLabels = file2matrix("datingTestSet.txt")
	normMat, ranges, minVals = autoNorm(datingDataMat)
	inArr = array([ffMiles, percentTats, iceCream])
	classifierResult = classify0((inArr - minVals)/ranges, normMat, datingLabels, 3)

	print "You will probably like this person: ", resultList[int(classifierResult) - 1]

def img2vector(filename):
	returnVect = zeros((1, 1024))
	fr = open(filename)

	for i in range(32):
		lineStr = fr.readline()

		for j in range(32):
			returnVect[0, 32*i+j] = int(lineStr[j])

	return returnVect

def hadnwritingClassTest():
	hwLabels = []

	# trainig 데이터 파일 개수
	trainingFileList = listdir("digits/trainingDigits")
	m = len(trainingFileList)
	trainingMat = zeros((m, 1024))

	for i in range(m):
		fileNameStr = trainingFileList[i]
		fileStr = fileNameStr.split(".")[0]
		classNumStr = int(fileStr.split("_")[0])
		hwLabels.append(classNumStr)
		trainingMat[i, :] = img2vector("digits/trainingDigits/%s" % fileNameStr)

	# test 데이터 파일 개수
	testFileList = listdir("digits/testDigits")
	errorCount = 0.0
	mTest = len(testFileList)

	for i in range(mTest):
		fileNameStr = testFileList[i]
		fileStr = fileNameStr.split(".")[0]
		classNumStr = int(fileStr.split("_")[0])
		vectorUnderTest = img2vector("digits/testDigits/%s" % fileNameStr)

		classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)

		print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)

		if(classifierResult != classNumStr):
			errorCount += 1.0

	print "the total number of errors is: %d" % (errorCount)
	print "the total error rate is: %lf" % (errorCount/float(mTest))