#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'zhaopeng@miaozhen.com'
__date__ = '18-7-10 下午4:29'

from math import log


def calcShannonEnt(dataSet):
    '''
    计算数据集的熵值
    :param dataSet: 数据集,样本最后一列为label
    :return:熵的值
    '''
    numEntries = len(dataSet)
    labelCount = {}

    for featVec in dataSet:
        currentLable = featVec[-1]
        labelCount[currentLable] = labelCount.get(currentLable, 0) + 1

    shannonEnt = 0.0
    for key in labelCount:
        prob = float(labelCount[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)

    return shannonEnt


def splitDataSet(dataSet, axis, value):
    '''
    制定某特征的一具体值提取数据值
    :param dataSet:数据集
    :param axis:特征列
    :param value:该特征列的某一具体值
    :return:
    '''
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)

    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    '''
    选择出当前数据集信息增益最大的特征
    :param dataSet: 数据集
    :return: 信息增益最大的特征列索引t
    '''
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)

    bestInfoGain = 0.0
    bestFeature = -1

    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)

        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i

    return bestFeature


def majorityCnt(classList):
    '''
    当叶子节点存在多个分类时,采用投票法给出结果
    :param classList: 叶子节点中所有的分类
    :return:数量出现最多的分类
    '''

    classCount = {}
    for vote in classList:
        classCount[vote] = classCount.get(vote, 0) + 1
    sortedCount = sorted(classCount.items(), key=lambda x: x[1], reverse=True)
    return sortedCount[0][0]


def creatTree(dataSet, labels):
    '''
    决策树生成算法
    :param dataSet: 数据集合
    :param labels: 特征列的名称
    :return: 返回决策树
    '''
    classList = [example[-1] for example in dataSet]
    # 样本中的label为一种分类,直接返回
    if classList.count(classList[0]) == len(dataSet):
        return classList[0]

    # 样本中只剩label没有特征
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])

    featValues = [example[bestFeat] for example in dataSet]
    uniqueValue = set(featValues)

    for value in uniqueValue:
        sublabels = labels[:]
        myTree[bestFeatLabel][value] = creatTree(splitDataSet(dataSet, bestFeat, value), sublabels)

    return myTree


dataSet = [
    ['晴', '否', '睡觉'],
    ['阴', '是', '上班'],
    ['多云', '否', '睡觉'],
    ['雨', '否', '睡觉'],
    ['晴', '是', '上班'],
    ['阴', '是', '上班'],
    ['雨', '是', '上班']
]

print(creatTree(dataSet, ['天气', '是否是工作日']))
