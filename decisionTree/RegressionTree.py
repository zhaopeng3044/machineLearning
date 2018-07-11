#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'zhaopeng@miaozhen.com'
__date__ = '2018/7/11 下午6:20'

from numpy import *


def lodDataSet(fileName):
    dataMat = []
    with open(fileName, 'r') as fd_in:
        for line in fd_in:
            dataMat.append(map(float, line.strip().split('\t')))
    return dataMat


def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0], :][0]
    mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :][0]

    return mat0, mat1


def regLeaf(dataSet):
    return mean(dataSet[:, -1])


def regErr(dataSet):
    return var(dataSet[:, -1]) * shape(dataSet)[0]


def chooseBestSplit(dataSet, leafType, errType, ops):
    tols = ops[0]
    tolN = ops[1]

    if len(set(dataSet[:, -1].T.tolist())) == 1:
        return None, leafType(dataSet)

    m, n = shape(dataSet)
    s = errType(dataSet)
    bestS = inf
    bestIndex = 0
    bestValue = 0

    for featIndex in range(n - 1):
        for splitVal in set(dataSet[:, featIndex]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if shape(mat0)[0] < tolN or shape(mat1)[0] < tolN:
                continue

            newS = errType(mat0) + errType[1]
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS

    if (s - bestS) < tols:
        return None, leafType(dataSet)

    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)

    if shape(mat0)[0] < tolN or shape(mat1)[0] < tolN:
        return None, leafType(dataSet)
    return bestIndex, bestValue


def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None:
        return val

    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val

    lSet, rSet = binSplitDataSet(dataSet, feat, val)

    retTree['left'] = createTree(lSet, leafType, errType, ops)

    retTree['right'] = createTree(rSet, leafType, errType, ops)

    return retTree
