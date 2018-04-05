#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 21:50:47 2017

@author: yutingyw
"""

class node:
    def __init__(self,value,key):
        self.value = value
        self.key = key
        self.left = None
        self.right = None
        self.equal = []
        
def insertBST(value,key,root):
    if root == None:
        return node(value,key)
    if value < root.value:
        root.left = insertBST(value,key,root.left)
    elif value > root.value:
        root.right = insertBST(value,key,root.right)
    else:
        root.equal.append(key)
    return root

def buildBST(lst,keys):
    assert len(lst) == len(keys)
    root = node(lst[0],keys[0])
    for i in range(1,len(lst)):
        insertBST(lst[i],keys[i],root)
    return root

def searchBST(value,root):
    if root == None:
        return -1
    if root.value == value:
        return root
    elif value < root.value:
        return searchBST(value,root.left)
    elif value > root.value:
        return searchBST(value,root.right)
    
def postOrder(root):
    if root == None:
        return []
    else:
        return postOrder(root.right) + [root] + postOrder(root.left)
    
def inOrder(root):
    if root == None:
        return []
    else:
        return inOrder(root.left) + [root] + inOrder(root.right)