#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 18:24:59 2017

@author: yutingyw
"""

class node:
    def __init__(self,value):
        self.value = value
        self.left = None
        self.right = None
        
def insertBST(value,root):
    if root == None:
        return node(value)
    if value <= root.value:
        root.left = insertBST(value,root.left)
    else:
        root.right = insertBST(value,root.right)
    return root

def buildBST(lst):
    root = node(lst[0])
    for i in range(1,len(lst)):
        insertBST(lst[i],root)
    return root

def searchBST(value,root):
    if root == None:
        return -1
    if root.value == value:
        return root
    elif value < root.value:
        return searchBST(value,root.left)
    else:
        return searchBST(value,root.right)
    
def postOrder(root):
    if root == None:
        return []
    else:
        return postOrder(root.right) + [root.value] + postOrder(root.left)
    
def inOrder(root):
    if root == None:
        return []
    else:
        return inOrder(root.left) + [root.value] + inOrder(root.right)