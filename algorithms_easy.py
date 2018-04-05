#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 19:49:48 2018

@author: yutingyw
"""

"""FB
We have our lists of orders sorted numerically already, in lists.
Write a function to merge our lists of orders into one sorted list.
"""
def merge_sorted_lst(lst1, lst2):
    ret = []
    i, j = 0, 0
    while (i < len(lst1)) and (j < len(lst2)):
        if lst1[i] < lst2[j]:
            ret.append(lst1[i])
            i += 1
        else:
            ret.append(lst2[j])
            j += 1
    if i == len(lst1):
        return ret + lst2[j:]
    if j == len(lst2):
        return ret + lst1[i:]

def test_msl():
    assert merge_sorted_lst([3, 4, 6, 10, 11, 15], [1, 5, 8, 12, 14, 19]) \
                           == [1, 3, 4, 5, 6, 8, 10, 11, 12, 14, 15, 19]

"""FB
Given an array of random numbers, push (in place) all the zero’s of a given
array to the end of the array.
"""
def sink_zeros(lst):
    n = len(lst)
    count = 0
    for i in range(n):
        if lst[i] != 0:
            lst[count] = lst[i]
            count += 1
    while count < n:
        lst[count] = 0
        count += 1
    return lst

def test_sz():
    sink_zeros([1, 9, 8, 4, 0, 0, 2, 7, 0, 6, 0, 9]) \
        == [1, 9, 8, 4, 2, 7, 6, 9, 0, 0, 0, 0]

"""
given the highest possible score, return a sorted list of
scores in descending order
"""
def count_sort(lst,highest):
    sort_dic = [0]*highest
    for i in lst:
        sort_dic[i] += 1
    sort = []
    for i in range(highest-1,-1,-1):
        count = sort_dic[i]
        if count>0:
            sort += [i]*count
    return sort

def test_cs():
    assert count_sort([37, 89, 65, 65, 91, 53],100)==\
                     [91, 89, 65, 65, 53, 37]

"""
find 2 numbers sumed up to an amount
"""
def sum2(amount,lst):
    counts = {}
    for n in lst:
        counts[n] = 1+counts.get(n,0)
    for n in lst:
        rest = amount-n
        if rest in counts:
            return (n,rest)
    return False

def test_s2():
    assert sum2(13,[2,3,4,12])==False
    assert sum2(12,[1,3,4,6,8])==(4,8)

"""FB
Given strings s and x, locate the occurrence of x in s. The function 
returns the integer denoting the first occurrence of the string x .
"""
def locate_substring(s, x):
    i = 0
    for i in range(len(s) - len(x) + 1):
        for j in range(len(x)):
            if s[i+j] == x[j]:
                j += 1
            else:
                break
        if j == len(x):
            return j - len(x) + 1
        i += 1
    return -1

def test_ls():
    assert locate_substring('abcdefgbcd', 'bcd')
    assert locate_substring('abcdefgbcd', 'abcd')

"""
takes a list of multiple meeting time ranges and returns
a list of condensed ranges
"""
def merge_ranges(lst):
    slst = sorted(lst)
    merged = [slst[0]]
    for i,(start,end) in enumerate(slst):
        if start<=merged[-1][1]:
            merged[-1] = (merged[-1][0],max(end,merged[-1][1]))
        else:
            merged.append((start,end))
    return merged

def test_mr():
    assert merge_ranges([(0, 1), (3, 5), (4, 8), (10, 12),\
                         (9, 10)])==[(0, 1), (3, 8), (9, 12)]

"""
compute fibonacci
"""
def fib(n):
    if n in [0,1]:
        return n
    prev = 1
    prev_prev = 0
    for _ in range(2,n+1):
        s = prev+prev_prev
        prev_prev = prev
        prev = s
    return s

def test_f():
    assert [fib(n) for n in range(5)]==[0,1,1,2,3]
    
"""FB
Given an array, verify if the reverse is the same.
"""
def palindrome(lst):
    n = len(lst)
    for i in range(n//2):
        if lst[i] != lst[n-i-1]:
            return False
    return True

def test_palindrome():
    assert palindrome([1,3,4,3,1]) == True
    assert palindrome([1,2,2,1]) == True
    assert palindrome([1,4,5]) == False

"""FB
Given a string, determine if it is a palindrome, ignoring special characters
and upper/lower cases.
"""
def special_palindrome(string):
    alpha = ''
    for c in string:
        if c.isalpha():
            alpha += c.lower()
    return palindrome(alpha)

def test_sp():
    assert special_palindrome("A7*kkA") == True

"""
checks if there is any permutation of an input string is
a palindrome
"""
def permutation_palindrome(string):
    pair = set()
    for c in string:
        if c in pair:
            pair.remove(c)
        else:
            pair.add(c)
    return len(pair)<=1

def test_pp():
    assert permutation_palindrome('civic')==True
    assert permutation_palindrome('ivicc')==True
    assert permutation_palindrome('civil')==False

"""FB
Find the minimum depth of a binary tree and return the path.
"""
def min_depth(root, prev):
    if root == None:
        return 0, prev
    else:
        left = min_depth(root.left, prev + [root.value])
        right = min_depth(root.right, prev + [root.value])
        if left[0] <= right[0]:
            return left
        else:
            return right

def test_md():
    class Node(object):
        def __init__(self, value):
            self.value = value
            self.left = None
            self.right = None
    root = Node(0)
    root.left = Node(1)
    root.right = Node(2)
    root.left.left = Node(3)
    print(min_depth(root, []))

"""
Delete a node from a singly-linked list
"""
def delete_node(node):
    next_node = node.next
    if next_node:
        node.next = next_node.next
        node.value = next_node.value
    else:
        node = None

def test_dn():
    class LinkedListNode:
        def __init__(self, value):
            self.value = value
            self.next  = None

    a = LinkedListNode('A')
    b = LinkedListNode('B')
    c = LinkedListNode('C')
    a.next = b
    b.next = c
    delete_node(b)
    print(a.value,a.next,b.value,b.next)

"""
You want to be able to access the largest element in a stack
"""
class Stack(object):
    def __init__(self):
        self.items = []
    def push(self, item):
        self.items.append(item)
    def pop(self):
        if not self.items:
            return None
        return self.items.pop()
    def peek(self):
        if not self.items:
            return None
        return self.items[-1]

class MaxStack(object):
    def __init__(self):
        self.stack = Stack()
        self.maxes_stack = Stack()
    def push(self, item):
        self.stack.push(item)
        if self.maxes_stack.peek() is None or \
                                item >= self.maxes_stack.peek():
            self.maxes_stack.push(item)
    def pop(self):
        item = self.stack.pop()
        if item == self.maxes_stack.peek():
            self.maxes_stack.pop()
        return item
    def get_max(self):
        return self.maxes_stack.peek()

"""
You have a function rand5() that generates a random integer
from 1 to 5. Use it to write a function rand7() that generates
a random integer from 1 to 7
"""
def rand7():
    while True:
        # do our die rolls
        roll1 = rand5()
        roll2 = rand5()
        outcome_number = (roll1-1) * 5 + (roll2-1) + 1
        # if we hit an extraneous outcome we just re-roll
        if outcome_number > 21:
            continue
        # our outcome was fine. return it!
        return outcome_number % 7 + 1
