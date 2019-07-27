#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 18:57:47 2018

@author: yutingyw
"""

"""
Regular expression matching.
"""
def regex_match(s, p):
    T = {} # shall not use array, as reference is easily changed unexpectedly [[False] * (len(p) + 1)] * (len(s) + 1)
    T[(0, 0)] = True # '' matched with ''

    for j in range(1, len(p) + 1):
        if p[j - 1] == '*':
            T[(0, j)] = T[(0, j - 1)] # only '*' matches with ''
        else:
            T[(0, j)] = False

    for i in range(1, len(s) + 1):
        T[(i, 0)] = False

    for i in range(1, len(s) + 1):
        for j in range(1, len(p) + 1):

            if p[j - 1] == '*': # corresonds to j'th column of T
                print('* i=%d j=%d T[(%d,%d)]=%d T[(%d,%d)]=%d' % (i, j, i, j-1, T[(i, j-1)], i-1, j, T[(i-1,j)]))
                T[(i, j)] = T[(i, j - 1)] or T[(i - 1, j)]

            elif p[j - 1] == '?' or p[j - 1] == s[i - 1]:
                print('i=%d j=%d T[(%d,%d)]=%d' % (i, j, i-1, j-1, T[(i-1, j-1)]))
                T[(i, j)] = T[(i - 1, j - 1)]

            else:
                print('false', i,j)
                T[(i, j)] = False

    print(T)
    return T[(len(s), len(p))]

assert regex_match('ba', 'ba') == True
assert regex_match('baaabab', 'baaa?ab') == True
assert regex_match('baaabab', 'ba*a?') == True
assert regex_match('baaabab', 'a*ab') == False
assert regex_match('baaabab', '*****ba*****ab') == True

"""FB
Given two arrays, return True if the words array is sorted according to the
ordering array.
"""
def compare_two_words(words, ordering):
    if '' in words:
        return True
    else:
        try:
            first_index = ordering.index(words[0][0])
            second_index = ordering.index(words[1][0])
        except:
            return False
        if first_index > second_index:
            return False
        elif first_index == second_index:
            return compare_two_words([w[1:] for w in words], ordering)
        else:
            return True

def sort_by_order(words, ordering):
    if len(words) == 1:
        return True
    return compare_two_words(words[:2], ordering) and sort_by_order(words[1:], ordering)

words = ['cc','cb','bb','ac']
assert sort_by_order(words, ['c','b','a']) == True
assert sort_by_order(words, ['b','c','a']) == False
assert sort_by_order(words, ['c','b']) == False

"""FB
Given a string of keys (can have different combinations of keys), and a
key-to-patterns dictionary, return a list of all patterns possible.
"""
def combinations_lst(keystr, keylst, prev):
    if keystr == '':
        print([prev])
        return [prev]
    combinations = []
    for i in range(1,len(keystr)+1):
        if keystr[:i] in keylst:
            print(keystr[:i])
            combinations += combinations_lst(keystr[i:], keylst, prev + [keystr[:i]])
    return combinations

def test_cl():
    assert combinations_lst('12',['1','12','2'],[]) == [['1','2'], ['12']]
    assert combinations_lst('1234',['1','2','12','23','3','34'],[]) \
        == [['1','2','34'], ['12','34']]
    assert combinations_lst('1234',['1','2','12','23','3','34','4'],[]) \
        == [['1','2','3','4'], ['1','2','34'], ['1','23','4'], ['12','3','4'], ['12','34']]

def patterns_lst(combination, key_patterns, prefix):
    if combination == []:
        return [prefix]
    patterns = []
    for pattern in key_patterns[combination[0]]:
        patterns += patterns_lst(combination[1:], key_patterns, prefix + pattern)
    return patterns

def test_pl():
    keys_patterns = {'1':['A','B','C'], \
                     '2':['D','E'], \
                     '3':['P','Q']}
    assert patterns_lst(['1','2','3'], keys_patterns, '') == ['ADP', 'ADQ', 'AEP', \
        'AEQ', 'BDP', 'BDQ', 'BEP', 'BEQ', 'CDP', 'CDQ', 'CEP', 'CEQ']

def keys_to_patterns(keystr, keys_patterns):
    keylst = list(keys_patterns.keys())
    combinations = combinations_lst(keystr, keylst, [])
    patterns = []
    for combination in combinations:
        patterns += patterns_lst(combination, keys_patterns, '')
    return patterns

def test_ktp():
    keys_patterns = {'1':['A','B','C'], \
                     '2':['D','E'], \
                     '12':['X'], \
                     '3':['P','Q']}
    assert keys_to_patterns('123', keys_patterns) == ['ADP', 'ADQ', 'AEP', \
        'AEQ', 'BDP', 'BDQ', 'BEP', 'BEQ', 'CDP', 'CDQ', 'CEP', 'CEQ', 'XP', 'XQ']    

"""FB
Count the number of possible decodings given a digit sequence.
"""
ord_chr = {str(i - ord('A') + 1): chr(i) for i in range(ord('A'), ord('Z') + 1)}
def decode_combinations(digits, prefix):
    if len(digits) == 0:
        return [prefix]
    combinations = []
    for i in range(1, len(digits) + 1):
        if digits[:i] in ord_chr:
            decode = ord_chr[digits[:i]]
            combinations += decode_combinations(digits[i:], prefix + decode)
    return combinations

assert decode_combinations('121', '') == ['ABA','AU','LA']
assert decode_combinations('1234', '') == ['ABCD','AWD','LCD']

"""FB
Construct binary tree from given inorder and preorder traversal lists.
"""
class Node(object):
    def __init__(self):
        self.value = None
        self.left = None
        self.right = None

def construct_bt(root, inorder):
    if len(construct_bt.preorder) == 0:
        return
    if len(inorder) == 0:
        return
    value = construct_bt.preorder.popleft()
    root.value = value
    print(value)
    print('inorder ', inorder)
    print('preorder ', construct_bt.preorder)
    inorder_index = inorder.index(value)
    print('inorder_index ', inorder_index)
    if inorder_index > 0:
        root.left = Node()
        root.right = Node()
        print('left inorder ', inorder[:inorder_index])
        print('right inorder ', inorder[(inorder_index+1):])
        construct_bt(root.left, inorder[:inorder_index])
        construct_bt(root.right, inorder[(inorder_index+1):])
    else:
        root.right = Node()
        print('right inorder ', inorder[1:])
        construct_bt(root.right, inorder[1:])

def inorder_lst(root):
    if not root:
        return []
    return inorder_lst(root.left) + [root.value] + inorder_lst(root.right)

def preorder_lst(root):
    if not root:
        return []
    return [root.value] + preorder_lst(root.left) + preorder_lst(root.right)

def test_cb():    
    from collections import deque
    inorder = ['D','B','E','A','F','C']
    preorder = deque(['A','B','D','E','C','F'])
    root = Node()
    construct_bt.preorder = preorder
    construct_bt(root, inorder)
    assert inorder_lst(root) == ['D', None, 'B', 'E', None, 'A', 'F', None, 'C', None]
    assert preorder_lst(root) == ['A', 'B', 'D', None, 'E', None, 'C', 'F', None, None]

"""
A tree is "superbalanced" if the difference between the
depths of any two leaf nodes is no greater than one
"""
def is_balanced_dfs(root):
    if root==None:
        return True
    depths = []
    stacks = [(root,0)]
    while len(stacks)>0:
        node,depth = stacks.pop()
        if not node.left and not node.right:
            if depth not in depths:
                depths.append(depth)
            if len(depths)>2:
                return False
            if len(depths)==2 and abs(depths[0]-depths[1])>1:
                return False
        else:
            if node.left:
                stacks.append((node.left,depth+1))
            if node.right:
                stacks.append((node.right,depth+1))
    return True

def is_balanced_bfs(root):
    from collections import deque
    if root==None:
        return True
    depths = []
    queue = deque([(root,0)])
    while len(queue)>0:
        node,depth = queue.popleft()
        if not node.left and not node.right:
            if depth not in depths:
                depths.append(depth)
            if len(depths)>2:
                return False
            if len(depths)==2 and abs(depths[0]-depths[1])>1:
                return False
        else:
            if node.left:
                queue.append((node.left,depth+1))
            if node.right:
                queue.append((node.right,depth+1))
    return True

"""
Construct balanced bst from sorted list.
"""
def bst_from_sorted(root, lst):
    if not lst:
        return
    mid = len(lst) // 2
    root.value = lst[mid]
    print(mid, root.value)
    if mid > 0:
        root.left = Node()
        bst_from_sorted(root.left, lst[:mid])
    if mid < len(lst) - 1:
        root.right = Node()
        bst_from_sorted(root.right, lst[(mid+1):])

def tree_lst(root, left_space, right_space):
    if not root:
        return ' '*left_space + '*' + ' '*right_space
    if not root.value:
        root.value = '*'
    lst = tree_lst(root.left, left_space, right_space + 2) + str(root.value) + \
                  tree_lst(root.right, left_space + 2, right_space)
    print(lst)
    return

def test_bfs():
    root = Node()
    bst_from_sorted(root, [1,2,3,4,5,6,7])
    assert inorder_lst(root) == [1, 2, 3, 4, 5, 6,7]
    print(preorder_lst(root))
    tree_lst(root, 1, 1)
    return root

"""
check if a binary tree is a BST
"""
def is_bst(root):
    if root==None:
        return True
    stacks = [(root,float('-inf'),float('inf'))]
    while len(stacks):
        node,low,up = stacks.pop()
        if node.val<=low or node.val>=up:
            return False
        else:
            if node.left:
                stacks.append((node.left,low,node.val))
            if node.right:
                stacks.append((node.right,node.val,up))
    return True

def test_ib():
    class BinaryTreeNode(object):
        def __init__(self, value):
            self.val = value
            self.left  = None
            self.right = None

        def insert_left(self, value):
            self.left = BinaryTreeNode(value)
            return self.left

        def insert_right(self, value):
            self.right = BinaryTreeNode(value)
            return self.right

    t0 = BinaryTreeNode(10)
    t00 = t0.insert_left(5)
    t01 = t0.insert_right(15)
    t010 = t01.insert_left(12)
    t010.insert_right(14)
    assert is_balanced_dfs(t0)==False
    assert is_balanced_dfs(t01)==True
    assert is_balanced_bfs(t0)==False
    assert is_balanced_bfs(t01)==True
    assert is_bst(t0)==True

    t00.insert_right(13)
    assert is_bst(t0)==False

"""
save large number of URLs in an efficient way
"""
def save_url(lst):
    root = {}
    for word in lst:
        current = root
        for c in word:
            if c not in current:
                current[c] = {}
            current = current[c]
        if '*' in current:
            print(word+' existed')
        else:
            print('saved '+word)
            current['*'] = {}
    return root

assert save_url(['go.ai','go.ai/alpha','go.ai'])== \
    {'g': {'o': {'.': {'a': {'i': {'*': {}, \
    '/': {'a': {'l': {'p': {'h': {'a': {'*': {}}}}}}}}}}}}}

"""
find common prefix of a list of words
"""
def common_prefix(lst):
    root = {}
    for word in lst:
        current = root
        for c in word:
            if c not in current:
                current[c] = {}
            current = current[c]
        if '*' in current:
            print(word + ' existed')
        else:
            print('saved ' + word)
            current['*'] = {}

    current = root
    common = ''
    while(len(current.keys()) == 1):
        k = list(current.keys())[0]
        if k != '*':
            common += k
            current = current[k]
        else:
            break
    return common

assert common_prefix(['abcde', 'abcre', 'abdde']) == 'ab'
assert common_prefix(['abcde', 'abcdefg', 'abcdefgh']) == 'abcde'

"""FB
Given a dictionary and a M x N board where every cell has one character. 
Find all possible words that can be formed by a sequence of adjacent 
characters. Note that we can move to any of 8 adjacent characters, but 
a word should not have multiple instances of same cell.
"""
def build_trie(words):
    root = {}
    for word in words:
        current = root
        for c in word:
            if c not in current:
                current[c] = {}
            current = current[c]
    return root

def test_br():
    assert build_trie(["GEEKS", "FOR", "QUIZ", "GO"]) == \
        {'G': {'E': {'E': {'K': {'S': {}}}}, 'O': {}}, \
         'F': {'O': {'R': {}}}, 'Q': {'U': {'I': {'Z': {}}}}}

def search_word(current_root, i, j, visited, prefix, boggle):
    if not current_root:
        return [prefix]
    words = []
    visited[i][j] = True
    for row in [i-1, i, i+1]:
        if row < 0 or row >= len(boggle):
            continue
        for col in [j-1, j, j+1]:
            if col < 0 or col >= len(boggle[0]):
                continue
            if not visited[row][col]:
                c = boggle[row][col]
                if c in current_root:
                    words += search_word(current_root[c], row, col, visited, 
                                         prefix + c, boggle)
                    visited[row][col] = True
    return words

def test_sw():
    boggle = [['G','I','Z'], \
              ['U','E','K'], \
              ['Q','S','E']]
    words = {"GEEKS", "FOR", "QUIZ", "GO"}
    trie = build_trie(words)
    found = []
    for i in range(len(boggle)):
        for j in range(len(boggle[0])):
            visited = [[False for _ in range(len(boggle[0]))] \
                        for _ in range(len(boggle))]
            c = boggle[i][j]
            if c in trie:
                found += search_word(trie[c], i, j, visited, c, boggle)
    assert found == ['GEEKS', 'QUIZ']

"""
Find the longest common subsequence.
"""
def lcs(X, Y):
    m = len(X)
    n = len(Y)
    memo = [[None] * (n+1) for _ in range(m+1)]
    for i in range(m+1):
        for j in range(n+1):
            if i == 0 or j == 0:
                memo[i][j] = ''
            elif X[i-1] == Y[j-1]:
                memo[i][j] = memo[i-1][j-1] + X[i-1]
            elif len(memo[i-1][j]) > len(memo[i][j-1]):
                memo[i][j] = memo[i-1][j]
            else:
                memo[i][j] = memo[i][j-1]
    return memo[m][n]

assert lcs('AGGTAB', 'GXTXAYB') == 'GTAB'

"""FB
Determine if a string is k-palindrome, i.e. palindrome upon removing at most
k characters.
"""
def is_k_palindrome(string, k):
    reverse = string[::-1]
    s = lcs(string, reverse)
    return len(string) - len(s) <= k

assert is_k_palindrome('acdcb', 2)
assert is_k_palindrome('acdcb', 1) == False

"""
computes the number of ways to make the amount of money
with coins of the available denominations
"""
memo={}
def ways_make_amount(amount,lst):
    memo_key = (amount,len(lst))
    if memo_key in memo:
        print('grabbing '+str(memo_key))
        return memo[memo_key]

    if amount==0:
        return 1
    if amount<0:
        return 0
    if lst==[]:
        return 0

    n_ways = 0
    while amount>=0:
        print('looping with '+str(amount)+' and '+str(lst[1:]))
        n_ways += ways_make_amount(amount,lst[1:])
        print('deducting '+str(lst[0])+' from '+str(amount))
        amount -= lst[0]
    memo[memo_key] = n_ways
    return n_ways

def test_wma():
    assert ways_make_amount(4,[1,2,3])==4

"""
Find the number of contiguous subarrays with given sum.
"""
def subarray_given_sum(lst, k):
    s = 0
    dic = {0: 1}
    count = 0
    for n in lst:
        s += n
        if s - k in dic:
            print(s, s-k, count, dic[s-k])
            count += dic[s - k]
            print(count)

        if s not in dic:
            dic[s] = 1
        else:
            dic[s] += 1
    return count

assert subarray_given_sum([1, -1, 1], 2) == 0, subarray_given_sum([1, -1, 1], 2)
assert subarray_given_sum([1, -1, 1, 1], 2) == 2, subarray_given_sum([1, -1, 1, 1], 2)
assert subarray_given_sum([1, -1, 1, 1, 1, -1, 2], 2) == 8, subarray_given_sum([1, -1, 1, 1, 1, -1, 2], 2)

"""FB
Find positive subarray with given sum.
"""
def psubarray_given_sum(lst, amt):
    curr_sum = 0
    start = 0
    for i in range(len(lst)):
        curr_sum += lst[i]
        while (curr_sum > amt) and (start < i):
            curr_sum -= lst[start]
            start += 1
        if curr_sum == amt:
            return start, i
    return -1

assert psubarray_given_sum([1, 4, 20, 3, 10, 5], 33) == (2, 4)
assert psubarray_given_sum([1, 4, 0, 0, 3, 10, 5], 7) == (1, 4)
assert psubarray_given_sum([1, 4], 0) == -1

"""
Largest sum of contiguous subarray.
"""
def subarray_max_sum(lst):
    max_sum = 0
    max_start = 0
    max_end = 0
    curr_sum = 0
    curr_start = 0
    for i in range(len(lst)):
        if curr_sum < 0:
            curr_sum = lst[i]
            curr_start = i
        else:
            curr_sum += lst[i]
        if max_sum < curr_sum:
            max_sum = curr_sum
            max_start = curr_start
            max_end = i
    return max_sum, max_start, max_end

    # version 2
    sliding_sum, max_sum = 0, 0
    sliding_start, max_start, max_end = 0, 0, 0
    for sliding_end in range(len(lst)):
        sliding_sum += lst[sliding_end]
        if sliding_sum < 0:
            sliding_sum = 0
            sliding_start = sliding_end + 1
        if sliding_sum > max_sum:
            max_sum = sliding_sum
            max_start = sliding_start
            max_end = sliding_end
    return max_sum, max_start, max_end

assert subarray_max_sum([-2, -3, 4, -1, -2, 1, 5, -3]) == (7, 2, 6)

"""
Largest sum of k contiguous subarray.
"""
def k_subarray_max_sum(lst, k):
    assert len(lst) >= k
    max_sum = 0
    max_start = 0
    max_end = 0
    curr_sum = sum(lst[:k])
    for i in range(k, len(lst)):
        curr_sum += lst[i]
        curr_sum -= lst[i - k]
        if max_sum < curr_sum:
            max_sum = curr_sum
            max_start = i - k + 1
            max_end = i
    return max_sum, max_start, max_end

    # version 2
    n = len(lst)
    sliding_start, max_start, max_end = 0, 0, k-1
    sliding_sum = sum(lst[:k])
    max_sum = sliding_sum
    for sliding_end in range(k, n):
        sliding_sum += lst[sliding_end] - lst[sliding_start]
        sliding_start += 1
        if sliding_end > max_sum:
            max_sum = sliding_sum
            max_start = sliding_start
            max_end = sliding_end
    return max_sum, max_start, max_end

assert k_subarray_max_sum([1, 2, 3, 1, 4, 5, 2, 3, 6], 3) == (11, 4, 6), k_subarray_max_sum([1, 2, 3, 1, 4, 5, 2, 3, 6], 3)

"""FB
Given an array and a number k, find the largest sum of the subarray containing 
at least k numbers. It may be assumed that the size of array is at least k.
"""
def greaterk_subarray_max_sum(lst, k):
    running_sum = sum(lst[:k])
    max_sum = running_sum
    max_start = 0
    max_end = k
    start = 0
    for end in range(k, len(lst)):
        running_sum += lst[end]
        adjusted = False
        while (start <= end - k + 1):
            if max_sum < running_sum:
                max_sum = running_sum
                max_start = start
                max_end = end
            running_sum -= lst[start]
            start += 1
            adjusted = True
        if adjusted:
            start -= 1
            running_sum += lst[start]
        end += 1
    return [lst[i] for i in range(max_start, max_end+1)]

#    running_sum = sum(lst[:k])
#    max_sum = running_sum
#    max_start = 0
#    max_end = k
#    start = 0
#    for end in range(k, len(lst)):
#        adjusted = False
#        while (running_sum <= max_sum) and (start <= end - k + 1):
#            running_sum -= lst[start]
#            start += 1
#            adjusted = True
#        if adjusted:
#            start -= 1
#            running_sum += lst[start]
#        if running_sum > max_sum:
#            max_start = start
#            max_end = end
#            max_sum = running_sum
#        end += 1
#    return [lst[i] for i in range(max_start, max_end+1)]

def test_gsms():
    assert greaterk_subarray_max_sum([-4, -2, 1, -3], 2) == [-2, 1]
    print(greaterk_subarray_max_sum([-4, 1, 1, -1, 1, 1, 1, 1], 3) )
#        == [1, 1, -1, 1, 1, 1, 1]

"""FB
Find the smallest subarray with sum greater than x.
"""
def min_subarray_greaterx_sum(lst, x):
    running_sum = 0
    start = 0
    min_start = 0
    min_end = len(lst)
    for end in range(len(lst)):
        running_sum += lst[end]
        subtracted = False
        while (running_sum > x) and (start <= end):
            if min_end-min_start > end-start:
                min_end = end
                min_start = start
            running_sum -= lst[start]
            start += 1
            subtracted = True
        if subtracted:
            start -= 1
            running_sum += lst[start]
        end += 1
    return [lst[i] for i in range(min_start, min_end+1)]

def test_msgs():
    assert min_subarray_greaterx_sum([1, 4, 45, 6, 0, 19], 51) == [4, 45, 6]
    assert min_subarray_greaterx_sum([1, 10, 5, 2, 7], 9) == [10]

"""FB
Balanced parentheses.
"""
def balance_parentheses(lst):
    op = 0
    cl = 0
    ret = ''
    for c in lst:
        if c == '(':
            op += 1
            ret += c
        elif c == ')':
            if cl < op:
                cl += 1
                ret += c
    rett = ''
    for c in ret:
        if (op > cl) and (c == '('):
            op -= 1
        else:
            rett += c
    return rett

    # version 2
    op = 0
    cl = 0
    ret = ''
    for c in lst:
        if c == '(':
            op += 1
        else:
            cl += 1
        if op >= cl:
            ret += c
        else:
            cl -= 1
        print(ret, op, cl)
    rett = ''
    count = 0
    for c in ret:
        if c == '(':
            count += 1
            if count > cl:
                continue
        rett += c
    return rett

assert balance_parentheses(')((())') == '(())'
assert balance_parentheses(')))(()))())') == '(())()'

"""
Check whether or not an input string's openers and closers are properly nested.
"""
def is_valid(code):
    openers_to_closers = {
        '(' : ')',
        '{' : '}',
        '[' : ']'
    }

    openers = frozenset(openers_to_closers.keys())
    closers = frozenset(openers_to_closers.values())

    openers_stack = []

    for char in code:
        if char in openers:
            openers_stack.append(char)
        elif char in closers:
            if not openers_stack:
                return False
            else:
                last_unclosed_opener = openers_stack.pop()

                # if this closer doesn't correspond to the most recently
                # seen unclosed opener, short-circuit, returning false
                if not openers_to_closers[last_unclosed_opener] == char:
                    return False

    return openers_stack == []

assert is_valid('(()[]') == False
assert is_valid('(())[{}[]]') == True

"""
takes stock_prices_yesterday and returns the best profit
I could have made from 1 purchase and 1 sale of 1 Apple
stock yesterday
"""
def max_profit(lst):
    if len(lst)<2:
        raise ValueError('should have at least 2 values')

    maxdiff = lst[1]-lst[0]
    minval = min(lst[0],lst[1])
    for i in range(2,len(lst)):
        val = lst[i]
        diff = val-minval
        if diff>maxdiff:
            maxdiff = diff
        if val<minval:
            minval = val
    return maxdiff

def test_mp():
    assert max_profit([10, 7, 5, 8, 11, 9])==6

"""
You have a list of integers, and for each index you want
to find the product of every integer except the integer
at that index
"""
def prod_except_at_index(lst):
    left = [1]
    right = [1]
    for i in range(1,len(lst)):
        left.append(lst[i-1]*left[-1])
    for i in range(len(lst)-2,-1,-1):
        right.append(lst[i+1]*right[-1])
    return [left[i]*right[len(lst)-i-1] for i in range(len(lst))]

def test_peai():
    assert prod_except_at_index([1, 7, 3, 4])==[84, 12, 28, 21]

"""
Given a list of integers, find the highest product you
can get from three of the integers
"""
def highest_3prod(lst):
    highest = max(lst[0],lst[1])
    lowest = min(lst[0], lst[1])
    highest2 = lst[0]*lst[1]
    lowest2 = highest2
    highest3 = highest2*lst[2]
    for i in range(2,len(lst)):
        val = lst[i]
        prod3 = max(highest2*val,lowest2*val)
        if prod3>highest3:
            print(highest3, 'highest3', prod3)
            highest3 = prod3
        prod2 = max(highest*val,lowest*val)
        prod2_lo = min(highest*val,lowest*val)
        if prod2>highest2:
            print(highest2, 'highest2', prod2)
            highest2 = prod2
        if prod2_lo<lowest2:
            print(lowest2, 'lowest2', prod2_lo)
            lowest2 = prod2_lo
        if val>highest:
            print(val, 'highest', val)
            highest = val
        if val<lowest:
            print(val, 'lowest', val)
            lowest = val
    return highest3

assert highest_3prod([1, 10, -5, 1, -100])==5000, highest_3prod([1, 10, -5, 1, -100])

"""
Find the unique ID among other IDs which have 2 copies
"""
def find_unique(lst):
    unique = 0
    for i in lst:
        unique ^= i
    return unique

def test_fu():
    assert find_unique([3234134,990,3234134,33345, \
                        990,234,234])==33345

"""
check if a singly-linked list has a cycle
"""
def contain_cycle(node):
    if node==None:
        return False
    i = 0
    slowrunner = node
    fastrunner = node.next
    while fastrunner:
        if fastrunner==slowrunner:
            return True
        fastrunner = fastrunner.next
        if i%2==0:
            slowrunner = slowrunner.next
        i +=1
    return False

"""
reverse words in a string
"""
def reverse_words(string):
    def reverse_char(w):
        new = [None]*len(w)
        n = len(w)-1
        i = 0
        while i<=n-i:
            new[i],new[n-i] = w[n-i],w[i]
            i += 1
        return ''.join(new)
    string = reverse_char(string)
    lst = string.split()
    for i in range(len(lst)):
        lst[i] = reverse_char(lst[i])
    return ' '.join(lst)

assert reverse_words('secret team solving codes')== \
                    'codes solving team secret'

"""
return all permutations of a string (no duplicates in it)
"""
def string_permutations(string):
    if len(string)<=1:
        return set([string])
    perm = string_permutations(string[:-1])
    allperm = set()
    for p in perm:
        for i in range(len(string)):
            allperm.add(p[:i]+string[-1]+p[i:])
    return allperm

assert string_permutations('cat')==\
    set(['cat','cta','atc','act','tac','tca'])

"""
in-place shuffle a list uniformly, meaning each item in the
original list must have the same probability of ending up in
each spot in the final list
"""
def inplace_shuffle(lst):
    import random
    if len(lst)<=1:
        return lst
    for i in range(len(lst)-1):
        random_index = random.randrange(i,len(lst))
        if random_index!=i:
            lst[i],lst[random_index]=lst[random_index],lst[i]
    return lst

print(inplace_shuffle([1,3,2,56,23,555]))

