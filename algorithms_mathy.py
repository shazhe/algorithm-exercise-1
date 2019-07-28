
################################# sum | hash or recursion #################################

"""
Find 2 numbers summed up to an amount.
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

assert sum2(13,[2,3,4,12])==False
assert sum2(12,[1,3,4,6,8])==(4,8)
assert sum2(6, [1, 1, 2, 3, 5]) == (1, 5)

"""
Find all distinct 3-numbers summed up to an amount.
"""
def sum3(amount, lst):
    ret = []
    n = len(lst)
    lst = sorted(lst)
    for i in range(n - 1):
        x = lst[i]
        l = i + 1
        r = n - 1
        seen = []
        while l < r:
            if x + lst[l] + lst[r] == amount:
                if lst[l] not in seen:
                    ret.append([x, lst[l], lst[r]])
                l += 1
                r -= 1
            elif x + lst[l] + lst[r] < amount:
                l += 1
            else:
                r -= 1
    return ret

assert sum3(2, [-1, 0, 1, 1, 1, 2, 3]) == [[-1, 0, 3], [-1, 1, 2], [0, 1, 1]]

"""
Find all 4-numbers summed up to an amount.
"""
def sum4(amount, lst):
    dic = {}
    for n in lst:
        if n in dic:
            dic[n] += 1
        else:
            dic[n] = 1

    n = len(lst)
    lst = sorted(lst)
    twosums = {}
    for i in range(n - 1):
        for j in range(i + 1, n):
            s = lst[i] + lst[j]
            if s in twosums:
                twosums[s].append([lst[i], lst[j]])
            else:
                twosums[s] = [[lst[i], lst[j]]]

    ks = list(twosums.keys())
    ks = sorted(ks)
    l = 0
    r = len(ks) - 1
    ret = []
    while l < r:
        if ks[l] + ks[r] == amount:
            twosuml = twosums[ks[l]]
            twosumr = twosums[ks[r]]

            for (l1, l2) in twosuml:
                for (r1, r2) in twosumr:
                    if (l1 == r1) and (dic[l1] < 2):
                        continue
                    if (l2 == r1) and (dic[l2] < 2):
                        continue
                    if (l2 == r2) and (dic[l2] < 2):
                        continue
                    if (l1 == l2 == r1) and (dic[l1] < 3):
                        continue
                    if (l2 == r1 == r2) and (dic[l2] < 3):
                        continue
                    if (l1 == l2 == r1 == r2) and (dic[l1] < 4):
                        continue
                    if ks[l] == ks[r]:
                        if l1 >= r1:
                            continue
                    if l2 < r1: # to ensure no repitition
                        continue
                    ret.append([l1, l2, r1, r2])
                    #ret.append(sorted([l1, l2, r1, r2]))

            l += 1
            r -= 1

        elif ks[l] + ks[r] < amount:
            l += 1

        else:
            r -= 1

    return ret

print(sum4(2, [-1, 0, 1, 2, 3]))
print(sum4(2, [-1, 0, 1, 1, 1, 2, 3]))

"""
Check if there are 5 numbers summed up to an amount.
"""
def sum5(amount, lst):
    dic = {}
    for n in lst:
        if n not in dic:
            dic[n] = 1
        else:
            dic[n] += 1

    for n in lst:
        res = amount - n
        lcp = lst.copy()
        lcp.remove(res)
        is_sum4 = sum4(res, lcp) != []
        if is_sum4:
            return True
    return False

"""
Compute fibonacci.
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

assert [fib(n) for n in range(5)]==[0,1,1,2,3]

"""AKN
Check whether a number x is a sum of n<=5 distinct fibonacci numbers.
If it helps: https://www.geeksforgeeks.org/interesting-facts-fibonacci-numbers/
"""
def fibonacci_sum(x, n):
    f = 1
    i = 2
    fibs = []
    while f < x:
        f = fib(i)
        i += 1
        if f <= x:
            fibs.append(f)

    print(fibs)
    print(i)

    if n == 1:
        return x in fibs

    if n == 2:
        return sum2(x, fibs) != False

    if n == 3:
        return sum3(x, fibs) != []

    if n == 4:
        return sum4(x, fibs) != []

    if n == 5:
        return sum5(x, fibs) != False

    return False

assert fibonacci_sum(6, 2)
assert fibonacci_sum(5, 3) == False
assert fibonacci_sum(10059560, 4)

################################# knapsack #################################

"""
0-1 knapsack problem: find maximum value that can be put in a knapsack of capacity W.
"""
def knapsack(W, weights, values):
    n = len(weights)
    assert n == len(values)

    memo = {} # memo[(i, w)] = value of knapsack of capacity w containing some of the first i-1 items
    for i in range(n + 1): # the first i items considered
        for w in range(W + 1): # capacity of knapsack
            if i == 0 or w == 0:
                memo[(i, w)] = 0
            elif weights[i - 1] <= w:
                memo[(i, w)] = max(values[i - 1] + memo[(i - 1, w - weights[i - 1])], # value of the i-1 item in knapsack + max value of an hypothesis knapsack of capacity w - weights[i - 1]
                                   memo[(i - 1, w)]) # the i-1 item is not in knapsack of capacity w
            else:
                memo[(i, w)] = memo[(i - 1, w)]
    return memo[(n, W)]

assert knapsack(50, [10, 20, 30], [60, 100, 120]) == 220


################################# maximum scores #################################

"""
Step through an array of n numbers, with step size at most k. Collect score of the
number stepped on. Find maximum score.
"""

"""
Exchange apples for scores. 1 apple + x cents = 1 score. 1 apple = y cents.
Start with n apples m cents. Find maximum scores.
"""


################################# minimum cost painting | 1D #################################

"""
Paint n houses in a row with m colors, adjacent ones with different colors.
A cost (n x m) gives the cost of painting each color for each house. Find minimum cost.
"""

################################# minimum cost traversal | non-planar #################################

"""
Given an adjacent matrix of directional traversal cost, and the original node.
Find the minimum cost of travelling all nodes.
"""

################################# minima height | 3D #################################

"""
Find minima of a matrix of values representing heigths on 2D.
"""

################################# escape grid | 2D #################################

"""
Escape a square grid, with 0, 1, -1. On 1, can move 4 directions;
on 0, through original direction; hit -1, game over. Given starting
positions of me and a puppy to help (both on 1).
"""

################################# GCD #################################

"""
Find gcd of two numbers.
"""
def gcd(a, b):
    if b == 0:
        return a
    return gcd(b, a%b)

assert gcd(98, 56) == 14

"""
Find gcd of an array.
"""
def gcd_array(lst):
    if len(lst) == 1:
        return lst[0]
    ret = gcd(lst[0], lst[1])
    for i in range(2, len(lst)):
        ret = gcd(ret, lst[i])
    return ret

assert gcd_array([2, 4, 6, 8, 16]) == 2

"""
Find minimum partition of an array such that each pair has gcd at most 1 inside a group.
"""

################################# geometry #################################

"""
Given n distinct lattice points, check if there are three in the same line.
"""
def colinear(points):
    assert len(points) >= 3
    # TODO

#assert colinear([(-1, -1), (0, 0), (1, 1)])
#assert colinear([(-1, -1), (0, 0), (1, 2)]) == False

