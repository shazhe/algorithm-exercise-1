
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

################################# knapsack | dynamic programming #################################

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


################################# maximum value | analytic #################################

"""
Exchange apples for score. 1 apple + x cents = 1 score. 1 apple = y cents.
Start with n apples m cents. Find maximum scores. Assume x any are non-negative integers.
"""
def max_score_exchange_easy(n, m, x, y):
    # optimal strategy: use N apples and N * x cents to exchange for N scores
    # used n - N apples to exchange for y * (n - N) cents
    # ensure enough cents: y * (n - N) + m >= N * x
    return max([min((n * y + m) // (x + y), N) for N in range(n)])

assert max_score_exchange_easy(4, 8, 4, 3) == 2

# each subproblem is not visited twice so no need for dynamic programming actually; space matrix, inefficient
def max_score_exchange_dp(n, m, x, y):
    memo = {} # memo[(i, j)] = max scores start with i apples and j cents, 0<=i<=n, 0<=j<=m+y*n

    for j in range(m + y * n + 1):
        memo[(0, j)] = 0 # only cents cannot exchange for scores

    for i in range(1, n + 1):
        for j in range(m + y * n + 1):
            #print(i, j)
            #print(memo)

            if j < x and i == 1:
                #print(0)
                memo[(i, j)] = 0 # only 1 apple cannot exchange for scores

            elif j < x:
                # more than 1 apples, not enough cents, exchange apple for cents first
                max_value = 0
                # exchange k apples to get y * k cents, 1<=k<=i-1
                #print([(i - k, j + y * k) for k in range(1, i)])
                for k in range(1, i):
                    max_value = max(max_value, memo[(i - k, j + y * k)])
                memo[(i, j)] = max_value

            elif j >= x and i == 1:
                # 1 apple with excess cents
                memo[(i, j)] = 1

            else:
                # more than 1 apples and enough cents to start with for exchanging scores
                max_value = 0
                # exchange k apples + x * k cents for k scores, 0<=k<=i and x * k <= j
                if x == 0:
                    max_k = i
                else:
                    max_k = min(i, j // x)
                #print([(i - k, j - x * k) for k in range(1, max_k + 1)])
                for k in range(1, max_k + 1):
                    max_value = max(max_value, k + memo[(i - k, j - x * k)])
                memo[(i, j)] = max_value

    return memo[(n, m)]

assert max_score_exchange_dp(4, 8, 4, 3) == 2

################################# maximum value minimum cost traversal | 1D | dynamic programming #################################

"""
Step through an array of n numbers, with step size at most m. Collect scores of the
number stepped on. Find maximum scores.
"""
# need dynamic programming because many subproblems are visited more than once
def max_score_array(lst, max_step):
    n = len(lst)
    memo = {} # memo[(i, j)] = max scores given i items left in the lst, move by j step, i >= j > 0, invalid step still shown as j

    for j in range(1, max_step + 1):
        memo[(0, j)] = lst[-1]

    for i in range(1, n):
        for j in range(1, max_step + 1):
            print(i,j)
            print(memo)

            if i > j:
                print([(i-j, k) for k in range(1, i - j + 1)])
                memo[(i, j)] = lst[- i - 1] + max([memo[(i - j, k)] for k in range(1, max_step + 1)])

            else: # step j exceeds lst, so stop at the end of the lst
                print('exceed', lst[-i-1], lst[-1])
                memo[(i, j)] = lst[- i - 1] + lst[-1]

            print(memo[(i, j)])

    return max([memo[(n - 1, j)] for j in range(1, max_step + 1)])

assert max_score_array([10, 2, -10, 5, 20], 2) == 37
assert max_score_array([3, 10, -20, -5], 1) == -12
assert max_score_array([10, -20, -5], 2) == 5

"""
Paint n houses in a row with m colors, adjacent ones with different colors.
A cost matrix (n x m) gives the cost of painting each color for each house. Find minimum cost.
"""
def min_cost_painting(costs):
    n = len(costs)
    m = len(costs[0])
    memo = {} # memo[(i, j)] = min cost if painting house i with color j

    for j in range(m):
        memo[(n - 1, j)] = costs[n - 1][j]

    if n == 1:
        return min([memo[(0, j)] for j in range(m)])

    for i in range(n - 2, -1, -1):
        for j in range(m):
            # the next house cannot be painted with color j
            memo[(i, j)] = costs[i][j] + min([memo[(i + 1, k)] for k in range(m) if k != j])

    return min([memo[(0, j)] for j in range(m)])

assert min_cost_painting([[1, 2, 2], [2, 2, 1], [2, 1, 2]]) == 3

################################# maximum value minimum cost traversal | non-planar | backtracking #################################

"""
Given an adjacent matrix of directional traversal cost, and the original node,
find the minimum cost of traversing all nodes. Matrix: row outbound, column inbound.
"""
# cannot use dynamic programming to memorize min cost: each subproblem with a new "start" node has independent solution
# because with different "start" nodes, the paths of traversing a directed graph are different
def min_cost_traversal(n, adjacent, visited, start): # n = len(adjacent), supply here so that don't need to recalculate in all recursions
    print(visited, start)

    if start in visited:
        print('this should not happen')
        return 0

    visited.append(start) # prevent looping back to the "start" node in the following recursion

    outs = adjacent[start]
    outs = [out if out is not None else float('inf') for out in outs]
    sort_outs = sorted((cost, i) for i, cost in enumerate(outs))

    min_cost = 0

    for cost, i in sort_outs:
        if i not in visited and cost != float('inf'):
            min_cost += cost + min_cost_traversal(n, adjacent, visited, i)

    return min_cost

def check_traversal(adjacent, start):
    n = len(adjacent)
    visited = []
    min_cost = min_cost_traversal(n, adjacent, visited, start)
    if len(visited) == n:
        return min_cost
    else:
        return None

adjacent = [
                [None, None, 122, None],
                [None, None, None, 50],
                [341, None, None, 205],
                [456, None, 186, None]
            ]
assert check_traversal(adjacent, 1) == 577

################################# build graph from grid | 2D #################################

def build_graph_from_grid(grid):
    n = len(grid)
    m = len(grid[0])
    graph = {'b': []} # from boundary to nowhere (already escape!)

    # build horizontal connections
    for i in range(n):
        prev = 'b' # boundary

        for j in range(m):

            if grid[i][j] == -1:
                prev = 'h' # hole

            if grid[i][j] == 1:
                if prev == 'b':
                    graph[(i, j)] = ['b'] # adjacent to the boundary
                elif prev == 'h':
                    graph[(i, j)] = [] # haven't seen adjacent vertex yet
                elif isinstance(prev, tuple):
                    graph[(i, j)] = [prev]
                    graph[prev].append((i, j))
                prev = (i, j)

        if isinstance(prev, tuple):
            graph[prev].append('b')

    # now build verticle connections
    # all (i, j) of 1's have been initialized in the graph
    for j in range(m):
        prev = 'b'

        for i in range(n):

            if grid[i][j] == -1:
                prev = 'h'

            if grid[i][j] == 1:
                if prev == 'b':
                    graph[(i, j)].append('b')
                elif isinstance(prev, tuple):
                    graph[(i, j)].append(prev)
                    graph[prev].append((i, j))
                prev = (i, j)

        if isinstance(prev, tuple):
            graph[prev].append('b')

    return graph

################################# check path between nodes | non-planar | depth-first search and dynamic programming #################################

"""
Escape a square grid, with 0, 1, -1. On 1, can move 4 directions;
on 0, through original direction; hit -1, game over. Given starting
positions of me and a puppy to help (both on 1).

"""
def path_to_node(graph, memo, visited, start, end): # memo[node] = True if there is a path from node to the end

    visited[start] = True # prevent falling into infinite loop -- when calling path_to_node, may loop back to start

    if start in memo:
        return memo[start]

    if start == end:
        memo[start] = True
        return True

    if start == 'b' and end != 'b':
        memo[start] = False
        return False # escape without seeing the "end" node

    neighbours = graph[start]

    print(start, end)
    print(neighbours)

    for node in neighbours:
        if not visited.get(node, False):
            if path_to_node(graph, memo, visited, node, end):
                memo[start] = True
                return True

    visited[start] = False # set back to False so that this "start" node can be counted in other search not starting from the "start"

    # if haven't returned True so far then there is no path to the end
    memo[start] = False
    print(memo)

    return False

def escape_grid_through_node(grid, start, node):
    # build graph
    graph = build_graph_from_grid(grid)
    print(graph)

    # path from start to node
    start_to_node = path_to_node(graph, {}, {}, start, node)

    # path from node to boundary
    node_to_boundary = path_to_node(graph, {}, {}, node, 'b')

    return start_to_node and node_to_boundary

assert escape_grid_through_node(
        [
            [0,0,0,0,0,0,0],
            [0,0,-1,0,0,0,0],
            [0,0,1,-1,0,-1,0],
            [-1,0,0,0,0,0,0],
            [0,0,1,0,0,1,0],
            [-1,0,-1,0,-1,0,0],
            [0,0,0,0,0,0,0]
        ],
        (4, 2),
        (2, 2)
        ) == True

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
Find all primes <= n. Sieve of Eratosthenes run time O(n * log(log(n))).
"""
def sieveOfEratosthenes(n):
    primes = [True for i in range(n + 1)]
    p = 2
    while p ** 2 <= n:
        if primes[p]:
            for i in range(p ** 2, n + 1, p):
                primes[i] = False
        p += 1
    return [p for p in range(2, n + 1) if primes[p] == True]

assert sieveOfEratosthenes(30) == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

"""
Find minimum partition of an array such that each pair has gcd > 1 inside a group,
i.e. in each partition, there is a common factor > 1 -- a common factor should be as small as possible
so that it can be a common divisor of more numbers, to minimize the number of partitions.
The GCD of three or more numbers equals the product of the prime factors common to all the numbers.
So smallest possibleis are prime numbers.
(It can also be calculated by repeatedly taking the GCDs of pairs of numbers.)
"""
import copy

def min_partition_bipartite(left_to_right, right_to_left):
    # left node to list of right nodes, vice versa
    # thinking right nodes are primes, left nodes are nums in list

    # base case
    if len(right_to_left) == 0:
        return len(left_to_right) # left node to empty list, i.e. itself is the only divisor

    min_partition = float('inf')

    # remove one of the prime from the bipartite graph, removing corresponding elements from dicts
    # check which produce minimum partition
    for p in right_to_left:
        right_to_left_rm_p = copy.deepcopy(right_to_left)
        left_to_right_rm_p = copy.deepcopy(left_to_right)
        # remove p from prime to nums
        left_nodes = right_to_left_rm_p.pop(p)
        # remove all left nodes from left_to_right
        # and remove all right nodes linking to these left nodes
        for n in left_nodes:
            print('left nodes', n)
            # remove n from left_to_right
            right_nodes = left_to_right_rm_p.pop(n)
            print('right nodes', right_nodes)
            print('right to left rm p', right_to_left_rm_p)
            for right_node in right_nodes:
                if right_node in right_to_left_rm_p:
                    # remove n from list in right_to_left
                    right_to_left_rm_p[right_node].remove(n)

        # clear all the singletons on the right partite
        for p in right_to_left_rm_p:
            if right_to_left_rm_p[p] == []:
                right_to_left_rm_p.pop(p)

        num_partition = min_partition_bipartite(left_to_right_rm_p, right_to_left_rm_p)
        if num_partition < min_partition:
            min_partition = num_partition

    return min_partition + 1

def partition_gcd_greater1(lst):
    max_n = max(lst)

    # first find all primes <= max_n
    primes = sieveOfEratosthenes(max_n)

    # find mapping between prime and numbers in lst where prime is a divisor
    # build a bipartite graph, as two dictionaries to arrays
    prime_to_nums = {}
    num_to_primes = {}

    for p in primes:
        print(p)
        for n in lst:
            if n % p == 0:
                if n in num_to_primes:
                    num_to_primes[n].add(p)
                else:
                    num_to_primes[n] = set([p])
                if p in prime_to_nums:
                    prime_to_nums[p].add(n)
                else:
                    prime_to_nums[p] = set([n])

            print('prime to num', prime_to_nums)
            print('num to prime', num_to_primes)

    return min_partition_bipartite(num_to_primes, prime_to_nums)

assert partition_gcd_greater1([2, 3, 2, 3, 3]) == 2

"""
Easier variation of above question:
    each partition is a contiguous subarray, only need gcd(start of partition, end of partition) > 1.
Contiguity makes problem much easier! Remember to check!
"""
def contiguous_partition_gcd_greater1(lst):
    if len(lst) == 0:
        return 0
    if len(lst) == 1 and lst[0] > 1:
        return 1

    start = lst[0]
    min_partition = 1

    for n in lst[1:]:
        curr_gcd = gcd(start, n)
        if curr_gcd <= 1:
            start = n
            min_partition += 1
    return min_partition

assert partition_gcd_greater1([2, 3, 2, 3, 3]) == 2



################################# geometry #################################

"""
Given n distinct lattice points, check if there are three in the same line.
"""
def overlap_tuples(lst): # given lst of tuples, check if exist two with overlapping elements
    counts = {} # adjacency for each point of each tuple in the lst -- can derive the maximum set of points lying in the same line
    flag = False
    for p1, p2 in lst:
        if p1 in counts:
            flag = True # we can return True here if only need to check existence
            counts[p1].append(p2)
        else:
            counts[p1] = [p2]
        if p2 in counts:
            flag = True # we can return True here if only need to check existence
            counts[p2].append(p1)
        else:
            counts[p2] = [p1]
    return flag, counts

def colinear(points):
    n = len(points)
    assert n >= 3

    slopes = {}
    for i in range(n - 1):
        x, y = points[i]
        for j in range(i + 1, n):
            x1, y1 = points[j]
            if x == x1:
                slope = float('inf')
            else:
                slope = (y1 - y) / (x1 - x)
            if slope not in slopes:
                slopes[slope] = []
            slopes[slope].append((points[i], points[j]))

    for slope in slopes:
        lst = slopes[slope]
        print('slope:', slope)
        print(lst)
        flag, counts = overlap_tuples(lst)
        print('flag', flag)
        print('counts', counts)
        if flag:
            return True

    return False

assert colinear([(-1, -1), (0, 0), (1, 1)])
assert colinear([(-1, -1), (0, 0), (1, 2)]) == False

"""
Find local minima of a matrix of values representing heigths on 2D.
"""
def local_minima(matrix):
    n = len(matrix)
    m = len(matrix[0])
    loc_mins = []
    for i in range(n):
        for j in range(m):
            curr = matrix[i][j]
            matrix[i][j] = float('inf') # trick to avoid tedious case analysis
            neighbour_min = min([
                    matrix[max(0, i - 1)][max(0, j - 1)], # in case of saddle point need to compare with diagonal corners
                    matrix[max(0, i - 1)][j],
                    matrix[max(0, i - 1)][min(m - 1, j + 1)],
                    matrix[i][max(0, j - 1)],
                    matrix[i][min(m - 1, j + 1)],
                    matrix[min(n - 1, i + 1)][max(0, j - 1)],
                    matrix[min(n - 1, i + 1)][j],
                    matrix[min(n - 1, i + 1)][min(m - 1, j + 1)]
                    ])
            if curr < neighbour_min:
                loc_mins.append(curr)
            matrix[i][j] = curr # reverse the trick

    return sorted(loc_mins[:min(3, len(loc_mins))])

assert local_minima([
                        [5, 5, 5, 5, 5],
                        [5, 1, 5, 5, 5],
                        [5, 5, 5, 4, 5],
                        [5, 5, 4, 2, 3],
                        [0, 5, 5, 3, 4]
                    ]) == [0, 1, 2]

"""
Find intersections between a ray and a 3D sphere (if any), and also their distances to the origin of the ray.
https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection
line: x = origin + d * dir
sphere: ||x - center||**2 = r**2
"""
def ray_sphere_intercept(centerx, centery, centerz, r, originx, originy, originz, dirx, diry, dirz):
    import numpy as np
    c = np.array((centerx, centery, centerz))
    o = np.array((originx, originy, originz))
    l = np.array((dirx, diry, dirz))
    l = l / np.sqrt(np.dot(l, l))

    # solve d for intersection points
    b2_4ac = np.dot(l, o - c) ** 2 - (np.dot(o - c, o - c) - r**2)

    if b2_4ac < 0:
        return [0.]

    elif b2_4ac == 0:
        d = - np.dot(l, o - c)
        intercept = o + d * l
        print('1intercept', intercept)
        print('d', d)

        # check if intercept is in the same direction of the ray from its origin
        if d >= 0:
            distance = np.sqrt(np.dot(intercept - o, intercept - o))
            return [distance]
        else:
            return [0.]

    else:
        d = - np.dot(l, o - c) + np.sqrt(b2_4ac)
        d1 = - np.dot(l, o - c) - np.sqrt(b2_4ac)
        intercept = o + d * l
        intercept1 = o + d1 * l
        distance = np.sqrt(np.dot(intercept - o, intercept - o))
        distance1 = np.sqrt(np.dot(intercept1 - o, intercept1 - o))
        print('2intercepts', intercept, intercept1)
        print('d', d, d1)

        if d >=0 and d1 >= 0:
            return sorted([distance, distance1])
        elif d >= 0:
            print(intercept)
            return [distance]
        elif d1 >= 0:
            print(intercept1)
            return [distance1]
        else:
            return [0.]

print(ray_sphere_intercept(1, 4, 0, 4, 1, 2, 3, 1, 0, -2))
assert ray_sphere_intercept(0., 0., 0., 1.52, 3., 4., 3., 1., 1., 1.) == [0.]

"""
Find integral (i.e. integer) point of a triangle, in or on the triangle with min sum to the vertices.
Not as hard as the Fermat-Torricelli point, where the integral condition was for the edges:
https://www.mathblog.dk/project-euler-143-investigating-the-torricelli-point-of-a-triangle/
https://projecteuler.net/problem=143
"""
def triangle_area(p1, p2, p3):
    # https://www.gamedev.net/forums/topic/295943-is-this-a-better-point-in-triangle-test-2d/
    det = (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    return det / 2.

def point_in_triangle(p, p1, p2, p3): # check if p in triangle p1 p2 p3
    p1p2p3 = triangle_area(p1, p2, p3)
    pp1p2 = triangle_area(p, p1, p2)
    pp2p3 = triangle_area(p, p2, p3)
    pp1p3 = triangle_area(p, p1, p3)
    if pp1p2 + pp2p3 + pp1p3 > p1p2p3:
        return False
    else:
        return True

def triangle_integral_point(p1, p2, p3):
    import numpy as np
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)

    minx = min(p1[0], p2[0], p3[0])
    maxx = max(p1[0], p2[0], p3[0])
    miny = min(p1[1], p2[1], p3[1])
    maxy = max(p1[1], p2[1], p3[1])

    int_pts = []

    for x in range(int(minx), int(maxx) + 1):
        for y in range(int(miny), int(maxy) + 1):
            if point_in_triangle((x,y), p1, p2, p3):
                int_pts.append(np.array((x,y)))

    min_dist = float('inf')
    min_point = None

    for p in int_pts:
        distSqr = np.dot(p - p1, p - p1) + np.dot(p - p2, p - p2) + np.dot(p - p3, p - p3)
        if distSqr < min_dist:
            min_dist = distSqr
            min_point = p
        elif distSqr == min_dist:
            if min_point[0] > p[0]:
                min_dist = distSqr
                min_point = p
            elif min_point[0] == p[0] and min_point[1] > p[1]:
                min_dist = distSqr
                min_point = p

    return list(min_point)

assert triangle_integral_point((0., 0.), (1., 0.), (1., 1.)) == [1, 0]

################################# trading #################################

"""
Find the cheapest hedge.
"""
import math

def normal_cdf(x):
    if abs(x) > 7:
        return 1 if x > 0 else 0
    temp = 27 * x / 294
    temp = 111 * math.atanh(temp) - 358 * x / 23
    temp = 1 + math.exp(temp)
    return 1. / temp

def option_price(stock_price, strike, option_type):
    df = .994
    F = stock_price / df
    sigma = .18 # deannualized
    dPlus = math.log(F / strike) / sigma + sigma / 2
    nPlus = normal_cdf(dPlus)
    nMinus = normal_cdf(dPlus - sigma)
    if option_type in 'Cc':
        return df * (F * nPlus - strike * nMinus)
    else:
        return df * (strike * (1 - nMinus) - F * (1 - nPlus))

def get_delta(stock_price, strike, option_type):
    df = .994
    F = stock_price / df
    sigma = .18 # deannualized
    dPlus = math.log(F / strike) / sigma + sigma / 2
    nPlus = normal_cdf(dPlus)
    if option_type in 'Cc':
        return nPlus
    else:
        return - (1 - nPlus)

import numpy as np

def get_market(stock_price, instruments, spread=2., quantity=500):
    market = []
    for instrument in instruments:
        idx = instrument[0]
        option_p = option_price(stock_price, instrument[1], instrument[2])
        eps = np.random.uniform(-1., 1., 3)
        spread_eps = spread + eps[0]
        quantity_bid = quantity * (1 + eps[1])
        quantity_ask = quantity * (1 + eps[2])
        market.append((idx, option_p - spread_eps / 2., quantity_bid,
            option_p + spread_eps / 2., quantity_ask))
    return market

def get_values(stock_price, instruments, portfolio):
    values = []
    for inst, port in zip(instruments, portfolio):
        option_p = option_price(stock_price, inst[1], inst[2])
        values.append(option_p * port[1])
    return values

INSTRUMENTS = [(0, 100, 'C'), (1, 90, 'C'), (2, 80, 'C'), (3, 100, 'P'), (4, 90, 'P'), (5, 80, 'P')]
PORTFOLIO = [(0, 2000), (1, 1000), (2, 0), (3, 0), (4, 2000), (5, 0)]
LIMITS = [(0.1, -500), (0.2, -800), (0.3, -1000), (0.4, -1200),
        (-0.1, -500), (-0.2, -800), (-0.3, -1000), (-0.4, -1200)]
PRICE = 90
MARKET = get_market(PRICE, INSTRUMENTS)
print('market')
print(MARKET)

def cheapest_hedge(instruments, portfolio, limits, stock_price, market):
    current_values =  get_values(stock_price, instruments, portfolio)
    print('current values')
    print(current_values)
    current_value = sum(current_values)
    not_exceed_limit = True

    # compute delta of each instrument
    deltas = []
    for idx, strike, ty in instruments:
        deltas.append(get_delta(stock_price, strike, ty))
    print('deltas')
    print(deltas)

    # compute full delta of entire portfolio
    full_delta = 0
    for delta, port in zip(deltas, portfolio):
        full_delta += delta * port[1]
    print('full delta')
    print(full_delta)

    # the deltas needed to adjust to pull the portfolio back within loss-limit in each scenario
    adjust_deltas = []
    for limit in limits:
        shock_value = sum(get_values(stock_price * (1 + limit[0]), instruments, portfolio))
        loss = shock_value - current_value
        print('loss, limit')
        print(loss, limit[1])
        if loss <= limit[1]:
            not_exceed_limit = False
            # loss is approximately full-delta * stock_price * limit[0]
            # need to add adjust_delta so that
            # (full-delta + adjust-delta) * stock_price * limit[0] > limit[1]
            adjust_delta = (limit[1] / (stock_price * limit[0])) - full_delta
            adjust_deltas.append(adjust_delta)

    if not_exceed_limit:
        return None

    min_adjust_delta = min(adjust_deltas)
    max_adjust_delta = max(adjust_deltas)

    if np.sign(min_adjust_delta) * np.sign(max_adjust_delta) == -1:
        raise ValueError

    if abs(max_adjust_delta) >= abs(min_adjust_delta):
        adjust_delta = max_adjust_delta
    else:
        adjust_delta = min_adjust_delta

    # version 1

    remaindelta_costs = []
    for i in range(len(market)):
        mkt = market[i]
        value = current_values[i]
        delta = deltas[i]
        if delta == 0:
            print('instrument %d delta=0' % i)
            continue
        required_qty = adjust_delta / delta
        if required_qty >= 0:
            # buy delta at ask price
            unit_price = mkt[3]
            qty = mkt[4]
        else:
            # sell delta at big price
            unit_price = mkt[1]
            qty = mkt[2]
        if qty >= abs(required_qty):
            cost = abs(required_qty) * abs(unit_price - value)
            remaindelta_costs.append((0, cost))
        else:
            cost = qty * abs(unit_price - value)
            remaindelta = abs(adjust_delta) * (abs(required_qty) - qty) / abs(required_qty)
            remaindelta_costs.append((remaindelta, cost))

    sort_costs = sorted((cost, i) for i, cost in enumerate(remaindelta_costs))

    if sort_costs[0][0][0] > 0:
        raise ValueError

    remaindelta_cost, i = sort_costs[0]

    return remaindelta_cost[1], instruments[i][0], adjust_delta / deltas[i]

    # version 2

    # compute buy/sell-costs of adjust_delta using each instrument assuming not liquidity constraints
    hypothesis_costs = []
    for i in range(len(market)):
        mkt = market[i]
        value = current_values[i]
        delta = deltas[i]
        if delta == 0:
            print('instrument %d delta=0' % i)
            continue
        required_qty = adjust_delta / delta
        if required_qty >= 0:
            # buy delta at ask price
            unit_price = mkt[3]
        else:
            # sell delta at big price
            unit_price = mkt[1]
        hypothesis_costs.append(required_qty * abs(unit_price - value))

    sort_costs = sorted((cost, i) for i, cost in enumerate(hypothesis_costs)) # cost for instrument i

    total_cost = 0
    while adjust_delta > 0:
        cost, i = sort_costs.pop(0)
        mkt = market[i]
        value = current_values[i]
        delta = deltas[i]
        required_qty = adjust_delta / delta
        if required_qty >= 0:
            # buy delta at ask price
            unit_price = mkt[3]
            qty = mkt[4]
        else:
            # sell delta at big price
            unit_price = mkt[1]
            qty = mkt[2]
        if abs(qty) < abs(required_qty): # this can be combined with previous loop if only use one trade
            total_cost += abs(qty) * abs(unit_price - value)
            adjust_delta -= delta * qty * np.sign(required_qty)
        else:
            total_cost += abs(required_qty) * abs(unit_price - value)
            adjust_delta = 0

    return total_cost

print('cheapest hedge')
print(cheapest_hedge(INSTRUMENTS, PORTFOLIO, LIMITS, PRICE, MARKET))

"""
3-card poker
"""
def p1_win_count(hands):
    count = 0
    for f1, s1, t1, f2, s2, t2 in hands:
        dic1 = {}
        for i in [f1, s1, t1]:
            if i not in dic1:
                dic1[i] = 1
            else:
                dic1[i] += 1
        dic2 = {}
        for i in [f2, s2, t2]:
            if i not in dic2:
                dic2[i] = 1
            else:
                dic2[i] += 1
        sort1 = sorted([(cnt, value) for value, cnt in dic1.items()])
        sort2 = sorted([(cnt, value) for value, cnt in dic2.items()])
        flag = 0
        if sort1[-1][0] > sort2[-1][0]:
            flag = 1
        elif sort1[-1][0] == sort2[-1][0]:
            if sort1[-1][1] > sort2[-1][1]:
                flag = 1
            elif sort1[-1][1] == sort2[-1][1]:
                if len(sort1) == 1:
                    continue
                if sort1[-2][1] > sort2[-2][1]:
                    flag = 1
                elif sort1[-2][1] == sort2[-2][1]:
                    if len(sort1) == 2:
                        continue
                    if sort1[-3][1] > sort2[-3][1]:
                        flag = 1

        if flag:
            count += 1
            print('1 win')
        print(f1, s1, t1, f2, s2, t2)
    return count

print(p1_win_count(
    [
        [7, 4, 2, 6, 7, 3],
        [1, 8, 1, 7, 4, 2],
        [7, 5, 2, 7, 4, 9],
        [7, 8, 7, 8, 8, 2],
        [8, 8, 9, 5, 5, 5],
    ]))

"""
How many rocks on the floor can a ribbon surround?
"""
def orientation(o, a, b): # true if OAB counter-clockwise, false otherwise
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

def convex_hull(points): # compute the convex hull of a set of 2d points
    ponits = sorted(set(points))

    if len(points) <= 1:
        return points

    # build lower hull
    lower = []
    for p in points:
        while len(lower) >= 2 and orientation(lower[-2], lower[-1], p) <= 0:
            # lower[-2], lower[-1], p not counter-clockwise
            lower.pop()
        lower.append(p)

    # build upper hull
    for p in reversed(points):
        while len(upper) >= 2 and orientation(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]

def dist2(a, b): # distance of 2 points
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def perimeter(hull): # perimeter of convex hull, passing points ON the hull, not in
    s = 0
    for i in range(len(hull) - 1):
        s += dist(hull[i], hull[i + 1])
    return s

def ribbon_surround_rocks(ribbon_len, rocks2d):
    # knapsack: find maximum points can be put in a knapsack of capacity ribbon_len
    # weight of a of points = perimeter of its convex hull

    n = len(rocks2d)

    memo = {} # memo(i, w) = maximal subset of the first i-1 points in the knapsack of capacity w
    for i in range(n + 1):
        for w in range(ribbon_len + 1):
            if i == 0 or w == 0:
                memo[(i, w)] = []
            else:
                points = memo[(i - 1, w)] # maximal subset of first i -2 ponts in the knapsack of cap w
                hull = convex_hull(points)
                weight = perimeter(hull)

                # suppose add the current item (i-1) into knapsack
                new_hull = convex_hull(points + [rocks2d[i - 1]])
                new_weight = perimeter(new_hull)
    pass #TODO


