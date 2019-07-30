
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
Find minimum partition of an array such that each pair has gcd at most 1 inside a group.
"""

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
"""


"""
Find integral point of a triangle, in or on the triangle with min sum to the vertices.
"""



