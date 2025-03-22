def get_powerset(arr, index):
    all_subsets = []
    if index == len(arr):
        all_subsets.append([])
    else:
        all_subsets = get_powerset(arr, index + 1)
        item = arr[index]
        more_subsets = []
        for subset in all_subsets:
            new_subset = [item]
            new_subset.extend(subset)
            more_subsets.append(new_subset)
        all_subsets.extend(more_subsets)
    return all_subsets


def tower_of_hanoi(n, source, destination, auxiliary):
    if n == 1:
        print(f"Move disk {n} from {source} to {destination}")
        return
    tower_of_hanoi(n-1,source,auxiliary,destination)
    print(f"Move disk {n} from {source} to {destination}")
    tower_of_hanoi(n-1,auxiliary,destination,source)
    
    
def magic_index(arr, start, end):
    if start > end:
        return -1
    mid = (start + end) // 2
    if arr[mid] == mid:
        return mid
    left_index = min(mid-1,arr[mid])
    left = magic_index(arr,start,left_index)
    if left >= 0:
        return left
    right_index = max(mid+1,arr[mid])
    right = magic_index(arr,right_index,end)
    return right

def robot_in_a_grid(grid):
    if not grid:
        return []
    path = []
    failed_paths = set()
    if robot_in_a_grid_helper(grid,len(grid)-1,len(grid[0])-1,path,failed_paths):
        return path
    return []

def robot_in_a_grid_helper(grid,row,col,path,failed_paths):
    if row < 0 or col < 0 or row >= len(grid) or col >= len(grid[0]):
        return False
    point = (row,col)   
    if point in failed_paths:
        return False
    is_origin = (row == 0 and col == 0)
    if is_origin or robot_in_a_grid_helper(grid,row-1,col,path,failed_paths) or robot_in_a_grid_helper(grid,row,col-1,path,failed_paths):
        path.append(point)
        return True
    failed_paths.add(point)
    return False

def count_ways(n,memo):
    if n < 0:
        return 0
    if n == 0:
        return 1
    if memo[n] != -1:
        return memo[n]
    memo[n] = count_ways(n-1,memo) + count_ways(n-2,memo) + count_ways(n-3,memo)
    return memo[n]

def min_product(a,b):
    smaller = a if a < b else b
    bigger = a if a > b else b
    return min_product_helper(smaller,bigger)

def min_product_helper(smaller,bigger):
    if smaller == 0:
        return 0
    if smaller == 1:
        return bigger
    s = smaller >> 1
    half_prod = min_product_helper(s,bigger)
    if smaller%2==0:
        return half_prod+half_prod
    else:
        return half_prod+half_prod+bigger
    
    
def permutation(str):
    if len(str) == 0:
        return []
    permutations = set()
    permutation_helper(str,0,permutations)
    return permutations

def permutation_helper(str,index,permutations):
    if index == len(str):
        permutations.add("".join(str))
    else:
        for i in range(index,len(str)):
            str[index],str[i] = str[i],str[index]
            permutation_helper(str,index+1,permutations)
            str[index],str[i] = str[i],str[index]


class Solution:
    def generateParenthesis(self, n: int):
        res = []

        def dfs(openP, closeP, s):
            if openP == closeP and openP + closeP == n * 2:
                res.append(s)
                return
            
            if openP < n:
                dfs(openP + 1, closeP, s + "(")
            
            if closeP < openP:
                dfs(openP, closeP + 1, s + ")")

        dfs(0, 0, "")

        return res   
    

def paint_fill(screen,r,c,color):
    if screen[r][c] == color:
        return False
    return paint_fill_helper(screen,r,c,color,screen[r][c])

def paint_fill_helper(screen,r,c,new_color,original_color):
    if r < 0 or r >= len(screen) or c < 0 or c >= len(screen[0]):
        return False
    
    if screen[r][c] == original_color:
        screen[r][c] = new_color
        paint_fill_helper(screen,r-1,c,new_color,original_color)
        paint_fill_helper(screen,r+1,c,new_color,original_color)
        paint_fill_helper(screen,r,c-1,new_color,original_color)
        paint_fill_helper(screen,r,c+1,new_color,original_color)
    return True

def make_change(n,list_coins):
    if n<0:
        return 0
    def count_ways(amount,list_coins,index,memo):
        if memo[amount][index]>0:
            return memo[amount][index]
        if index>=len(list_coins)-1:
            return 1
        ways = 0
        for i in range(amount//list_coins[index]+1):
            amount_remaining = amount - i*list_coins[index]
            ways += count_ways(amount_remaining,list_coins,index+1,memo)
        memo[amount][index] = ways
        return ways
    memo = [[-1 for _ in range(len(list_coins))] for _ in range(n+1)]
    return count_ways(n,list_coins,0,memo)
    
print("Tower of Hanoi :")
tower_of_hanoi(3, "A", "C", "B")
print("--------------------------------")
print("Powerset :")
print(get_powerset([1, 2, 3], 0))
print("--------------------------------")
print("Magic Index :")
print(magic_index([-10, -5, 2, 2, 2, 3, 4, 7, 9, 12, 13], 0, 10))
print("--------------------------------")
print("Robot in a Grid :")
print(robot_in_a_grid([[1, 0, 0, 0], [1, 1, 0, 1], [0, 1, 0, 0], [1, 1, 1, 1]]))
print("--------------------------------")
print("Count Ways :")
print(count_ways(3, [-1]*4))
print("--------------------------------")
print("Min Product :")
print(min_product(3,4))
print("--------------------------------")
print("Permutations :")
s = "aaa"
print(permutation(list(s)))
print("--------------------------------")
print("Generate Parenthesis :")
print(Solution().generateParenthesis(3))
print("--------------------------------")
print("Paint Fill :")
screen = [[1, 1, 1], [1, 1, 0], [1, 0, 1]]
paint_fill(screen, 1, 1, 2)
print(screen)
print("--------------------------------")
print("Make Change :")
print(make_change(100,[1,5,10,25]))