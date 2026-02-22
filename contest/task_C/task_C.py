from sys import stdin


def solve():
    n, m = map(int, stdin.readline().split())
    sx, sy = map(int, stdin.readline().split())
    sx -= 1
    sy -= 1
    grid = [stdin.readline().strip() for _ in range(n)]
    s = stdin.readline().strip()
    
    INF = 10**9
    dp = [[INF] * m for _ in range(n)]
    
    # Initial delivery
    for i in range(n):
        for j in range(m):
            if grid[i][j] == s[0]:
                dp[i][j] = abs(i - sx) + abs(j - sy)
    
    for ch_idx in range(1, len(s)):
        # 1D convolution along rows for y
        temp = [[INF] * m for _ in range(n)]
        for i in range(n):
            row = [dp[i][j] for j in range(m)]
            # left to right
            for j in range(1, m):
                row[j] = min(row[j], row[j-1] + 1)
            # right to left
            for j in range(m-2, -1, -1):
                row[j] = min(row[j], row[j+1] + 1)
            temp[i] = row
        
        # 1D convolution along columns for x
        new_dp = [[INF] * m for _ in range(n)]
        for j in range(m):
            col = [temp[i][j] for i in range(n)]
            # top to bottom
            for i in range(1, n):
                col[i] = min(col[i], col[i-1] + 1)
            # bottom to top
            for i in range(n-2, -1, -1):
                col[i] = min(col[i], col[i+1] + 1)
            for i in range(n):
                new_dp[i][j] = col[i]
        
        # Only cells with letter s[ch_idx] matter
        dp = [[INF] * m for _ in range(n)]
        for i in range(n):
            for j in range(m):
                if grid[i][j] == s[ch_idx]:
                    dp[i][j] = new_dp[i][j]
    
    ans = min(dp[i][j] for i in range(n) for j in range(m) if grid[i][j] == s[-1])
    print(ans)

if __name__ == "__main__":
    solve()