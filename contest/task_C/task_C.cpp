#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <climits>

using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    
    int n, m;
    cin >> n >> m;
    
    int sx, sy;
    cin >> sx >> sy;
    sx--; // Переводим в 0-индексацию
    sy--;
    
    vector<string> grid(n);
    for (int i = 0; i < n; i++) {
        cin >> grid[i];
    }
    
    string s;
    cin >> s;
    
    const int INF = 1e9;
    vector<vector<int>> dp(n, vector<int>(m, INF));
    
    // Первая доставка
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            if (grid[i][j] == s[0]) {
                dp[i][j] = abs(i - sx) + abs(j - sy);
            }
        }
    }
    
    // Для каждой следующей доставки
    for (int ch_idx = 1; ch_idx < s.size(); ch_idx++) {
        // Временная матрица для свертки
        vector<vector<int>> temp(n, vector<int>(m, INF));
        
        // Свертка по строкам (по y)
        for (int i = 0; i < n; i++) {
            vector<int> row(m);
            for (int j = 0; j < m; j++) {
                row[j] = dp[i][j];
            }
            
            // Слева направо
            for (int j = 1; j < m; j++) {
                row[j] = min(row[j], row[j - 1] + 1);
            }
            
            // Справа налево
            for (int j = m - 2; j >= 0; j--) {
                row[j] = min(row[j], row[j + 1] + 1);
            }
            
            temp[i] = row;
        }
        
        // Свертка по столбцам (по x)
        vector<vector<int>> new_dp(n, vector<int>(m, INF));
        for (int j = 0; j < m; j++) {
            vector<int> col(n);
            for (int i = 0; i < n; i++) {
                col[i] = temp[i][j];
            }
            
            // Сверху вниз
            for (int i = 1; i < n; i++) {
                col[i] = min(col[i], col[i - 1] + 1);
            }
            
            // Снизу вверх
            for (int i = n - 2; i >= 0; i--) {
                col[i] = min(col[i], col[i + 1] + 1);
            }
            
            for (int i = 0; i < n; i++) {
                new_dp[i][j] = col[i];
            }
        }
        
        // Обновляем dp только для клеток с текущей буквой
        dp = vector<vector<int>>(n, vector<int>(m, INF));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                if (grid[i][j] == s[ch_idx]) {
                    dp[i][j] = new_dp[i][j];
                }
            }
        }
    }
    
    // Находим минимальное время для последней доставки
    int ans = INF;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            if (grid[i][j] == s.back()) {
                ans = min(ans, dp[i][j]);
            }
        }
    }
    
    cout << ans << endl;
    
    return 0;
}