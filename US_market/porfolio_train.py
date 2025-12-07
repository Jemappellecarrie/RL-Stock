import yfinance as yf
import pandas as pd
import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt

# ==========================================
# 1. 数据准备：对齐多只股票数据
# ==========================================
def load_portfolio_data(tickers, start="2010-01-01", end="2023-12-31"):
    print(f"Downloading data for portfolio: {tickers}")
    
    raw_data = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=False)
    df_close = raw_data["Close"].bfill().ffill()
    df_close.dropna(axis=1, how='all', inplace=True)
    df_close.dropna(axis=0, how='any', inplace=True)

    # === VIX ===
    print("Downloading VIX (^VIX) as Macro Feature...")
    vix = yf.download("^VIX", start=start, end=end, progress=False)["Close"]
    vix = vix.reindex(df_close.index).ffill() / 100.0

    # === 核心金融特征 ===
    df_return = np.log(df_close / df_close.shift(1)).dropna()
    
    # ✅ 趋势特征：20 日动量
    momentum = df_close.pct_change(20).loc[df_return.index]
    
    # ✅ 风险特征：20 日波动率
    volatility = df_return.rolling(20).std().dropna()
    
    # 对齐
    df_return = df_return.loc[volatility.index]
    momentum = momentum.loc[volatility.index]
    vix = vix.loc[volatility.index]

    # 标准化
    df_return = df_return / df_return.std()
    momentum = momentum / momentum.std()
    volatility = volatility / volatility.std()

    # ✅ 拼接最终特征矩阵
    data_matrix = np.hstack([
        df_return.values,
        momentum.values,
        volatility.values,
        vix.values.reshape(-1, 1)
    ])

    prices_matrix = df_close.loc[volatility.index].values
    dates = volatility.index

    print(f"Final Feature Shape: {data_matrix.shape}")
    return data_matrix, prices_matrix, dates

# ==========================================
# 2. 投资组合环境 (Portfolio Environment)
# ==========================================
class PortfolioEnv(gym.Env):
    def __init__(self, return_data, price_data, window_size=50, initial_balance=10000):
        super(PortfolioEnv, self).__init__()

        self.data = return_data
        self.prices = price_data

        self.n_features = self.data.shape[1]
        self.n_stocks = self.prices.shape[1]

        self.window_size = window_size
        self.initial_balance = initial_balance

        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.n_stocks + 1,), dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(window_size, self.n_features),
            dtype=np.float32
        )

    def reset(self):
        self.current_step = self.window_size
        self.portfolio_value = self.initial_balance

        self.current_weights = np.zeros(self.n_stocks + 1)
        self.current_weights[-1] = 1.0

        return self._get_obs()

    def _get_obs(self):
        return self.data[self.current_step - self.window_size : self.current_step]

    # ✅ 趋势友好的 Softmax
    def softmax(self, x, temperature=2.0, min_weight=0.005):
        """
        temperature ↑  -> 更容易出现集中押注（进攻性）
        min_weight ↓   -> 允许“弃掉弱势股”
        """
        x = x / temperature
        e_x = np.exp(x - np.max(x))
        w = e_x / e_x.sum()

        # 允许弱势股真正变成接近 0
        w = np.maximum(w, min_weight)
        return w / w.sum()


    def step(self, action):
        target_weights = self.softmax(action)

        current_prices = self.prices[self.current_step]
        last_prices = self.prices[self.current_step - 1]
        stock_returns = (current_prices / (last_prices + 1e-9)) - 1

        # ✅ 成本按资产数归一化
        turnover = np.sum(
            np.abs(target_weights[:-1] - self.current_weights[:-1])
        ) / self.n_stocks

        transaction_cost = turnover * 0.0005

        gross_return = np.sum(target_weights[:-1] * stock_returns)
        net_return = gross_return - transaction_cost

        self.portfolio_value *= (1 + net_return)


        reward = (
            0.9 * net_return
            - 0.05 * abs(net_return)
            - 0.05 * transaction_cost
        )
        self.current_weights = target_weights
        self.current_step += 1

        done = self.current_step >= len(self.data) - 1
        obs = self._get_obs() if not done else np.zeros(self.observation_space.shape)

        return obs, reward, done, {}

# ==========================================
# 3. 训练与测试流程
# ==========================================
if __name__ == "__main__":
    # --- 1. 策略配置 ---
    # 选取5只低相关性股票构建组合
    tickers = ['NVDA', 'JPM', 'WMT', 'XOM', 'JNJ'] 
    
    # 获取全量数据 (2010 - 2023)
    # 注意：min_rows=2000 保证即使某只股票上市晚一点也能被处理，但最好选老牌股
    full_data, full_prices, full_dates = load_portfolio_data(tickers, start="2010-01-01", end="2023-12-31")
    
    # --- 2. 切分 Train / Test ---
    # 我们按照日期切分，找到 2021-01-01 的索引位置
    split_date = pd.Timestamp("2021-01-01")
    
    # 找到 split_date 在 full_dates 中的位置
    # searchsorted 需要保证 dates 是排序的 (yfinance 下载的通常已排序)
    split_index = np.searchsorted(full_dates, split_date)
    
    print(f"\nTotal Data Points: {len(full_dates)}")
    print(f"Splitting at index: {split_index} (Date: {split_date.date()})")
    
    # 训练集: 2010 ~ 2020
    train_data = full_data[:split_index]
    train_prices = full_prices[:split_index]
    
    # 测试集: 2021 ~ 2023
    test_data = full_data[split_index:]
    test_prices = full_prices[split_index:]
    test_dates = full_dates[split_index:]
    
    print(f"Train Set Shape: {train_data.shape}")
    print(f"Test Set Shape: {test_data.shape}")

    # --- 3. 训练 (Train) ---
    # window_size 可以设大一点，比如 50，因为我们有10年数据
    train_env = DummyVecEnv([lambda: PortfolioEnv(train_data, train_prices, window_size=50)])
    
    print("\nStart Training (2010-2020)...")
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,   # 比你现在更激进
        n_steps=1024,         # 更频繁更新策略
        batch_size=128,       # 更噪声，有利探索
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.02,       # ✅ 强制增加探索性（非常关键）
        verbose=1
    )

    model.learn(total_timesteps=350_000)
    
    # --- 4. 回测 (Test / Backtest) ---
    print("\nStart Backtesting (2021-2023)...")
    test_env = PortfolioEnv(test_data, test_prices, window_size=50, initial_balance=10000)
    
    obs = test_env.reset()
    done = False
    
    history_value = []
    history_weights = []
    
    while not done:
        action, _ = model.predict(obs, deterministic=True) # 确定性模式
        obs, reward, done, _ = test_env.step(action)
        
        history_value.append(test_env.portfolio_value)
        history_weights.append(test_env.softmax(action))

    # --- 5. 可视化 ---
    history_value = np.array(history_value)
    
    # 对比 Benchmark (等权重持有)
    # 1. 计算测试集所有日期的等权净值
    norm_prices = test_prices / test_prices[0] 
    benchmark_value = np.mean(norm_prices, axis=1) * 10000
    
    # 2. 截取有效区间 (对应 RL 模型的交易区间)
    # RL 是从 window_size 开始产出净值的
    valid_benchmark = benchmark_value[50:]
    
    # --- 关键修改：重新归一化 Benchmark ---
    # 让 Benchmark 在第 0 个有效点（即 RL 开始的时刻）也等于 10000
    valid_benchmark = valid_benchmark / valid_benchmark[0] * 10000
    
    # 确保长度一致 (防止由 done逻辑导致的 1 帧偏差)
    min_len = min(len(history_value), len(valid_benchmark))
    history_value = history_value[:min_len]
    valid_benchmark = valid_benchmark[:min_len]
    plot_dates = test_dates[50 : 50 + min_len]

    # 画净值图
    plt.figure(figsize=(12, 6))
    plt.plot(plot_dates, history_value, label='RL Model Portfolio', color='blue', linewidth=2)
    plt.plot(plot_dates, valid_benchmark, label='Equal Weight Benchmark', color='gray', linestyle='--')
    plt.title('Performance: RL vs Benchmark (2021-2023 Bear/Bull Cycle)')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
    
    # 画仓位图
    plt.figure(figsize=(12, 6))
    history_weights = np.array(history_weights)
    labels = tickers + ['Cash']
    plt.stackplot(range(len(history_weights)), history_weights.T, labels=labels, alpha=0.8)
    plt.title('Portfolio Allocation Dynamics')
    plt.legend(loc='upper left')
    plt.margins(0,0)
    plt.show()
    
    # 计算最终收益
    rl_return = (history_value[-1] - 10000) / 10000 * 100
    bn_return = (benchmark_value[-1] - 10000) / 10000 * 100
    print(f"\nFinal Results:")
    print(f"RL Model Return: {rl_return:.2f}%")
    print(f"Benchmark Return: {bn_return:.2f}%")