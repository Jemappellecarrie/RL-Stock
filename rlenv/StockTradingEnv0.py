import random
import gym
from gym import spaces
import numpy as np
import pandas as pd

MAX_ACCOUNT_BALANCE = 2147483647
MAX_NUM_SHARES = 2147483647
MAX_SHARE_PRICE = 5000
MAX_VOLUME = 1000e8
MAX_AMOUNT = 3e10

INITIAL_ACCOUNT_BALANCE = 10000


class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym + SB3"""
    metadata = {"render.modes": ["human"]}

    def __init__(self, df: pd.DataFrame):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.reward_range = (0, MAX_ACCOUNT_BALANCE)

        # Action: [action_type, amount]
        # action_type ∈ [0,3): <1=buy, <2=sell, ≥2=hold
        # amount ∈ [0,1]: 百分比
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([3.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Observation: 19 维向量，已做简单归一化
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(19,), dtype=np.float32
        )

        self.reset()

    # ---------- 内部工具函数 ----------

    def _next_observation(self) -> np.ndarray:
        row = self.df.loc[self.current_step]

        obs = np.array(
            [
                row["open"] / MAX_SHARE_PRICE,
                row["high"] / MAX_SHARE_PRICE,
                row["low"] / MAX_SHARE_PRICE,
                row["close"] / MAX_SHARE_PRICE,
                row["volume"] / MAX_VOLUME,
                row["amount"] / MAX_AMOUNT,
                row["adjustflag"] / 10.0,
                row["tradestatus"] / 1.0,
                row["pctChg"] / 100.0,
                row["peTTM"] / 1e4,
                row["pbMRQ"] / 100.0,
                row["psTTM"] / 100.0,
                row["pctChg"] / 1e3,  # 这一项和上面的有点重复，后续可改
                self.balance / MAX_ACCOUNT_BALANCE,
                self.max_net_worth / MAX_ACCOUNT_BALANCE,
                self.shares_held / MAX_NUM_SHARES,
                self.cost_basis / MAX_SHARE_PRICE,
                self.total_shares_sold / MAX_NUM_SHARES,
                self.total_sales_value / (MAX_NUM_SHARES * MAX_SHARE_PRICE),
            ],
            dtype=np.float32,
        )

        # 防止 NaN / Inf 污染网络
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return obs

    def _update_cost_basis(self):
        """根据当前总成本和持仓数量更新成本价，避免除零。"""
        if self.shares_held == 0:
            self.cost_basis = 0.0
        else:
            self.cost_basis = self.total_cost / float(self.shares_held)

    def _take_action(self, action: np.ndarray):
        # 强制裁剪动作到合法范围
        action = np.clip(action, self.action_space.low, self.action_space.high)
        action_type = float(action[0])
        amount = float(action[1])

        row = self.df.loc[self.current_step]
        open_price = float(row["open"])
        close_price = float(row["close"])

        # 如果 open/close 有 NaN 或非正值，直接“跳过交易”（hold）
        if not np.isfinite(open_price) or not np.isfinite(close_price) or open_price <= 0 or close_price <= 0:
            # 使用上一个 price 或直接退出买卖逻辑
            # 例如把 current_price 固定为上一个有效价格，或者干脆 hold：
            current_price = getattr(self, "last_price", 1.0)
            # 或者：直接不做任何买卖
            # self.net_worth = self.balance + self.shares_held * current_price
            # self.last_price = current_price
            # return
        else:
            current_price = random.uniform(open_price, close_price)

        # 记录 last_price，供下次 fallback 使用
        self.last_price = current_price

        # 如果 current_price 仍然不合法（极端情况下），直接返回
        if not np.isfinite(current_price) or current_price <= 0:
            self.net_worth = self.balance + self.shares_held * 0.0
            return

        if action_type < 1.0:
            # Buy: 用 amount 百分比的余额买入
            total_possible = int(self.balance / current_price)
            shares_bought = int(total_possible * amount)

            if shares_bought > 0:
                prev_cost = self.cost_basis * self.shares_held
                additional_cost = shares_bought * current_price

                self.balance -= additional_cost
                self.shares_held += shares_bought

                # 更新总成本和成本价
                self.total_cost = prev_cost + additional_cost
                self._update_cost_basis()

        elif action_type < 2.0:
            # Sell: 卖出 amount 百分比的持仓
            shares_sold = int(self.shares_held * amount)

            if shares_sold > 0:
                self.balance += shares_sold * current_price
                self.shares_held -= shares_sold
                self.total_shares_sold += shares_sold
                self.total_sales_value += shares_sold * current_price

                # 持仓变化后，总成本也要更新
                self.total_cost = self.cost_basis * self.shares_held

        else:
            # Hold: 不买不卖
            pass

        # 更新净值
        self.net_worth = self.balance + self.shares_held * current_price
        self.max_net_worth = max(self.max_net_worth, self.net_worth)

        # 如果完全空仓，成本价设为 0
        if self.shares_held == 0:
            self.cost_basis = 0.0
            self.total_cost = 0.0

    # ---------- Gym 接口 ----------

    def step(self, action):
        # 执行动作
        self._take_action(action)

        # 时间推进
        self.current_step += 1
        if self.current_step >= len(self.df):
            # 可以选择循环，也可以在到达末尾时终止一个 episode
            self.current_step = len(self.df) - 1
            done = True
        else:
            done = False

        # 奖励：使用归一化收益（更平滑）
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE
        reward = profit / INITIAL_ACCOUNT_BALANCE

        # 如果破产，可以直接终止
        if self.net_worth <= 0:
            done = True

        obs = self._next_observation()
        return obs, reward, done, {}

    def reset(self, new_df: pd.DataFrame = None):
        # 状态初始化
        self.balance = float(INITIAL_ACCOUNT_BALANCE)
        self.net_worth = float(INITIAL_ACCOUNT_BALANCE)
        self.max_net_worth = float(INITIAL_ACCOUNT_BALANCE)
        self.shares_held = 0
        self.cost_basis = 0.0
        self.total_shares_sold = 0
        self.total_sales_value = 0.0
        self.total_cost = 0.0

        if new_df is not None:
            self.df = new_df.reset_index(drop=True)

        self.current_step = 0
        return self._next_observation()

    def render(self, mode="human", close=False):
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE
        print("-" * 30)
        print(f"Step: {self.current_step}")
        print(f"Balance: {self.balance:.2f}")
        print(f"Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})")
        print(f"Avg cost for held shares: {self.cost_basis:.4f} (Total sales value: {self.total_sales_value:.2f})")
        print(f"Net worth: {self.net_worth:.2f} (Max net worth: {self.max_net_worth:.2f})")
        print(f"Profit: {profit:.2f}")
        return profit
