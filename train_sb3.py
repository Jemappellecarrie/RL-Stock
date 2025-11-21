"""
基于 stable-baselines3 的股票交易 PPO 训练脚本（PyTorch 后端）。

使用方式：
    conda activate stock
    cd /Users/zoe/P4ML/Final_Project/RL-Stock
    python train_sb3.py
"""

import os
import glob
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from rlenv.StockTradingEnv0 import StockTradingEnv as StockEnv, INITIAL_ACCOUNT_BALANCE



# ---------- 工具函数 ----------

def make_env(csv_path: str):
    def _init():
        df = pd.read_csv(csv_path)

        # 按日期排序
        if "date" in df.columns:
            df = df.sort_values("date")

        # 丢弃 open/close 为 NaN 的行
        if {"open", "close"}.issubset(df.columns):
            df = df.dropna(subset=["open", "close"])

        # 只保留交易状态正常的行（如果有 tradestatus）
        if "tradestatus" in df.columns:
            df = df[df["tradestatus"] == 1]

        df = df.reset_index(drop=True)
        env = StockEnv(df)
        return env

    return _init



def extract_code_from_path(path: str) -> str:
    """
    根据文件名提取股票代码部分。

    假设命名类似：
        sh.600036.招商银行.csv
    则返回：
        sh.600036
    """
    base = os.path.basename(path)
    parts = base.split(".")
    if len(parts) < 3:
        # 命名不符合预期时，退化为去掉扩展名
        return os.path.splitext(base)[0]
    return ".".join(parts[:2])


def list_train_codes(train_dir: str = "stockdata/train"):
    """
    列出 train 目录下所有股票代码（按文件名推断）。
    只返回在 test 目录中也存在对应文件的代码。
    """
    pattern = os.path.join(train_dir, "*.csv")
    train_files = sorted(glob.glob(pattern))
    if not train_files:
        raise FileNotFoundError(f"在 {train_dir} 下没有找到任何 CSV，请先运行 get_stock_data.py")

    codes = []
    for f in train_files:
        code = extract_code_from_path(f)
        # 确认 test 目录下也有该代码对应的文件
        test_pattern = os.path.join("stockdata", "test", f"{code}.*.csv")
        if glob.glob(test_pattern):
            codes.append(code)

    if not codes:
        raise RuntimeError("train 中的股票在 test 中都找不到对应文件，请检查数据生成方式。")

    return sorted(set(codes))


def pick_train_test_pair_by_index(index: int = 0):
    """
    通过索引选择一只股票，返回对应的 train/test CSV 路径。

    index: 在可用股票代码列表中的位置，从 0 开始。
    """
    codes = list_train_codes()
    if index < 0 or index >= len(codes):
        raise IndexError(f"index={index} 越界，可用股票数量为 {len(codes)}")

    code = codes[index]

    train_pattern = os.path.join("stockdata", "train", f"{code}.*.csv")
    test_pattern = os.path.join("stockdata", "test", f"{code}.*.csv")

    train_files = sorted(glob.glob(train_pattern))
    test_files = sorted(glob.glob(test_pattern))

    if not train_files:
        raise FileNotFoundError(f"在 train 目录下找不到代码 {code} 对应的 CSV")
    if not test_files:
        raise FileNotFoundError(f"在 test 目录下找不到代码 {code} 对应的 CSV")

    train_csv = train_files[0]
    test_csv = test_files[0]

    print(f"[INFO] 选定标的代码: {code}")
    print(f"[INFO] 训练集: {train_csv}")
    print(f"[INFO] 测试集: {test_csv}")

    return train_csv, test_csv, code


def pick_train_test_pair_by_code(code: str):
    """
    通过指定股票代码选择 train/test CSV。

    例如：
        code = "sh.600036"
    """
    train_pattern = os.path.join("stockdata", "train", f"{code}.*.csv")
    test_pattern = os.path.join("stockdata", "test", f"{code}.*.csv")

    train_files = sorted(glob.glob(train_pattern))
    test_files = sorted(glob.glob(test_pattern))

    if not train_files:
        raise FileNotFoundError(f"在 train 目录下找不到代码 {code} 对应的 CSV")
    if not test_files:
        raise FileNotFoundError(f"在 test 目录下找不到代码 {code} 对应的 CSV")

    train_csv = train_files[0]
    test_csv = test_files[0]

    print(f"[INFO] 选定标的代码: {code}")
    print(f"[INFO] 训练集: {train_csv}")
    print(f"[INFO] 测试集: {test_csv}")

    return train_csv, test_csv, code


# ---------- 评估函数 ----------

def evaluate_single(model: PPO, test_csv: str, code: str):
    """
    在单只股票的测试集上评估策略表现，并给出简短总结。
    """
    df = pd.read_csv(test_csv)
    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)

    env = StockEnv(df)

    obs = env.reset()
    done = False
    total_reward = 0.0
    steps = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += float(reward)
        steps += 1

    # 用环境内部的状态计算最终盈利
    final_net_worth = env.net_worth
    profit = final_net_worth - INITIAL_ACCOUNT_BALANCE

    print("单只股票\n")
    print(f"- 初始本金：{INITIAL_ACCOUNT_BALANCE:.0f}")
    print(f"- 股票代码：{code}")
    print(f"- 测试集：{test_csv}")
    print(f"- 模拟操作 {steps} 天，最终盈利约 {profit:.2f}\n")

    return profit



def evaluate_on_all_tests(model: PPO):
    """
    使用训练好的模型，在所有 test 股票上各跑一遍，
    做整体盈利/不亏不赚/亏损比例统计。
    """
    pattern = os.path.join("stockdata", "test", "*.csv")
    test_files = sorted(glob.glob(pattern))
    if not test_files:
        print("[WARN] test 目录下没有 CSV，跳过多标的评估。")
        return

    profits = []
    n_profit = n_loss = n_flat = 0

    for path in test_files:
        code = extract_code_from_path(path)

        df = pd.read_csv(path)
        if "date" in df.columns:
            df = df.sort_values("date").reset_index(drop=True)

        env = StockEnv(df)
        obs = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += float(reward)

        final_net_worth = env.net_worth
        profit = final_net_worth - INITIAL_ACCOUNT_BALANCE
        profits.append(profit)

        # 按盈利情况分类
        eps = 1e-6
        if profit > eps:
            n_profit += 1
        elif profit < -eps:
            n_loss += 1
        else:
            n_flat += 1

    total = len(profits)
    if total == 0:
        print("[WARN] 没有可评估的 test 股票。")
        return

    pct_profit = 100.0 * n_profit / total
    pct_flat   = 100.0 * n_flat   / total
    pct_loss   = 100.0 * n_loss   / total

    mean_profit = sum(profits) / total
    max_profit = max(profits)
    min_profit = min(profits)

    print("多只股票\n")
    print(f"- 选取 {total} 只股票，进行测试评估，共计：")
    print(f"  · 盈利：{pct_profit:.1f}% （{n_profit} 只）")
    print(f"  · 不亏不赚：{pct_flat:.1f}% （{n_flat} 只）")
    print(f"  · 亏损：{pct_loss:.1f}% （{n_loss} 只）")
    print(f"- 单支股票平均盈利：{mean_profit:.2f}")
    print(f"- 单支股票最高盈利：{max_profit:.2f}")
    print(f"- 单支股票最大亏损：{min_profit:.2f}")


# ---------- 主流程 ----------

def train_and_eval():
    # 1. 选一只股票训练 + 对应测试
    #
    # 方式 A：按索引选第几只
    train_csv, test_csv, code = pick_train_test_pair_by_index(index=1)
    #
    # 方式 B：指定代码（如果你知道想用哪只）
    # train_csv, test_csv, code = pick_train_test_pair_by_code("sh.600036")

    # 2. 创建向量化环境（用于训练）
    train_env = DummyVecEnv([make_env(train_csv)])

    # 3. 定义 PPO 模型
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        verbose=1,
        tensorboard_log="./sb3_logs",
        device="auto",
    )

    # 4. 训练
    total_timesteps = 50_000
    print(f"[INFO] 开始训练代码 {code}，共 {total_timesteps} 步...")
    model.learn(total_timesteps=total_timesteps)
    model.save(f"ppo_stock_sb3_{code.replace('.', '_')}")
    print("[INFO] 训练完成，模型已保存。")

    # 5. 在对应测试集上评估一轮（使用原始 env）
    print("\n[INFO] 在对应测试集上评估：")
    evaluate_single(model, test_csv,code)

    # 6.（可选）在所有 test 股票上做一次横向评估
    #    这时相当于“在其他股票上测试收益”。
    evaluate_on_all_tests(model)


if __name__ == "__main__":
    train_and_eval()
