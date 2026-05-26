import json
import os

import numpy as np
import time
import pickle
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def train_model():
    # 加载数据集，编码方式同MLP
    print("=" * 33 + "读取数据编码" + "=" * 33)
    X_train = np.load("input_train_scaled.npy")
    X_val = np.load("input_val_scaled.npy")
    y_train = np.load("output_train_scaled.npy")
    y_val = np.load("output_val_scaled.npy")
    print("=" * 33 + "读取完毕" + "=" * 33)
    print(f"训练集输入形状:{X_train.shape}")
    print(f"验证集输入形状:{X_val.shape}")
    print(f"训练集输出形状:{y_train.shape}")
    print(f"验证集输出形状:{y_val.shape}")
    print("=" * 33 + "读取完毕" + "=" * 33)

    alpha_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    best_score = float('inf')
    best_params = None
    results_log = dict()
    results_log['info'] = []
    print("=" * 33 + "开始进行网格搜索最佳超参数" + "=" * 33)
    count = 0
    total = len(alpha_values)
    for alpha in alpha_values:
        start = time.time()
        ridge = Ridge(
            alpha=alpha,
            fit_intercept=True,
            solver='auto',  # 自动选择求解器
            random_state=42
        )
        model = MultiOutputRegressor(ridge)
        model.fit(X_val, y_val)
        val_pred = model.predict(X_val)
        val_mse = mean_squared_error(y_val, val_pred, multioutput='uniform_average')
        val_r2 = r2_score(y_val, val_pred, multioutput='uniform_average')
        elapsed = time.time() - start
        result_info = {
            'params': alpha,
            'val_mse': val_mse,
            'val_r2': val_r2,
            'time': elapsed
        }
        results_log['info'].append(result_info)
        count += 1
        if val_mse < best_score:
            best_score = val_mse
            best_params = alpha
        print(f"alpha={alpha}")
        print(f"Val MSE={val_mse:.6f} | R²={val_r2:.4f} |")
        print(f"耗时={elapsed:.4f}s")
        print('=' * 33 + f"已完成:{count}/{total}" + '=' * 33)
    print("=" * 33 + "最佳超参数选择完毕" + "=" * 33)
    print(f"alpha: {best_params}")
    print(f"验证集最佳 MSE: {best_score:.6f}")
    json_str = json.dumps(results_log, indent=4)
    with open("history.log", 'w', encoding='utf-8') as f:
        f.write(json_str)
    print("=" * 33 + "日志保存完毕" + "=" * 33)
    final_ridge = Ridge(
        alpha=best_params,
        fit_intercept=True,
        solver='auto',
        random_state=42
    )
    train_start = time.time()
    final_ridge.fit(X_train, y_train)
    train_time = time.time() - train_start
    print("=" * 33 + f"最终模型训练完毕, 耗时:{train_time:4f}s" + "=" * 33)
    # 保存最终模型
    with open('ridge_model.pkl', 'wb') as f:
        pickle.dump(final_ridge, f)
    print("=" * 33 + f"模型保存完毕" + "=" * 33)
    return final_ridge


if __name__ == "__main__":
    model = train_model()
