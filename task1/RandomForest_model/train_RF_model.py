import copy
import json
import os

import numpy as np
import time
import pickle
from sklearn.ensemble import RandomForestRegressor
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
    param_grid = {
        'n_estimators': [100, 150, 200],
        'max_depth': [15, 25, 35, None],
        'min_samples_split': [15, 25, 40],
        'min_samples_leaf': [8, 12, 20]
    }
    best_score = float('inf')
    best_params = None
    results_log = dict()
    results_log['info'] = []
    print("=" * 33 + "开始进行网格搜索最佳超参数" + "=" * 33)
    count = 0
    total = len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split']) * len(param_grid['min_samples_leaf'])
    for n_est in param_grid['n_estimators']:
        for max_d in param_grid['max_depth']:
            for min_split in param_grid['min_samples_split']:
                for min_leaf in param_grid['min_samples_leaf']:
                    start = time.time()
                    # 构建并训练模型
                    rf = RandomForestRegressor(
                        n_estimators=n_est,
                        max_depth=max_d,
                        min_samples_split=min_split,
                        min_samples_leaf=min_leaf,
                        max_features='sqrt',
                        random_state=42,
                        n_jobs=-1,
                        verbose=0
                    )
                    model = MultiOutputRegressor(rf)
                    model.fit(X_val, y_val)
                    val_pred = model.predict(X_val)
                    val_mse = mean_squared_error(y_val, val_pred, multioutput='uniform_average')
                    val_r2 = r2_score(y_val, val_pred, multioutput='uniform_average')
                    elapsed = time.time() - start
                    result_info = {
                        'params': (n_est, max_d, min_split, min_leaf),
                        'val_mse': val_mse,
                        'val_r2': val_r2,
                        'time': elapsed
                    }
                    results_log['info'].append(result_info)
                    count += 1
                    if val_mse < best_score:
                        best_score = val_mse
                        best_params = (n_est, max_d, min_split, min_leaf)
                    print(f"n_est={n_est:3d} | depth={max_d} | min_split={min_split:2d} | min_leaf={min_leaf:2d}")
                    print(f"Val MSE={val_mse:.6f} | R²={val_r2:.4f} |")
                    print(f"耗时={elapsed:.4f}s")
                    print('='*33+f"已完成:{count}/{total}"+'='*33)

    print("=" * 33 + "最佳超参数选择完毕" + "=" * 33)
    print(f"n_estimators: {best_params[0]}")
    print(f"max_depth: {best_params[1]}")
    print(f"min_samples_split: {best_params[2]}")
    print(f"min_samples_leaf: {best_params[3]}")
    print(f"验证集最佳 MSE: {best_score:.6f}")
    json_str = json.dumps(results_log, indent=4)
    with open("history.log", 'w', encoding='utf-8') as f:
        f.write(json_str)
    print("=" * 33 + "日志保存完毕" + "=" * 33)

    # 使用最佳参数构建最终模型
    final_rf = RandomForestRegressor(
        n_estimators=best_params[0],
        max_depth=best_params[1],
        min_samples_split=best_params[2],
        min_samples_leaf=best_params[3],
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    final_model = MultiOutputRegressor(final_rf)
    train_start = time.time()
    final_model.fit(X_train, y_train)
    train_time = time.time() - train_start
    print("=" * 33 + f"最终模型训练完毕, 耗时:{train_time:4f}s" + "=" * 33)
    # 保存最终模型
    with open('rf_model.pkl', 'wb') as f:
        pickle.dump(final_model, f)
    print("=" * 33 + f"模型保存完毕" + "=" * 33)
    return final_model


if __name__ == "__main__":
    model = train_model()
