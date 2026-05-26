import numpy as np
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
import time
import pickle
import json


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
    models = []
    results_log = dict()
    print("=" * 33 + "开始训练XGBoost模型" + "=" * 33)
    # 单独训练每个输出维度（与LightGBM保持一致的策略）
    for i in range(y_train.shape[1]):
        print(f"{'=' * 33} 训练输出维度:{['X', 'Y', 'Z'][i]} {'=' * 33}")
        start_time = time.time()

        # 创建XGBoost回归器
        model = xgb.XGBRegressor(
            objective='reg:squarederror',  # 回归平方误差
            learning_rate=0.05,
            max_depth=10,  # 控制树深度，防止过拟合
            subsample=0.8,  # 行采样比例
            colsample_bytree=0.7,  # 列采样比例（应对高维特征）
            reg_alpha=0.1,  # L1正则化
            reg_lambda=0.1,  # L2正则化
            n_estimators=2000,  # 最大迭代次数
            random_state=42,
            n_jobs=-1,  # 使用所有可用CPU核心
            verbosity=1,  # 显示训练进度
            early_stopping_rounds=30  # 早停设置
        )

        # 训练模型
        model.fit(
            X_train, y_train[:, i],
            eval_set=[(X_val, y_val[:, i])],
            verbose=True
        )

        elapsed = time.time() - start_time

        # 获取最佳结果
        best_iter = len(model.evals_result()['validation_0']['rmse']) if hasattr(model, 'evals_result') else model.n_estimators
        val_rmse = min(model.evals_result()['validation_0']['rmse']) if hasattr(model, 'evals_result') else 'N/A'
        val_mse = val_rmse ** 2  # 转换为MSE便于与LightGBM比较

        dim_name = ['X', 'Y', 'Z'][i]
        results_log[dim_name] = model.evals_result()

        print(f"输出维度 {['X', 'Y', 'Z'][i]} | 训练时间: {elapsed:.4f}s | 训练轮次: {best_iter} | 验证RMSE: {val_rmse:.6f} | 验证MSE: {val_mse:.6f}")
        models.append(model)

    # 保存模型
    with open(f'xgb_model.pkl', 'wb') as f:
        pickle.dump(models, f)
    print("=" * 33 + "模型保存完毕" + "=" * 33)

    json_str = json.dumps(results_log, indent=4)
    with open('history.log', 'w', encoding='utf-8') as f:
        f.write(json_str)
    print("=" * 33 + "日志完毕" + "=" * 33)
    print("=" * 33 + "模型训练完毕" + "=" * 33)
    return models


def predict_lgbm(X, models):
    """模型前向推理"""
    preds = [m.predict(X).reshape(-1, 1) for m in models]
    return np.hstack(preds)


if __name__ == "__main__":
    model = train_model()
