import json
import os.path
import pickle
import time
import lightgbm as lgb
import numpy as np


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
    print("=" * 33 + "开始训练LightGBM模型" + "=" * 33)
    for i in range(y_train.shape[1]):
        print(f"{'='*33} 训练输出维度:{['X','Y','Z'][i]} {'='*33}")
        start_time = time.time()
        eval_result = {}  # 用于记录训练历史

        callbacks = [
            lgb.record_evaluation(eval_result),  # 记录指标历史
            lgb.early_stopping(stopping_rounds=30, verbose=True),  # 早停 + 触发提示
            lgb.log_evaluation(period=10)  # 每10轮打印一次
        ]
        model = lgb.LGBMRegressor(
            objective='regression',
            metric='mse',
            boosting_type='gbdt',
            learning_rate=0.05,
            num_leaves=63,  # 比默认31稍大，捕捉更复杂模式
            max_depth=10,  # 防止过深
            subsample=0.8,  # 行采样
            colsample_bytree=0.7,  # 列采样（应对高维）
            reg_alpha=0.1,  # L1正则
            reg_lambda=0.1,  # L2正则
            n_estimators=2000,  # 大迭代数 + 早停
            random_state=42,
            n_jobs=-1,  # 充分利用CPU
            verbose=0
        )
        model.fit(
            X_train, y_train[:, i],
            eval_set=[(X_val, y_val[:, i])],
            eval_metric='mse',
            callbacks=callbacks
        )
        elapsed = time.time() - start_time
        best_iter = model.best_iteration_ if hasattr(model, 'best_iteration_') else model.n_estimators
        val_mse = model.best_score_['valid_0']['l2'] if hasattr(model, 'best_score_') else 'N/A'
        dim_name = ['X', 'Y', 'Z'][i]
        results_log[dim_name] = eval_result
        print(f"输出维度 {['X','Y','Z'][i]} | 训练时间: {elapsed:.4f}s | 早停轮次: {best_iter} | 验证MSE: {val_mse:.6f}")
        models.append(model)

    with open(f'lgbm_model.pkl', 'wb') as f:
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
