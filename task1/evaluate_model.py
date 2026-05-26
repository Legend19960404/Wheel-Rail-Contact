import json
import os
import pickle

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import task1.MLP_AlexNet_model.train_AlexNet as alexnet
import task1.MLP_model.train_MLP as mlp
import task1.LightGBM_model.train_LightGBM_model as lgbm
import task1.XGBoost_model.train_XGBoost_model as xgb
import pandas as pd


def decoder(norm_force, scaler_X, scaler_Y, scaler_Z):
    """将归一化结果解码为真实值"""
    norm_force_x = norm_force[:, [0]]
    norm_force_y = norm_force[:, [1]]
    norm_force_z = norm_force[:, [2]]

    force_x = scaler_X.inverse_transform(norm_force_x).flatten()
    force_y = scaler_Y.inverse_transform(norm_force_y).flatten()
    force_z = scaler_Z.inverse_transform(norm_force_z).flatten()

    real_force = np.column_stack([force_x, force_y, force_z])

    return real_force


def plot_trainning_log(log_path):
    """绘制训练日志"""
    with open(log_path, 'r') as f:
        log = json.load(f)
        epochs = log['epoch_idx']
        train_loss = log['train_loss']
        valid_loss = log['valid_loss']
        lr = log['lr']
        epoch_time = log['epoch_time']

        plt.figure()
        plt.subplot(2, 2, 1)
        plt.title("Train Loss")
        plt.plot(epochs, train_loss, marker='*', linestyle='--')
        plt.ylabel("Train Loss")

        plt.subplot(2, 2, 2)
        plt.title("Valid Loss")
        plt.plot(epochs, valid_loss, marker='*', linestyle='--')
        plt.ylabel("Valid Loss")

        plt.subplot(2, 2, 3)
        plt.title("Learning Rate")
        plt.plot(epochs, lr, marker='*', linestyle='--')
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")

        plt.subplot(2, 2, 4)
        plt.title("Epoch Time")
        plt.plot(epochs, epoch_time, marker='*', linestyle='--')
        plt.xlabel("Epoch")
        plt.ylabel("Epoch Time")

        plt.show()


def plot_scatter(y_true, y_pred, label=None):
    plt.figure()
    plt.scatter(y_true, y_pred, label=label, alpha=0.6)
    plt.plot(y_true, y_true)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()


def nrmse(y_true, y_pred):
    """计算归一化RMSE"""
    true_range = np.ptp(y_true, axis=0)
    range_x = true_range[0]
    range_y = true_range[1]
    range_z = true_range[2]

    rmse_x = np.sqrt(mean_squared_error(y_true[:, 0], y_pred[:, 0]))
    rmse_y = np.sqrt(mean_squared_error(y_true[:, 1], y_pred[:, 1]))
    rmse_z = np.sqrt(mean_squared_error(y_true[:, 2], y_pred[:, 2]))

    nrmse_x = rmse_x / range_x
    nrmse_y = rmse_y / range_y
    nrmse_z = rmse_z / range_z

    return {
        'x': nrmse_x,
        'y': nrmse_y,
        'z': nrmse_z
    }


def NMAE(y_true, y_pred):
    """计算归一化MAE"""
    true_range = np.ptp(y_true, axis=0)
    range_x = true_range[0]
    range_y = true_range[1]
    range_z = true_range[2]

    mae_x = mean_absolute_error(y_true[:, 0], y_pred[:, 0])
    mae_y = mean_absolute_error(y_true[:, 1], y_pred[:, 1])
    mae_z = mean_absolute_error(y_true[:, 2], y_pred[:, 2])

    nmae_x = mae_x / range_x
    nmae_y = mae_y / range_y
    nmae_z = mae_z / range_z

    return {
        'x': nmae_x,
        'y': nmae_y,
        'z': nmae_z
    }


def max_ae_norm(y_true, y_pred):
    """计算每个方向的归一化最大绝对误差（MaxAE_norm）"""
    abs_errors = np.abs(y_pred - y_true)
    max_abs_errors = np.max(abs_errors, axis=0)
    ranges = np.ptp(y_true, axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        norm_max_ae = np.where(ranges > 0, max_abs_errors / ranges, np.nan)
    return {'x': norm_max_ae[0], 'y': norm_max_ae[1], 'z': norm_max_ae[2]}


def mae_3d(y_true, y_pred):
    """计算归一化合力平均绝对误差"""
    error_norm = np.linalg.norm(y_pred - y_true, axis=1)
    mean_error = np.mean(error_norm / np.linalg.norm(y_true, axis=1))
    return mean_error


def mde(y_true, y_pred, threshold=None):
    """计算平均方向误差（度），排除合力模过小的样本"""
    true_norm = np.linalg.norm(y_true, axis=1)
    pred_norm = np.linalg.norm(y_pred, axis=1)
    if threshold is None:
        threshold = 1e-12
    valid_mask = (true_norm > threshold) & (pred_norm > threshold)
    valid_count = np.sum(valid_mask)
    if valid_count == 0:
        return {'mean_angle_deg': np.nan, 'valid_samples': 0}
    # 提取有效样本
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    # 计算点积和模的乘积
    dot = np.sum(y_true_valid * y_pred_valid, axis=1)
    norm_product = np.linalg.norm(y_true_valid, axis=1) * np.linalg.norm(y_pred_valid, axis=1)
    # 计算夹角（弧度），处理数值舍入误差（clip到[-1,1]）
    cos_angle = dot / norm_product
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angles_rad = np.arccos(cos_angle)
    # 转换为度并求平均
    mean_angle_deg = np.mean(np.degrees(angles_rad))
    return {'mean_angle_deg': mean_angle_deg, 'valid_samples': valid_count}


def r2(y_true, y_pred):
    """计算三个方向的R^2"""
    r2_x = r2_score(y_true[:, 0], y_pred[:, 0])
    r2_y = r2_score(y_true[:, 1], y_pred[:, 1])
    r2_z = r2_score(y_true[:, 2], y_pred[:, 2])
    r2_total = r2_score(y_true, y_pred)
    return {
        'x': r2_x,
        'y': r2_y,
        'z': r2_z,
        'tot': r2_total
    }


def rmse(y_true, y_pred):
    rmse_x = np.sqrt(mean_squared_error(y_true[:, 0], y_pred[:, 0]))
    rmse_y = np.sqrt(mean_squared_error(y_true[:, 1], y_pred[:, 1]))
    rmse_z = np.sqrt(mean_squared_error(y_true[:, 2], y_pred[:, 2]))
    return {
        'x': rmse_x,
        'y': rmse_y,
        'z': rmse_z
    }


def max_ae(y_true, y_pred):
    abs_errors = np.abs(y_pred - y_true)
    max_abs_errors = np.max(abs_errors, axis=0)
    return {
        'x': max_abs_errors[0],
        'y': max_abs_errors[1],
        'z': max_abs_errors[2]
    }


def mae(y_true, y_pred):
    mae_x = mean_absolute_error(y_true[:, 0], y_pred[:, 0])
    mae_y = mean_absolute_error(y_true[:, 1], y_pred[:, 1])
    mae_z = mean_absolute_error(y_true[:, 2], y_pred[:, 2])
    return {
        'x': mae_x,
        'y': mae_y,
        'z': mae_z
    }


def evaluate_model(model_type):
    """计算模型在测试上的性能"""
    output_predict = None
    scaler_x = None
    scaler_y = None
    scaler_z = None
    y_test = None
    if model_type == 'mlp':

        X_test = np.load("MLP_model/input_test_scaled.npy")
        y_test = np.load("MLP_model/output_test_scaled.npy")

        scaler_x = joblib.load("MLP_model/output_scaler_forceX.pkl")
        scaler_y = joblib.load("MLP_model/output_scaler_forceY.pkl")
        scaler_z = joblib.load("MLP_model/output_scaler_forceZ.pkl")

        input_features = X_test.shape[1]
        output_features = y_test.shape[1]
        model = mlp.create_mlp_model(input_features, output_features, droup_rate=0.2, model_type=1)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model_pth = "MLP_best_model_1.pth"
        model_state_dict = torch.load(model_pth)['model_state_dict']
        model.load_state_dict(model_state_dict)
        model.eval()
        with torch.no_grad():
            input_tensor = torch.FloatTensor(X_test)
            input_tensor = input_tensor.to(device)
            output_tensor = model(input_tensor)
            output_predict = torch.detach(output_tensor).numpy()

    elif model_type == "alexnet":
        X_test = np.load("MLP_AlexNet_model/input_test_scaled.npy")
        y_test = np.load("MLP_AlexNet_model/output_test_scaled.npy")

        scaler_x = joblib.load("MLP_AlexNet_model/output_scaler_forceX.pkl")
        scaler_y = joblib.load("MLP_AlexNet_model/output_scaler_forceY.pkl")
        scaler_z = joblib.load("MLP_AlexNet_model/output_scaler_forceZ.pkl")

        model = alexnet.create_network()
        model_pth = "alex_best_model.pth"
        model_state_dic = torch.load(model_pth)['model_state_dict']

        wheel_branch_state = model_state_dic['wheel_branch_state']
        rail_brach_state = model_state_dic['rail_branch_state']
        mlp_head_state = model_state_dic['mlp_head_state']

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        wheel_branch = model['wheel_branch']
        rail_brach = model['rail_branch']
        mlp_head = model['mlp_head']
        forward_func = model['forward']

        wheel_branch.load_state_dict(wheel_branch_state)
        wheel_branch = wheel_branch.to(device)
        rail_brach.load_state_dict(rail_brach_state)
        rail_branch = rail_brach.to(device)
        mlp_head.load_state_dict(mlp_head_state)
        mlp_head = mlp_head.to(device)

        wheel_branch.eval()
        rail_branch.eval()
        mlp_head.eval()
        with torch.no_grad():
            input_tensor = torch.FloatTensor(X_test)
            add_features = input_tensor[:, :18]
            add_features.to(device)
            inputs_x1 = input_tensor[:, 18:18 + 1024]
            inputs_x1.to(device)
            inputs_x2 = input_tensor[:, 18 + 1024:]
            inputs_x2.to(device)
            outputs = forward_func(inputs_x1, inputs_x2, add_features)
            output_predict = torch.detach(outputs).numpy()

    elif model_type == 'lgbm':
        X_test = np.load("LightGBM_model/input_test_scaled.npy")
        y_test = np.load("LightGBM_model/output_test_scaled.npy")

        scaler_x = joblib.load("LightGBM_model/output_scaler_forceX.pkl")
        scaler_y = joblib.load("LightGBM_model/output_scaler_forceY.pkl")
        scaler_z = joblib.load("LightGBM_model/output_scaler_forceZ.pkl")

        if os.path.exists('lgbm_model.pkl'):
            with open('lgbm_model.pkl', 'rb') as f:
                model = pickle.load(f)
        output_predict = lgbm.predict_lgbm(X_test, model)

    elif model_type == 'xgb':
        X_test = np.load("XGBoost_model/input_test_scaled.npy")
        y_test = np.load("XGBoost_model/output_test_scaled.npy")

        scaler_x = joblib.load("XGBoost_model/output_scaler_forceX.pkl")
        scaler_y = joblib.load("XGBoost_model/output_scaler_forceY.pkl")
        scaler_z = joblib.load("XGBoost_model/output_scaler_forceZ.pkl")

        if os.path.exists('xgb_model.pkl'):
            with open('xgb_model.pkl', 'rb') as f:
                model = pickle.load(f)
        output_predict = xgb.predict_lgbm(X_test, model)

    elif model_type == 'rf':
        X_test = np.load("RandomForest_model/input_test_scaled.npy")
        y_test = np.load("RandomForest_model/output_test_scaled.npy")

        scaler_x = joblib.load("RandomForest_model/output_scaler_forceX.pkl")
        scaler_y = joblib.load("RandomForest_model/output_scaler_forceY.pkl")
        scaler_z = joblib.load("RandomForest_model/output_scaler_forceZ.pkl")

        if os.path.exists('rf_model.pkl'):
            with open('rf_model.pkl', 'rb') as f:
                model = pickle.load(f)
        output_predict = model.predict(X_test)

    elif model_type == 'ridge':
        X_test = np.load("Ridge_model/input_test_scaled.npy")
        y_test = np.load("Ridge_model/output_test_scaled.npy")

        scaler_x = joblib.load("Ridge_model/output_scaler_forceX.pkl")
        scaler_y = joblib.load("Ridge_model/output_scaler_forceY.pkl")
        scaler_z = joblib.load("Ridge_model/output_scaler_forceZ.pkl")

        if os.path.exists('ridge_model.pkl'):
            with open('ridge_model.pkl', 'rb') as f:
                model = pickle.load(f)
        output_predict = model.predict(X_test)

    y_pred = decoder(output_predict, scaler_x, scaler_y, scaler_z)
    y_true = decoder(y_test, scaler_x, scaler_y, scaler_z)
    df_res = pd.DataFrame(np.column_stack([y_true[:, 0], y_pred[:, 0], y_true[:, 1], y_pred[:, 1], y_true[:, 2], y_pred[:, 2]]))
    df_res.to_excel(f'{model_type}_pred_true.xlsx')
    # 计算评价指标
    res_rmse = rmse(y_true, y_pred)
    res_nrmse = nrmse(y_true, y_pred)
    res_mae = mae(y_true, y_pred)
    res_max_ae = max_ae(y_true, y_pred)
    res_r2 = r2(y_true, y_pred)
    res_mae_3d = mae_3d(y_true, y_pred)
    res_mde = mde(y_true, y_pred)

    res = np.array([res_rmse['x'], res_rmse['y'], res_rmse['z'],
                    res_nrmse['x'], res_nrmse['y'], res_nrmse['z'],
                    res_mae['x'], res_mae['y'], res_mae['z'],
                    res_r2['x'], res_r2['y'], res_r2['z'], res_r2['tot'],
                    res_mae_3d, res_mde['mean_angle_deg']])
    return res


if __name__ == "__main__":
    # MLP评估
    print("=" * 33 + "开始评估" + "=" * 33)
    print('=' * 33 + ' mlp ' + '=' * 33)
    res_mlp = evaluate_model('mlp')
    print('=' * 33 + ' xgboost ' + '=' * 33)
    res_xgb = evaluate_model('xgb')
    res_alexnet = evaluate_model('alexnet')
    print('=' * 33 + ' lightbgm ' + '=' * 33)
    res_lightgbm = evaluate_model('lgbm')
    print('=' * 33 + ' ridge ' + '=' * 33)
    res_ridge = evaluate_model('ridge')
    print('=' * 33 + ' random forest ' + '=' * 33)
    res_rf = evaluate_model('rf')
    print("=" * 33 + "评估完毕" + "=" * 33)
    res = np.column_stack([res_mlp, res_alexnet, res_rf, res_ridge, res_xgb, res_lightgbm])
    df = pd.DataFrame(res.T)
    df.to_excel('task_1_res.xlsx')
