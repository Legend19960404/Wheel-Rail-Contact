import json
import os
import AlexNet.train_model as alex
import GoogLeNet.train_model as googlenet
import NiN.train_model as nin
import ResNet.train_model as resnet
import UNet.train_model as unet
import VGG.train_model as vgg
import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim


def batch_denorm_mat(data_list, scaler):
    """批量反归一化矩阵"""
    normalized_array = np.stack(data_list)  # (n, 50, 50)
    flattened = normalized_array.reshape(len(data_list), -1)  # (n, 2500)
    denormalized_flat = scaler.inverse_transform(flattened)  # (n, 2500)
    denormalized_3d = denormalized_flat.reshape(len(data_list), 50, 50)  # (n, 50, 50)
    return denormalized_3d


def RMSA(y_pred, y_true):
    """均方根精度，y_pred,y_true均为N*50*50维度"""
    N = y_pred.shape[0]
    acc_list = []
    for i in range(N):
        pred = y_pred[i]
        true = y_true[i]
        true_range = true.max() - true.min()
        if true_range == 0:
            # 真实应力全为零，预测也应为全零
            acc = 1.0 if np.allclose(pred, 0) else 0.0
        else:
            rmse = np.sqrt(np.mean((pred - true) ** 2))
            rel_rmse = rmse / true_range
            acc = 1.0 - min(rel_rmse, 1.0)
        acc_list.append(acc)
    return np.mean(acc_list)


def SSIM(y_pred, y_true):
    """结构相似性指数"""
    N = y_pred.shape[0]
    ssim_list = []
    for i in range(N):
        pred = y_pred[i]
        true = y_true[i]
        # 数据范围：真实应力最大值与最小值之差，若为0则设为1避免除零
        data_range = true.max() - true.min()
        if data_range == 0:
            data_range = 1.0
        # ssim返回(score, diff)，我们只取score
        score = ssim(pred, true, data_range=data_range)
        ssim_list.append(score)
    return np.mean(ssim_list)


def IoU(y_pred, y_true, threshold=200):
    """
    接触斑交并比 (Intersection over Union)
    根据阈值（默认0）生成二值掩膜，计算交集与并集之比。
    若并集为0（两掩膜均为空），则IoU定义为1。
    """
    np.save('y_pred.npy', y_pred)
    np.save('y_true.npy', y_true)
    N = y_pred.shape[0]
    iou_list = []
    for i in range(N):
        pred_mask = y_pred[i] > threshold
        true_mask = y_true[i] > threshold
        intersection = np.logical_and(pred_mask, true_mask).sum()
        union = np.logical_or(pred_mask, true_mask).sum()
        if union == 0:
            iou_val = 1.0  # 两者均无接触斑
        else:
            iou_val = intersection / union
        iou_list.append(iou_val)
    return np.mean(iou_list)


def FA(y_pred, y_true):
    """
    总法向力精度 (Force Accuracy)
    基于总力相对误差的补数。
    """
    N = y_pred.shape[0]
    acc_list = []
    for i in range(N):
        sum_pred = y_pred[i].sum()
        sum_true = y_true[i].sum()
        if sum_true == 0:
            acc = 1.0 if sum_pred == 0 else 0.0
        else:
            rel_error = abs(sum_pred - sum_true) / sum_true
            acc = 1.0 - min(rel_error, 1.0)
        acc_list.append(acc)
    return np.mean(acc_list)


def AA(y_pred, y_true, threshold=200):
    """
    接触斑面积精度 (Area Accuracy)
    基于接触斑面积相对误差的补数。
    """
    N = y_pred.shape[0]
    acc_list = []
    for i in range(N):
        area_pred = (y_pred[i] > threshold).sum()
        area_true = (y_true[i] > threshold).sum()
        if area_true == 0:
            acc = 1.0 if area_pred == 0 else 0.0
        else:
            rel_error = abs(area_pred - area_true) / area_true
            acc = 1.0 - min(rel_error, 1.0)
        acc_list.append(acc)
    return np.mean(acc_list)


def PA(y_pred, y_true):
    """峰值应力精度 (Peak Accuracy)"""
    N = y_pred.shape[0]
    acc_list = []
    for i in range(N):
        peak_pred = y_pred[i].max()
        peak_true = y_true[i].max()
        if peak_true == 0:
            acc = 1.0 if peak_pred == 0 else 0.0
        else:
            rel_error = abs(peak_pred - peak_true) / peak_true
            acc = 1.0 - min(rel_error, 1.0)
        acc_list.append(acc)
    return np.mean(acc_list)


def plot_cloud(V, grid_staus_dic, title):
    rx = V.shape[0]
    ry = V.shape[1]
    mx = grid_staus_dic['max_mx']
    my = grid_staus_dic['max_my']
    dx = grid_staus_dic['dx']
    dy = grid_staus_dic['dy']

    x_coords = np.linspace(-mx / 2 * dx, mx / 2 * dx, rx)
    y_coords = np.linspace(-my / 2 * dy, my / 2 * dy, ry)
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)

    plt.figure(figsize=(10, 8))
    contourf_pred = plt.contourf(grid_x, grid_y, V, levels=50, cmap='viridis')
    # 添加颜色条
    cbar = plt.colorbar(contourf_pred)
    cbar.set_label('pn/MPa')

    plt.xlabel('X/mm')
    plt.ylabel('Y/mm')
    plt.title(title)

    plt.grid(True, alpha=0.3)
    # 显示图形
    plt.tight_layout()
    plt.show()


def evaluate_model(model_type):
    """评估模型预测结果"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler = joblib.load('output_scaler.pkl')
    y_true = np.load('output_test_true.npy')
    x_test = np.load('input_test_scaled.npy')
    x_tensor = torch.FloatTensor(x_test).to(device)
    model = None
    model_state_dic = None

    if model_type == 'AlexNet':
        model = alex.AlexNet()
        model.to(device)
        model_state_dic = torch.load('alex_best_model.pth')['model_state_dict']

    elif model_type == 'GoogLeNet':
        model = googlenet.GoogLeNet()
        model.to(device)
        model_state_dic = torch.load('googlenet_best_model.pth')['model_state_dict']

    elif model_type == 'NiN':
        model = nin.NiNNet()
        model.to(device)
        model_state_dic = torch.load('nin_best_model.pth')['model_state_dict']

    elif model_type == 'ResNet':
        model = resnet.ResNet()
        model.to(device)
        model_state_dic = torch.load('resnet_best_model.pth')['model_state_dict']

    elif model_type == 'UNet':
        model = unet.UNet()
        model.to(device)
        model_state_dic = torch.load('unet_best_model.pth')['model_state_dict']

    elif model_type == 'VGG':
        model = vgg.VGGNet()
        model.to(device)
        model_state_dic = torch.load('vgg_best_model.pth')['model_state_dict']

    model.load_state_dict(model_state_dic)
    model.eval()
    with torch.no_grad():
        y_pred = model(x_tensor)
        y_pred = torch.detach(y_pred).cpu().numpy()

    y_pred = batch_denorm_mat(y_pred, scaler)
    res_RMSA = RMSA(y_pred, y_true)
    res_SSIM = SSIM(y_pred, y_true)
    res_IoU = IoU(y_pred, y_true)
    res_FA = FA(y_pred, y_true)
    res_PA = PA(y_pred, y_true)
    res_AA = AA(y_pred, y_true)

    print(f"model name:{model_type}")
    print(f"RMSA:{res_RMSA:.4f}")
    print(f"SSIM:{res_SSIM:.4f}")
    print(f"IoU:{res_IoU:.4f}")
    print(f"FA:{res_FA:.4f}")
    print(f"PA:{res_PA:.4f}")
    print(f"AA:{res_AA:.4f}")
    return {
        'RMSA': res_RMSA,
        'SSIM': res_SSIM,
        'IoU': res_IoU,
        'FA': res_FA,
        'PA': res_PA,
        'AA': res_AA,
    }


if __name__ == "__main__":
    print("=" * 33 + "开始评估" + "=" * 33)
    # AlexNet评估
    print('=' * 33 + ' AlexNet ' + '=' * 33)
    res_alex = evaluate_model('AlexNet')
    # GoogLeNet评估
    print('=' * 33 + ' GoogLeNet ' + '=' * 33)
    res_googlenet = evaluate_model('GoogLeNet')
    # NiN评估
    print('=' * 33 + ' NiN ' + '=' * 33)
    res_nin = evaluate_model('NiN')
    # ResNet评估
    print('=' * 33 + ' ResNet ' + '=' * 33)
    res_resnet = evaluate_model('ResNet')
    # UNet评估
    print('=' * 33 + ' UNet ' + '=' * 33)
    res_unet = evaluate_model('UNet')
    # VGG评估
    print('=' * 33 + ' VGG ' + '=' * 33)
    res_vgg = evaluate_model('VGG')
    print("=" * 33 + "评估完毕" + "=" * 33)
