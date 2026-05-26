import copy
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime


class NiNNet(nn.Module):
    def __init__(self, profile_len=1024, param_dim=18):
        super().__init__()
        # 廓形分支 (2通道 x 1024)
        # NiN风格,车轮、钢轨廓形对应两个通道,1D图像
        self.profile_branch = nn.Sequential(
            nn.Conv1d(2, 128, kernel_size=5, stride=2, padding=2),  # 降低空间维度
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 96, kernel_size=1),  # 1x1卷积降维
            nn.BatchNorm1d(96),
            nn.ReLU(inplace=True),
            nn.Conv1d(96, 64, kernel_size=1),  # 1x1卷积进一步降维
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 192, kernel_size=5, stride=2, padding=2),  # 降低空间维度
            nn.BatchNorm1d(192),
            nn.ReLU(inplace=True),
            nn.Conv1d(192, 144, kernel_size=1),  # 1x1卷积降维
            nn.BatchNorm1d(144),
            nn.ReLU(inplace=True),
            nn.Conv1d(144, 128, kernel_size=1),  # 1x1卷积进一步降维
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),  # 降低空间维度
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 192, kernel_size=1),  # 1x1卷积降维
            nn.BatchNorm1d(192),
            nn.ReLU(inplace=True),
            nn.Conv1d(192, 192, kernel_size=1),  # 1x1卷积保持维度
            nn.BatchNorm1d(192),
            nn.ReLU(inplace=True),

            nn.Conv1d(192, 384, kernel_size=3, stride=2, padding=1),  # 降低空间维度
            nn.BatchNorm1d(384),
            nn.ReLU(inplace=True),
            nn.Conv1d(384, 256, kernel_size=1),  # 1x1卷积升维
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=1),  # 1x1卷积保持维度
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1),  # 降低空间维度
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 384, kernel_size=1),  # 1x1卷积保持维度
            nn.BatchNorm1d(384),
            nn.ReLU(inplace=True),
            nn.Conv1d(384, 384, kernel_size=1),  # 1x1卷积保持维度
            nn.BatchNorm1d(384),
            nn.ReLU(inplace=True),

            # 全局平均池化替代最后的分类器
            nn.AdaptiveAvgPool1d(8),
            nn.Flatten(),

            # 最终特征映射
            nn.Linear(3072, 1024),  # 降维到合适的特征维度
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        # 参数分支 (18维)
        self.param_branch = nn.Sequential(
            # nn.Linear(18, 64),
            # nn.BatchNorm1d(64),
            # nn.ReLU(inplace=True),
            # nn.Linear(64, 128),
            # nn.BatchNorm1d(128),
            # nn.ReLU(inplace=True)
            nn.Linear(18, 256),
            nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512), nn.ReLU(inplace=True)
        )
        # 计算融合层的输入维度
        with torch.no_grad():
            self.profile_branch.eval()
            self.param_branch.eval()
            dummy_profile = torch.zeros(1, 2, profile_len)
            dummy_param = torch.zeros(1, param_dim)
            profile_feat = self.profile_branch(dummy_profile)
            param_feat = self.param_branch(dummy_param)
            fusion_in_dim = profile_feat.size(1) + param_feat.size(1)
            # 恢复训练模式（后续训练时会统一设置，此处可选）
            self.profile_branch.train()
            self.param_branch.train()
        # 融合头 - 将NiN特征和参数特征结合
        self.fusion = nn.Sequential(
            # nn.Linear(fusion_in_dim, 256),
            # nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            # nn.Dropout(0.4),
            # nn.Linear(256, 128),
            # nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            # nn.Dropout(0.4),
            # nn.Linear(128, 2500)  # 50 * 50 = 2500
            nn.Linear(fusion_in_dim, 1024),
            nn.BatchNorm1d(1024), nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 2500)
        )

    def forward(self, x):
        params = x[:, :18]
        wheel = x[:, 18:18 + 1024]
        rail = x[:, 18 + 1024:]
        profiles = torch.stack([wheel, rail], dim=1)  # [B, 2, 1024]
        prof_feat = self.profile_branch(profiles)
        param_feat = self.param_branch(params)
        fused = torch.cat([prof_feat, param_feat], dim=1)
        result = self.fusion(fused)
        result = result.reshape(-1, 50, 50)
        return result


def get_param_number(model):
    """获取模型参数总量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def init_param(m):
    """初始化网络权重"""
    # 卷积层初始化
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    # 全连接层初始化
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    # BatchNorm层初始化
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.constant_(m.weight, 1.0)  # gamma=1
        nn.init.constant_(m.bias, 0.0)  # beta=0
        if hasattr(m, 'running_mean'):
            m.running_mean.zero_()
        if hasattr(m, 'running_var'):
            m.running_var.fill_(1)
    # LayerNorm初始化（如有）
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0.0)
        nn.init.constant_(m.weight, 1.0)


def get_loader(input_train, input_val, output_train, output_val, batch_size=256):
    """获取DataLoader"""
    x_train_tensor = torch.FloatTensor(input_train)
    x_val_tensor = torch.FloatTensor(input_val)
    y_train_tensor = torch.FloatTensor(output_train)
    y_val_tensor = torch.FloatTensor(output_val)
    # 创建数据集和数据加载器
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(x_val_tensor, y_val_tensor)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    return train_loader, val_loader


def train_model(model, train_loader, val_loader, num_epochs=250):
    """模型训练"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f"train_log_{timestamp}_alex.log"

    # optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)  # L2正则
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    min_delta = 1e-5
    # 连续15次valid_loss未改善，则更新学习率，lrx0.5
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-7, threshold=min_delta
    )
    early_stop_patience = 30
    patience_counter = 0
    best_model_path = f"nin_best_model_{timestamp}.pth"

    # 存储训练的历史记录
    history = {
        'epoch_idx': [],
        'train_loss': [],
        'valid_loss': [],
        'lr': [],
        'epoch_time': [],
        'total_time': 0.0
    }
    print("=" * 33 + "开始训练" + "=" * 33)
    best_val_loss = float('inf')
    best_epoch = 0
    best_model_state_dict = None
    best_optimizer_state_dict = None
    best_lr = 0
    criterion = nn.MSELoss()
    start_time = time.time()
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        model.train()
        train_total_loss = 0.0
        train_num_batches = len(train_loader)
        for x_true, y_true in train_loader:
            x_true, y_true = x_true.to(device), y_true.to(device)
            optimizer.zero_grad()
            y_pred = model(x_true)
            loss = criterion(y_pred, y_true)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_total_loss += loss.item()
        train_loss = train_total_loss / train_num_batches
        # 验证阶段
        model.eval()
        val_total_loss = 0.0
        val_num_batches = len(val_loader)
        with torch.no_grad():
            for x_true, y_true in val_loader:
                x_true, y_true = x_true.to(device), y_true.to(device)
                y_pred = model(x_true)
                loss = criterion(y_pred, y_true)
                val_total_loss += loss.item()
        val_loss = val_total_loss / val_num_batches
        # 学习率调度
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        # 模型更新判断
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            # 更新最佳保存参数
            best_epoch = epoch
            best_model_state_dict = copy.deepcopy(model.state_dict())
            best_optimizer_state_dict = copy.deepcopy(optimizer.state_dict())
            best_lr = current_lr
            status_msg = f"最佳模型已更新,val_loss:{val_loss:6f}"
        else:
            patience_counter += 1
            status_msg = f"验证集损失未改善:{patience_counter}/{early_stop_patience}"
        epoch_time = time.time() - epoch_start
        # 保存训练过程信息
        history['epoch_idx'].append(epoch)
        history['train_loss'].append(train_loss)
        history['valid_loss'].append(val_loss)
        history['lr'].append(current_lr)
        history['epoch_time'].append(epoch_time)
        # 打印训练信息
        print("=" * 33 + f"Epoch [{epoch}/{num_epochs}]" + "=" * 33)
        print(f"Train Loss:{train_loss:6f}")
        print(f"Valid Loss:{val_loss:6f}")
        print(f"Learning Rate:{current_lr:6f}")
        print(f"Epoch Time:{epoch_time:6f}s")
        print(f"Status Message:{status_msg}")
        # 早停逻辑判断，决定是否终止学习
        if patience_counter >= early_stop_patience:
            print(f"早停触发,停止学习，best_val_loss:{best_val_loss:6f}")
            break
    total_time = time.time() - start_time
    history['total_time'] = total_time
    # 写日志
    json_str = json.dumps(history, indent=4)
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(json_str)
    # 保存模型
    torch.save({
        'epoch': best_epoch,
        'model_state_dict': best_model_state_dict,
        'optimizer_state_dict': best_optimizer_state_dict,
        'val_loss': best_val_loss,
        'lr': best_lr,
        'timestamp': timestamp
    }, best_model_path)
    print("=" * 33 + f"训练完毕,总用时:{total_time:6f}s" + "=" * 33)
    return model, history


if __name__ == "__main__":
    print("=" * 33 + "读取数据编码" + "=" * 33)
    input_train_mat = np.load("../input_train_scaled.npy")
    input_val_mat = np.load("../input_val_scaled.npy")
    output_train_mat = np.load("../output_train_scaled.npy")
    output_val_mat = np.load("../output_val_scaled.npy")
    # 输出训练集、测试集输入输出维度
    print("=" * 33 + "读取完毕" + "=" * 33)
    print(f"训练集输入形状:{input_train_mat.shape}")
    print(f"验证集输入形状:{input_val_mat.shape}")
    print(f"训练集输出形状:{output_train_mat.shape}")
    print(f"验证集输出形状:{output_val_mat.shape}")
    train_loader, val_loader = get_loader(input_train_mat, input_val_mat, output_train_mat, output_val_mat)
    print("=" * 33 + "NiN网络建立完毕" + "=" * 33)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nin_model = NiNNet().to(device)
    print("=" * 33 + "网络权重初始化完毕" + "=" * 33)
    nin_model.apply(lambda m: init_param(m))
    print(f"NiN网络结构如下:{nin_model}")
    total, trainable = get_param_number(nin_model)
    print(f"模型总参数量:{total}, 模型可训练参数量:{trainable}")
    # 训练模型
    train_model(nin_model, train_loader, val_loader, num_epochs=300)
