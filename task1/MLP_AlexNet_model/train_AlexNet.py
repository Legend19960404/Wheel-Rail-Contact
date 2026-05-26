import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.init as init


def create_cnn_branch():
    cnn = nn.Sequential(
        # 第一层卷积：输入1x1024 -> 卷积 -> ReLU -> 最大池化
        nn.Conv1d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=3, stride=2),

        # 第二层卷积：增加通道数
        nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2),
        nn.ReLU(inplace=True),
        nn.MaxPool1d(kernel_size=3, stride=2),

        # 第三层卷积：进一步提取特征
        nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),

        # 第四层卷积：继续提取特征
        nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),

        # 第五层卷积：最终特征提取
        nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool1d(output_size=4),  # 自适应平均池化到长度4

        # 展平后通过线性层映射到512维
        nn.Flatten(),
        nn.Linear(128 * 4, 512),
        nn.ReLU(inplace=True)
    )
    return cnn


def create_mlp_head():
    """创建mlp头部: 512+512+15+3=1042"""
    mlp = nn.Sequential(
        nn.Linear(1042, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(256, 3)
    )
    return mlp


def create_network():
    """组装整个网络"""
    wheel_cnn_branch = create_cnn_branch()
    wheel_cnn_branch.apply(initialize_weights)

    rail_cnn_branch = create_cnn_branch()
    rail_cnn_branch.apply(initialize_weights)

    mlp_head = create_mlp_head()
    mlp_head.apply(initialize_weights)

    def forward(x1, x2, additional_features):
        """
            前向传播函数
            :param x1: 第一个cnn分支输入，形状(batch_size, 1024)
            :param x2: 第二个cnn分支输入，形状(batch_size, 1024)
            :param additional_features: 额外特征，形状(batch_size, 18)
            :return: 输出，形状(batch_size, 3)
        """
        x1 = x1.unsqueeze(1)  # (batch_size, 1, 1024)
        x2 = x2.unsqueeze(1)  # (batch_size, 1, 1024)
        # 通过各自的CNN分支
        wheel_feat = wheel_cnn_branch(x1)  # (batch_size, 512)
        rail_feat = rail_cnn_branch(x2)  # (batch_size, 512)
        # 拼接两个512维特征 + 18维额外特征
        combined_features = torch.cat([wheel_feat, rail_feat, additional_features], dim=1)  # (batch_size, 1042)
        # 通过MLP头得到最终输出
        output = mlp_head(combined_features)  # (batch_size, 3)
        return output

    return {
        'wheel_branch': wheel_cnn_branch,
        'rail_branch': rail_cnn_branch,
        'mlp_head': mlp_head,
        'forward': forward
    }


def get_total_params(model_dict):
    """计算模型总参数量"""
    total_params = 0
    trainable_params = 0
    for k, v in model_dict.items():
        if hasattr(v, 'parameters') and k not in ['forward']:
            for p in v.parameters():
                total_params += p.numel()
                if p.requires_grad:
                    trainable_params += p.numel()
    return total_params, trainable_params


def initialize_weights(m):
    """初始化网络权重"""
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.BatchNorm1d):
        init.constant_(m.weight, 1.0)
        init.constant_(m.bias, 0.0)
    # 其他层（ReLU, Dropout, Pooling等）无需初始化


def prepare_loader(input_train, input_val, output_train, output_val, batch_size=64):
    X_train_tensor = torch.FloatTensor(input_train)
    X_val_tensor = torch.FloatTensor(input_val)
    y_train_tensor = torch.FloatTensor(output_train)
    y_val_tensor = torch.FloatTensor(output_val)
    # 创建数据集和数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    return train_loader, val_loader


def train_model(network_components, train_loader, val_loader, num_epochs=250, learning_rate=1e-3):
    """训练模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 网络组件
    wheel_branch = network_components['wheel_branch'].to(device)
    rail_branch = network_components['rail_branch'].to(device)
    mlp_head = network_components['mlp_head'].to(device)
    forward_func = network_components['forward']

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f"train_log_{timestamp}_alex.log"
    # 优化器与调度器
    lr_patience = 10  # 连续十次valid_loss未改善，则更新学习率，lrx0.5
    optimizer = optim.Adam(
        list(wheel_branch.parameters()) + list(rail_branch.parameters()) + list(mlp_head.parameters()),
        lr=learning_rate,
        weight_decay=1e-4
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=lr_patience)
    # 设置早停变量
    early_stop_patience = 20
    patience_counter = 0
    best_model_path = f"Alex_best_model.pth"
    min_delta = 1e-5
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
    # 用于模型保存的变量
    best_val_loss = float('inf')
    best_epoch = 0
    best_model_state_dict = {
        'wheel_branch_state': copy.deepcopy(wheel_branch.state_dict()),
        'rail_branch_state': copy.deepcopy(rail_branch.state_dict()),
        'mlp_head_state': copy.deepcopy(mlp_head.state_dict())
    }
    best_optimizer_state_dict = copy.deepcopy(optimizer.state_dict())
    best_lr = optimizer.param_groups[0]['lr']
    # 定义损失函数
    criterion = nn.MSELoss()
    start_time = time.time()
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        wheel_branch.train()
        rail_branch.train()
        mlp_head.train()
        train_total_loss = 0.0
        train_num_batches = len(train_loader)
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            add_features = inputs[:, :18]
            inputs_x1 = inputs[:, 18:18 + 1024]
            inputs_x2 = inputs[:, 18 + 1024:]
            optimizer.zero_grad()  # 清空模型中所有可训练参数梯度
            outputs = forward_func(inputs_x1, inputs_x2, add_features)
            loss = criterion(outputs, targets)
            loss.backward()
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(list(wheel_branch.parameters()) + list(rail_branch.parameters()) + list(mlp_head.parameters()),
                                           max_norm=1.0)
            optimizer.step()
            train_total_loss += loss.item()
        train_loss = train_total_loss / train_num_batches

        # 验证阶段
        wheel_branch.eval()
        rail_branch.eval()
        mlp_head.eval()
        val_total_loss = 0.0
        val_num_batches = len(val_loader)
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                add_features = inputs[:, :18]
                inputs_x1 = inputs[:, 18:18 + 1024]
                inputs_x2 = inputs[:, 18 + 1024:]
                outputs = forward_func(inputs_x1, inputs_x2, add_features)
                loss = criterion(outputs, targets)
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
            best_model_state_dict = {
                'wheel_branch_state': copy.deepcopy(wheel_branch.state_dict()),
                'rail_branch_state': copy.deepcopy(rail_branch.state_dict()),
                'mlp_head_state': copy.deepcopy(mlp_head.state_dict())
            }
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
    return network_components, history


if __name__ == "__main__":
    print("=" * 33 + "读取数据编码" + "=" * 33)
    input_train_mat = np.load("input_train_scaled.npy")
    input_val_mat = np.load("input_val_scaled.npy")
    output_train_mat = np.load("output_train_scaled.npy")
    output_val_mat = np.load("output_val_scaled.npy")
    # 输出训练集、测试集输入输出维度
    print("=" * 33 + "读取完毕" + "=" * 33)
    print(f"训练集输入形状:{input_train_mat.shape}")
    print(f"验证集输入形状:{input_val_mat.shape}")
    print(f"训练集输出形状:{output_train_mat.shape}")
    print(f"验证集输出形状:{output_val_mat.shape}")
    train_loader, val_loader = prepare_loader(input_train_mat, input_val_mat, output_train_mat, output_val_mat)
    # 创建AlexNet模型
    network = create_network()
    print("=" * 33 + "MLP+AlexNet网络建立完毕" + "=" * 33)
    print("=" * 33 + "网络权重初始化完毕" + "=" * 33)
    # 计算模型参数量
    print(f"MLP网络结构如下:{network}")
    total_params, trainable_params = get_total_params(network)
    print(f"模型总参数量:{total_params}, 模型可训练参数量:{trainable_params}")
    # 训练模型
    network, history = train_model(network, train_loader, val_loader, num_epochs=250, learning_rate=1e-3)
