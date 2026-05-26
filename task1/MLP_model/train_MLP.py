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


def create_mlp_model(input_features, output_features, droup_rate=0.3, model_type=1):
    """建立MLP网络结构"""
    # 2066—256—3
    if model_type == 1:
        model = nn.Sequential(
            # 第一层
            nn.Linear(input_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(droup_rate),
            # 输出层,不在输出层添加Droupout和BatchNorm
            nn.Linear(256, output_features))

    elif model_type == 2:
        # 2066—512—128—3
        model = nn.Sequential(
            # 第一层
            nn.Linear(input_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(droup_rate),
            # 第二层
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(droup_rate),
            # 输出层,不在输出层添加Droupout和BatchNorm
            nn.Linear(128, output_features))

    elif model_type == 3:
        # 2066—512—256—64—3
        model = nn.Sequential(
            # 第一层
            nn.Linear(input_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(droup_rate),
            # 第二层
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(droup_rate),
            # 第三层
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(droup_rate),
            # 输出层,不在输出层添加Droupout和BatchNorm
            nn.Linear(64, output_features))
    else:
        # 2066—1024—256—128—64—3
        model = nn.Sequential(
            # 第一层
            nn.Linear(input_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(droup_rate),
            # 第二层
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(droup_rate),
            # 第三层
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(droup_rate),
            # 第四层
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(droup_rate),
            # 输出层,不在输出层添加Droupout和BatchNorm
            nn.Linear(64, output_features))
    return model


def prepare_loader(input_train, input_val, output_train, output_val, batch_size=256):
    """准备DataLoader"""
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


def initialize_model_weights(model):
    """初始化网络权重"""
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)


def train_model(model, train_loader, val_loader, num_epochs=250, learning_rate=1e-3):
    """训练模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    # 定义日志文件路径
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f"train_log_{timestamp}.log"
    # 优化器与调度器
    lr_patience = 10  # 连续十次valid_loss未改善，则更新学习率，lrx0.5
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=lr_patience)
    # 设置早停变量
    early_stop_patience = 20

    patience_counter = 0
    best_model_path = f"MLP_best_model.pth"
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
    best_model_state_dict = model.state_dict()
    best_optimizer_state_dict = optimizer.state_dict()
    best_lr = optimizer.param_groups[0]['lr']
    # 定义损失函数
    criterion = nn.MSELoss()

    start_time = time.time()
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        model.train()
        train_total_loss = 0.0
        train_num_batches = len(train_loader)
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()  # 清空模型中所有可训练参数梯度
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_total_loss += loss.item()
        train_loss = train_total_loss / train_num_batches

        # 验证集
        model.eval()
        val_total_loss = 0.0
        val_num_batches = len(val_loader)
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
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
    # 创建MLP模型
    input_features = input_train_mat.shape[1]
    output_features = output_train_mat.shape[1]
    model = create_mlp_model(input_features, output_features, droup_rate=0.2, model_type=1)
    print("=" * 33 + "MLP网络建立完毕" + "=" * 33)
    # 初始化权重
    initialize_model_weights(model)
    print("=" * 33 + "网络权重初始化完毕" + "=" * 33)
    # 计算模型参数量
    print(f"MLP网络结构如下:{model}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量:{total_params}, 模型可训练参数量:{trainable_params}")
    # 训练模型
    mlp_model, mlp_history = train_model(model, train_loader, val_loader, num_epochs=250, learning_rate=1e-3)
