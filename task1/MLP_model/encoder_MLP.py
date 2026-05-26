import json
import os
import os.path
import re
import joblib
import numpy as np
from scipy.interpolate import splprep, splev
from sklearn.preprocessing import MinMaxScaler


def smooth(x, num=5):
    if num // 2 == 0:
        num -= 1
    length = len(x)
    y = np.zeros(length)
    n = (num - 1) / 2
    for i in range(0, length):
        count_0 = i
        count_end = length - i - 1
        if count_0 in range(0, int(n)) or count_end in range(0, int(n)):
            count = min(count_0, count_end)
            y[i] = np.mean(x[i - count:i + count + 1])
        else:
            y[i] = np.mean(x[i - int(n):i + int(n) + 1])
    return y


def smooth_profile(profile):
    x_arr = profile[:, 0]
    y_arr = profile[:, 1]
    x_arr = smooth(x_arr)
    y_arr = smooth(y_arr)
    return np.column_stack((x_arr, y_arr))


def preprocess_rail_profile(path, target_num=512):
    """预处理钢轨廓形，执行固定采样操作，512点"""
    rail_profile = np.loadtxt(path)
    x = rail_profile[:, 0]
    y = rail_profile[:, 1]
    tck, u = splprep([x, y], s=0, k=3)
    u_new = np.linspace(0, 1, target_num)
    x_new, y_new = splev(u_new, tck)
    return x_new, y_new


def preprocess_wheel_profile(path, target_num=512):
    """预处理车轮廓形，平滑处理，执行固定采样操作，512点"""
    wheel_profile = np.loadtxt(path)
    wheel_profile = smooth_profile(wheel_profile)
    x = wheel_profile[:, 0]
    y = wheel_profile[:, 1]
    tck, u = splprep([x, y], s=0, k=3)
    u_new = np.linspace(0, 1, target_num)
    x_new, y_new = splev(u_new, tck)
    return x_new, y_new


def find_folder_by_index(root_dir, id):
    """根据id获取对应文件路径"""
    folder_path = None
    start_num = 0
    end_num = 0
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        if os.path.isdir(item_path):
            pattern = r'^(\d+)-(\d+)$'
            match = re.match(pattern, item)  # 确保文件夹名称为1_100
            if match:
                start_num = int(item.split('-')[0])
                end_num = int(item.split('-')[1])
                if start_num <= id <= end_num:
                    folder_path = item_path
                    break
    return folder_path, start_num, end_num


def extract_encode_info(root_dir, id):
    """根据Id抽取编码信息"""
    folder_path, start, end = find_folder_by_index(root_dir, id)
    if id < 10000:
        config_path = folder_path + f"\\{id:04d}\\{id:04d}_config.json"
        output_path = folder_path + f"\\{id:04d}\\output.json"
    else:
        config_path = folder_path + f"\\{id}\\{id}_config.json"
        output_path = folder_path + f"\\{id}\\output.json"
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            output = json.load(f)
    # vtu_path
    vtu_path = folder_path + f"\\{id}\\contact_field_1.vtu"
    if os.path.exists(vtu_path):
        gg = config['contact']['gg']
        poiss = config['contact']['poiss']
        fstat = config['contact']['fstat']
        fz = config['contact']['fz']
        gaught = config['contact']['gaught']
        gaugwd = config['contact']['gaugwd']
        cant = config['contact']['cant']
        fbdist = config['contact']['fbdist']
        fbpos = config['contact']['fbpos']
        nomrad = config['contact']['nomrad']
        y_ws = config['contact']['y_ws']
        yaw_ws = config['contact']['yaw_ws']
        roll_ws = config['contact']['roll_ws']
        vs = config['contact']['vs']
        vpitch = config['contact']['vpitch']

        contact_ref_x = output['contact_patch_pos']['xtr'][0]
        contact_ref_y = output['contact_patch_pos']['ytr'][0]
        contact_ref_z = output['contact_patch_pos']['ztr'][0]

        # 输出部分
        contact_fx = output['total_force']['fx_tr'][0]
        contact_fy = output['total_force']['fy_tr'][0]
        contact_fz = output['total_force']['fz_tr'][0]

        wheel_profile = config['contact']['wheel_profile']
        rail_profile = config['contact']['rail_profile']

        wheel_path = root_dir + "\\standard_profiles\\" + wheel_profile
        rail_path = root_dir + "\\standard_profiles\\" + rail_profile

        wheel_pt_x, wheel_pt_y = preprocess_wheel_profile(wheel_path, 512)
        wheel_pt = np.column_stack([wheel_pt_x, wheel_pt_y])
        wheel_pt = wheel_pt.ravel().tolist()

        rail_pt_x, rail_pt_y = preprocess_rail_profile(rail_path, 512)
        rail_pt = np.column_stack([rail_pt_x, rail_pt_y])
        rail_pt = rail_pt.ravel().tolist()

        input_info1 = [gg, poiss, fstat, fz, gaught, gaugwd, cant, fbdist,
                       fbpos, nomrad, y_ws, yaw_ws, roll_ws, vs, vpitch,
                       contact_ref_x, contact_ref_y, contact_ref_z]
        input_info = input_info1 + wheel_pt + rail_pt + [contact_fx, contact_fy, contact_fz]
        print(f"Id:{id}信息抽取完毕")
    else:
        input_info = None
        print(f"Id:{id}信息缺失,跳过操作...")
    return input_info


def encoder(root_dir, train_id, val_id, test_id):
    """信息编码, 输入参数15+接触区域参考中心坐标3+车轮廓形坐标(x1,y1,x2,y2...x512,y512)1024,钢轨廓形坐标(x1,y1,x2,y2...x512,y512)1024"""
    print("=" * 33 + "抽取编码信息" + "=" * 33)
    train_info_list = []
    val_info_list = []
    test_info_list = []
    # ======================= 1. 数据划分 =======================
    # 根据id进行编码
    count = 0
    for inner_id in train_id:
        train_info = extract_encode_info(root_dir, inner_id)
        if train_info is not None:
            train_info_list.append(train_info)
            count += 1
            print(f"已完成训练集信息抽取:[{count}/{len(train_id)}]")
    train_info = np.array(train_info_list, dtype=float)

    count = 0
    for inner_id in val_id:
        val_info = extract_encode_info(root_dir, inner_id)
        if val_info is not None:
            val_info_list.append(val_info)
            count += 1
            print(f"已完成验证集信息抽取:{count}/{len(val_id)}")
    val_info = np.array(val_info_list, dtype=float)

    count = 0
    for inner_id in test_id:
        test_info = extract_encode_info(root_dir, inner_id)
        if test_info is not None:
            test_info_list.append(test_info)
            count += 1
            print(f"已完成测试集信息抽取:{count}/{len(test_id)}")
    test_info = np.array(test_info_list, dtype=float)

    input_len = train_info.shape[1] - 3

    input_train_true = train_info[:, 0:input_len]
    input_val_true = val_info[:, 0:input_len]
    input_test_true = test_info[:, 0:input_len]

    output_train_true = train_info[:, input_len:]
    output_val_true = val_info[:, input_len:]
    output_test_true = test_info[:, input_len:]

    np.save("input_train_true.npy", input_train_true)
    np.save("input_val_true.npy", input_val_true)
    np.save("input_test_true.npy", input_test_true)
    np.save("output_train_true.npy", output_train_true)
    np.save("output_val_true.npy", output_val_true)
    np.save("output_test_true.npy", output_test_true)

    print("=" * 33 + "数据集划分完毕" + "=" * 33)
    # ======================= 2. 输入数据归一化 =======================
    input_scaler = MinMaxScaler()
    input_scaler.fit(input_train_true)
    # 应用到训练集、验证集、测试集
    input_train_scaled = input_scaler.transform(input_train_true)
    input_val_scaled = input_scaler.transform(input_val_true)
    input_test_scaled = input_scaler.transform(input_test_true)
    print("=" * 33 + "归一化输入信息编码完毕" + "=" * 33)

    # ======================= 3. 输出数据归一化 =======================
    output_train_scaled = np.zeros_like(output_train_true)
    output_val_scaled = np.zeros_like(output_val_true)
    output_test_scaled = np.zeros_like(output_test_true)

    output_scaler_forceX = MinMaxScaler()
    output_scaler_forceX.fit(output_train_true[:, [0]])
    output_train_scaled[:, 0] = output_scaler_forceX.transform(output_train_true[:, [0]]).flatten()
    output_val_scaled[:, 0] = output_scaler_forceX.transform(output_val_true[:, [0]]).flatten()
    output_test_scaled[:, 0] = output_scaler_forceX.transform(output_test_true[:, [0]]).flatten()

    output_scaler_forceY = MinMaxScaler()
    output_scaler_forceY.fit(output_train_true[:, [1]])
    output_train_scaled[:, 1] = output_scaler_forceY.transform(output_train_true[:, [1]]).flatten()
    output_val_scaled[:, 1] = output_scaler_forceY.transform(output_val_true[:, [1]]).flatten()
    output_test_scaled[:, 1] = output_scaler_forceY.transform(output_test_true[:, [1]]).flatten()

    output_scaler_forceZ = MinMaxScaler()
    output_scaler_forceZ.fit(output_train_true[:, [2]])
    output_train_scaled[:, 2] = output_scaler_forceZ.transform(output_train_true[:, [2]]).flatten()
    output_val_scaled[:, 2] = output_scaler_forceZ.transform(output_val_true[:, [2]]).flatten()
    output_test_scaled[:, 2] = output_scaler_forceZ.transform(output_test_true[:, [2]]).flatten()
    print("=" * 33 + "归一化输出信息编码完毕" + "=" * 33)

    # ======================= 4. 保存 =======================
    joblib.dump(input_scaler, "input_scaler.pkl")
    joblib.dump(output_scaler_forceX, "output_scaler_forceX.pkl")
    joblib.dump(output_scaler_forceY, "output_scaler_forceY.pkl")
    joblib.dump(output_scaler_forceZ, "output_scaler_forceZ.pkl")

    np.save("input_train_scaled.npy", input_train_scaled)
    np.save("input_val_scaled.npy", input_val_scaled)
    np.save("input_test_scaled.npy", input_test_scaled)
    np.save("output_train_scaled.npy", output_train_scaled)
    np.save("output_val_scaled.npy", output_val_scaled)
    np.save("output_test_scaled.npy", output_test_scaled)
    print("=" * 33 + "归一化编码信息保存完毕" + "=" * 33)


if __name__ == "__main__":
    root_dir = "E:\\steady_rolling_dataset"
    train_id_path = os.path.join(root_dir, "train_id.npy")
    valid_id_path = os.path.join(root_dir, "valid_id.npy")
    test_id_path = os.path.join(root_dir, "test_id.npy")

    train_id = np.load(train_id_path)
    valid_id = np.load(valid_id_path)
    test_id = np.load(test_id_path)
    encoder(root_dir, train_id, valid_id, test_id)
