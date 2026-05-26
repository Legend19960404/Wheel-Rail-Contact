import json
import re
import joblib
import numpy as np
import vtk
import vtkmodules.util.numpy_support as nps
import os
from scipy.interpolate import splrep, splev
from scipy.interpolate import griddata
from sklearn.preprocessing import MinMaxScaler


def get_grid_bound(root_dir):
    """获取最大区域范围"""
    mx_arr = []
    my_arr = []
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        if os.path.isdir(item_path):
            pattern = r'^(\d+)-(\d+)$'
            match = re.match(pattern, item)  # 确保文件夹名称为1-10000
            if match:
                start_idx = item.split('-')[0]
                end_idx = item.split('-')[1]
                report_json_path = os.path.join(item_path, f"report_{start_idx}-{end_idx}.json")
                if os.path.exists(report_json_path):
                    with open(report_json_path, 'r', encoding='utf-8') as f:
                        report = json.load(f)
                    info_list = report['info']
                    for info in info_list:
                        Id = info['Id']
                        mx = info['mx']
                        my = info['my']
                        mx_arr.append(mx)
                        my_arr.append(my)
                        print(f"Id:{Id}, mx:{mx}, my:{my}")
            max_mx = max(mx_arr)
            max_my = max(my_arr)
            grid_boundary = {
                'max_mx': max_mx,
                'max_my': max_my,
                'dx': 0.2,
                'dy': 0.2
            }
            json_str = json.dumps(grid_boundary, indent=4)
            with open("grid_status.json", 'w', encoding='utf-8') as f:
                f.write(json_str)
            return grid_boundary


def norm_field(vtu_file_path, mx=100, my=100, dx=0.2, dy=0.2, resolution=50):
    """将vtu文件插值到固定范围内"""
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(vtu_file_path)
    reader.Update()
    # 获取点坐标及法向应力数据
    data = reader.GetOutput()
    points_vtk = data.GetPoints()
    points_array = nps.vtk_to_numpy(points_vtk.GetData())
    pn_data = data.GetPointData().GetArray('pn')
    pn_array = nps.vtk_to_numpy(pn_data)
    x_coords = np.linspace(-mx / 2 * dx, mx / 2 * dx, resolution)
    y_coords = np.linspace(-my / 2 * dy, my / 2 * dy, resolution)
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)

    src_points = points_array[:, :2]
    src_values = pn_array
    grid_pn = griddata(
        src_points,
        src_values,
        (grid_x, grid_y),
        method='cubic',
        fill_value=0.0
    )
    return grid_pn


def norm_profile(path, start_x, end_x, target=1024):
    """规范化车轮廓形至1024点"""
    wheel_profile = np.loadtxt(path)
    x = wheel_profile[:, 0]
    clip_data = wheel_profile[(x >= start_x) & (x <= end_x)]
    sorted_indices = np.argsort(clip_data[:, 0])
    clip_data = clip_data[sorted_indices]

    _, unique_indices = np.unique(clip_data[:, 0], return_index=True)
    clip_data = clip_data[unique_indices]

    tck = splrep(clip_data[:, 0], clip_data[:, 1], s=0, k=3)
    interp_x = np.linspace(start_x, end_x, target)
    interp_y = splev(interp_x, tck)
    return interp_y


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
    # ======================= 抽取输入信息 =======================
    if id < 10000:
        id_name = f'{id:04d}'
    else:
        id_name = f'{id}'
    config_path = folder_path + f"\\{id_name}\\{id_name}_config.json"
    output_path = folder_path + f"\\{id_name}\\output.json"
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            output = json.load(f)
    # vtu_path
    vtu_path = folder_path + f"\\{id_name}\\contact_field_1.vtu"
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

        wheel_profile = config['contact']['wheel_profile']
        rail_profile = config['contact']['rail_profile']

        wheel_path = root_dir + "\\standard_profiles\\" + wheel_profile
        rail_path = root_dir + "\\standard_profiles\\" + rail_profile

        wheel_pt_z = norm_profile(wheel_path, -60, 50, target=1024)
        wheel_pt_z = np.asarray(wheel_pt_z).ravel().tolist()

        rail_pt_z = norm_profile(rail_path, -30, 30, target=1024)
        rail_pt_z = np.asarray(rail_pt_z).ravel().tolist()

        input_info1 = [gg, poiss, fstat, fz, gaught, gaugwd, cant, fbdist,
                       fbpos, nomrad, y_ws, yaw_ws, roll_ws, vs, vpitch,
                       contact_ref_x, contact_ref_y, contact_ref_z]
        input_info = input_info1 + wheel_pt_z + rail_pt_z
        print(f"Id:{id_name}输入信息抽取完毕")
        # ======================= 抽取输出信息 =======================
        grid_boundary_path = "grid_status.json"
        if os.path.exists(grid_boundary_path):
            with open(grid_boundary_path, 'r', encoding='utf-8') as f:
                grid = json.load(f)
        else:
            grid = get_grid_bound(root_dir)
        max_mx = grid['max_mx']
        max_my = grid['max_my']
        dx = grid['dx']
        dy = grid['dy']

        field_path = vtu_path
        output_info = norm_field(field_path, max_mx, max_my, dx, dy)
        print(f"Id:{id_name}输出信息抽取完毕")
    else:
        input_info = None
        output_info = None
        print(f"Id:{id_name}输入/输出信息缺失,跳过操作...")

    return input_info, output_info


def batch_norm_mat(data_list, scaler):
    """批量规范化矩阵"""
    matrices_array = np.stack(data_list)
    flattened = matrices_array.reshape(len(data_list), -1)  # (n, 2500)
    normalized_flat = scaler.transform(flattened)  # (n, 2500)
    normalized_3d = normalized_flat.reshape(len(data_list), 50, 50)  # (n, 50, 50)
    return normalized_3d


def encoder(root_dir, train_id, val_id, test_id):
    """对输入输出编码,输入参数15+车轮廓形固定区域坐标z:1024,钢轨廓形固定区域坐标z:1024"""
    print("=" * 33 + "抽取编码信息" + "=" * 33)
    input_train_list = []
    input_val_list = []
    input_test_list = []

    output_train_list = []
    output_val_list = []
    output_test_list = []
    # ======================= 1.数据划分 =======================
    # 根据id编码
    count = 0
    for inner_id in train_id:
        input_train_info, output_train_info = extract_encode_info(root_dir, inner_id)
        if input_train_info and output_train_info is not None:
            input_train_list.append(input_train_info)
            output_train_list.append(output_train_info)
            count += 1
            print(f"已完成训练集信息抽取:[{count}/{len(train_id)}]")
    input_train_true = np.array(input_train_list)
    output_train_true = np.array(output_train_list)

    count = 0
    for inner_id in val_id:
        input_val_info, output_val_info = extract_encode_info(root_dir, inner_id)
        if input_val_info and output_val_info is not None:
            input_val_list.append(input_val_info)
            output_val_list.append(output_val_info)
            count += 1
            print(f"已完成验证集信息抽取:{count}/{len(val_id)}")
    input_val_true = np.array(input_val_list)
    output_val_true = np.array(output_val_list)

    count = 0
    for inner_id in test_id:
        input_test_info, output_test_info = extract_encode_info(root_dir, inner_id)
        if input_test_info and output_test_info is not None:
            input_test_list.append(input_test_info)
            output_test_list.append(output_test_info)
            count += 1
            print(f"已完成测试集信息抽取:{count}/{len(test_id)}")
    input_test_true = np.array(input_val_list)
    output_test_true = np.array(output_val_list)

    np.save("input_train_true.npy", input_train_true)
    np.save("input_val_true.npy", input_val_true)
    np.save("input_test_true.npy", input_test_true)
    np.save("output_train_true.npy", output_train_true)
    np.save("output_val_true.npy", output_val_true)
    np.save("output_test_true.npy", output_test_true)

    print("=" * 33 + "数据集划分完毕" + "=" * 33)
    # ======================= 2.输入数据归一化 =======================
    input_scaler = MinMaxScaler()
    input_scaler.fit(input_train_true)
    # 应用到训练集、验证集、测试集
    input_train_scaled = input_scaler.transform(input_train_true)
    input_val_scaled = input_scaler.transform(input_val_true)
    input_test_scaled = input_scaler.transform(input_test_true)
    print("=" * 33 + "归一化输入信息编码完毕" + "=" * 33)
    # ======================= 3.输出数据归一化 =======================
    flattened_train = output_train_true.reshape(len(output_train_true), -1)  # (n, 2500)
    output_scaler = MinMaxScaler()
    output_scaler.fit(flattened_train)

    output_train_scaled = batch_norm_mat(output_train_true, output_scaler)
    output_val_scaled = batch_norm_mat(output_val_true, output_scaler)
    output_test_scaled = batch_norm_mat(output_test_true, output_scaler)
    print("=" * 33 + "归一化输出信息编码完毕" + "=" * 33)
    # ======================= 4.保存 =======================
    joblib.dump(input_scaler, "input_scaler.pkl")
    joblib.dump(output_scaler, "output_scaler.pkl")

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
