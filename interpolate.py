import json
import numpy as np


def read_trajectory_from_json(filename):
    """从JSON文件读取轨迹数据"""
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if "points" not in data or not isinstance(data["points"], list):
            raise ValueError("JSON文件格式不正确，应包含'points'键且为列表")
        return data["points"], data.get("post_process", 0)


def interpolate_trajectory(points, target_num):
    """将轨迹插值到目标数量的点"""
    current_num = len(points)
    if current_num < 2:
        raise ValueError("至少需要两个点进行插值")
    if target_num <= current_num:
        return points

    # 提取所有字段的数组
    pixel_x = np.array([p["pixel_x"] for p in points])
    pixel_y = np.array([p["pixel_y"] for p in points])
    x = np.array([p["x"] for p in points])
    y = np.array([p["y"] for p in points])
    yaw = np.array([p["yaw"] for p in points])

    # 计算现有点的累计距离
    distances = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    cum_distances = np.concatenate(([0], np.cumsum(distances)))
    total_distance = cum_distances[-1]

    # 生成插值点的位置
    new_distances = np.linspace(0, total_distance, target_num)

    # 对每个字段进行插值
    new_pixel_x = np.interp(new_distances, cum_distances, pixel_x)
    new_pixel_y = np.interp(new_distances, cum_distances, pixel_y)
    new_x = np.interp(new_distances, cum_distances, x)
    new_y = np.interp(new_distances, cum_distances, y)
    new_yaw = np.interp(new_distances, cum_distances, yaw)

    # 构建新的轨迹点
    new_points = [
        {
            "pixel_x": float(new_pixel_x[i]),
            "pixel_y": float(new_pixel_y[i]),
            "x": float(new_x[i]),
            "y": float(new_y[i]),
            "yaw": float(new_yaw[i])
        }
        for i in range(target_num)
    ]

    return new_points


def save_trajectory(points, post_process, filename="interpolated_trajectory.json"):
    """保存插值后的轨迹"""
    data = {
        "points": points,
        "post_process": post_process
    }
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"插值后的轨迹已保存到 {filename}")


def main():
    # 假设输入文件名为 "input.json"
    input_file = "traj2.json"

    # 读取原始轨迹
    points, post_process = read_trajectory_from_json(input_file)
    print(f"原始轨迹包含 {len(points)} 个点")

    # 插值到50个点
    target_num = 50
    interpolated_points = interpolate_trajectory(points, target_num)
    print(f"插值后轨迹包含 {len(interpolated_points)} 个点")

    # 保存结果
    save_trajectory(interpolated_points, post_process)


if __name__ == "__main__":
    # 将你的JSON数据保存为文件
    input_data = {
        "points": [
            {"pixel_x": 1480.1457763134215, "pixel_y": 1627.9384247558232, "x": 0, "y": -10,
             "yaw": 0.046429467579306785},
            {"pixel_x": 1480.1457763134215, "pixel_y": 1627.9384247558232, "x": 4, "y": -10,
             "yaw": 0.046429467579306785},
            {"pixel_x": 1480.1457763134215, "pixel_y": 1627.9384247558232, "x": 10, "y": -10,
             "yaw": 0.046429467579306785},
            {"pixel_x": 1490.2480753166903, "pixel_y": 1628.1741791528018, "x": 20, "y": -10,
             "yaw": 0.040474794706173517}
        ],
        "post_process": 0
    }
    with open("traj2.json", 'w', encoding='utf-8') as f:
        json.dump(input_data, f, indent=2)

    main()