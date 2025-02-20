import json
import math
import matplotlib.pyplot as plt


def generate_trajectory1(num_points=25):
    """生成第一条轨迹：从左下到右上，带有弧度和转弯"""
    trajectory = []
    x, y = 0.0, 0.0

    # 第一段：向上弯曲
    for i in range(8):
        x = i * 2.0
        y = math.sin(x * 0.5) * 5 + i
        trajectory.append({"x": round(x, 2), "y": round(y, 2)})

    # 第二段：向右上转弯
    for i in range(8):
        angle = math.pi / 4 + i * (math.pi / 8) / 7
        x = 16.0 + math.cos(angle) * (i + 2)
        y = 8.0 + math.sin(angle) * (i + 2)
        trajectory.append({"x": round(x, 2), "y": round(y, 2)})

    # 第三段：向下弯曲
    for i in range(9):
        x = 24.0 + i * 1.5
        y = 16.0 - math.cos(i * 0.3) * 4 - i * 0.5
        trajectory.append({"x": round(x, 2), "y": round(y, 2)})

    return trajectory


def generate_trajectory2(num_points=25):
    """生成第二条轨迹：从左上到右下，与第一条交叉"""
    trajectory = []
    x, y = 0.0, 20.0

    # 第一段：向右下弧形
    for i in range(8):
        x = i * 2.5
        y = 20.0 - math.cos(i * 0.4) * 6
        trajectory.append({"x": round(x, 2), "y": round(y, 2)})

    # 第二段：向下转弯（交叉区域）
    for i in range(8):
        angle = math.pi / 2 + i * (math.pi / 6) / 7
        x = 20.0 + math.sin(angle) * (i + 1)
        y = 14.0 - math.cos(angle) * (i + 1)
        trajectory.append({"x": round(x, 2), "y": round(y, 2)})

    # 第三段：向右下平滑曲线
    for i in range(9):
        x = 28.0 + i * 1.8
        y = 6.0 - math.sin(i * 0.2) * 2 - i * 0.3
        trajectory.append({"x": round(x, 2), "y": round(y, 2)})

    return trajectory


def save_to_json(trajectory1, trajectory2, filename="trajectories.json"):
    with open("trajectories1.json", 'w', encoding='utf-8') as f:
        json.dump(trajectory1, f, indent=2, ensure_ascii=False)
    with open("trajectories2.json", 'w', encoding='utf-8') as f:
        json.dump(trajectory2, f, indent=2, ensure_ascii=False)


def visualize_trajectories(trajectory1, trajectory2):
    """可视化两条轨迹"""
    # 提取x和y坐标
    x1 = [point["x"] for point in trajectory1]
    y1 = [point["y"] for point in trajectory1]
    x2 = [point["x"] for point in trajectory2]
    y2 = [point["y"] for point in trajectory2]

    # 创建图像
    plt.figure(figsize=(10, 6))

    # 绘制第一条轨迹
    plt.plot(x1, y1, 'b-', label='Trajectory 1', marker='o', markersize=4)
    # 绘制第二条轨迹
    plt.plot(x2, y2, 'r-', label='Trajectory 2', marker='o', markersize=4)

    plt.axis('equal')

    # 添加标题和标签
    plt.title('Two Intersecting Trajectories')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)


    # 显示图像
    plt.show()


def main():
    # 生成两条轨迹
    traj1 = generate_trajectory1()
    traj2 = generate_trajectory2()

    # 保存到JSON文件
    save_to_json(traj1, traj2)
    print(f"已生成两条轨迹，每条包含 {len(traj1)} 个点，并保存到 trajectories.json")

    # 可视化轨迹
    visualize_trajectories(traj1, traj2)


if __name__ == "__main__":
    main()