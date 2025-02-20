import json
import matplotlib.pyplot as plt

def read_trajectory_from_json(filename):
    """从JSON文件读取轨迹数据，仅提取x和y坐标"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if "points" not in data or not isinstance(data["points"], list):
                raise ValueError("JSON文件格式不正确，应包含'points'键且为列表")
            trajectory = [{"x": point["x"], "y": point["y"]} for point in data["points"]]
            return trajectory
    except FileNotFoundError:
        print(f"错误：文件 {filename} 未找到")
        return []
    except json.JSONDecodeError:
        print(f"错误：文件 {filename} 不是有效的JSON格式")
        return []
    except (KeyError, ValueError) as e:
        print(f"错误：{e}")
        return []
def lines_intersect(p1, p2, p3, p4):
    def ccw(A, B, C):
        val = (C["y"] - A["y"]) * (B["x"] - A["x"]) - (B["y"] - A["y"]) * (C["x"] - A["x"])
        return val > 1e-10  # 添加容差，避免浮点误差

    def get_intersection_point(p1, p2, p3, p4):
        x1, y1 = p1["x"], p1["y"]
        x2, y2 = p2["x"], p2["y"]
        x3, y3 = p3["x"], p3["y"]
        x4, y4 = p4["x"], p4["y"]

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:  # 处理平行或接近平行
            return None

        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom

        # 放宽边界检查
        eps = 1e-6
        if (min(x1, x2) - eps <= px <= max(x1, x2) + eps and
            min(y1, y2) - eps <= py <= max(y1, y2) + eps and
            min(x3, x4) - eps <= px <= max(x3, x4) + eps and
            min(y3, y4) - eps <= py <= max(y3, y4) + eps):
            return {"x": px, "y": py}
        return None

    if (ccw(p1, p3, p4) != ccw(p2, p3, p4) and
        ccw(p1, p2, p3) != ccw(p1, p2, p4)):
        return get_intersection_point(p1, p2, p3, p4)
    return None

def find_intersections(trajectory1, trajectory2):
    """找到两条轨迹的所有交叉点及其对应线段"""
    intersections = []
    for i in range(len(trajectory1) - 1):
        for j in range(len(trajectory2) - 1):
            intersect = lines_intersect(
                trajectory1[i], trajectory1[i + 1],
                trajectory2[j], trajectory2[j + 1]
            )
            if intersect:
                intersections.append({
                    "point": intersect,
                    "seg1": (i, i + 1),
                    "seg2": (j, j + 1)
                })
    return intersections

def build_graph(trajectory1, trajectory2, intersections):
    """构建图结构并返回节点和边"""
    def point_to_str(p):
        return f"{p['x']:.6f},{p['y']:.6f}"

    nodes = {}
    edges = []

    for i in range(len(trajectory1)):
        node_id = point_to_str(trajectory1[i])
        nodes[node_id] = trajectory1[i]
        if i > 0:
            edges.append([point_to_str(trajectory1[i - 1]), node_id])

    for i in range(len(trajectory2)):
        node_id = point_to_str(trajectory2[i])
        nodes[node_id] = trajectory2[i]
        if i > 0:
            edges.append([point_to_str(trajectory2[i - 1]), node_id])

    for inter in intersections:
        inter_id = point_to_str(inter["point"])
        nodes[inter_id] = inter["point"]
        p1 = point_to_str(trajectory1[inter["seg1"][0]])
        p2 = point_to_str(trajectory1[inter["seg1"][1]])
        p3 = point_to_str(trajectory2[inter["seg2"][0]])
        p4 = point_to_str(trajectory2[inter["seg2"][1]])
        edges.append([p1, inter_id])
        edges.append([inter_id, p2])
        edges.append([p3, inter_id])
        edges.append([inter_id, p4])

    return {"nodes": nodes, "edges": edges}

def save_graph(graph, filename="graph.json"):
    """将图保存为JSON文件"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)
    print(f"图已保存到 {filename}")

def visualize_graph(graph, trajectory1, trajectory2, intersections):
    """可视化轨迹和图结构"""
    plt.figure(figsize=(10, 6))

    # 绘制轨迹1（带箭头）
    x1 = [point["x"] for point in trajectory1]
    y1 = [point["y"] for point in trajectory1]
    plt.plot(x1, y1, 'b-', label='Trajectory 1', marker='o', markersize=4)
    for i in range(len(x1) - 1):
        plt.arrow(x1[i], y1[i], x1[i+1] - x1[i], y1[i+1] - y1[i],
                 head_width=0.1, head_length=0.2, fc='b', ec='b', alpha=0.5)

    # 绘制轨迹2（带箭头）
    x2 = [point["x"] for point in trajectory2]
    y2 = [point["y"] for point in trajectory2]
    plt.plot(x2, y2, 'r-', label='Trajectory 2', marker='o', markersize=4)
    for i in range(len(x2) - 1):
        plt.arrow(x2[i], y2[i], x2[i+1] - x2[i], y2[i+1] - y2[i],
                 head_width=0.1, head_length=0.2, fc='r', ec='r', alpha=0.5)

    # 绘制交叉点
    if intersections:
        x_inter = [inter["point"]["x"] for inter in intersections]
        y_inter = [inter["point"]["y"] for inter in intersections]
        plt.scatter(x_inter, y_inter, c='g', s=100, label='Intersections', zorder=5)

    # 绘制图的边
    for edge in graph["edges"]:
        start_node = graph["nodes"][edge[0]]
        end_node = graph["nodes"][edge[1]]
        plt.plot([start_node["x"], end_node["x"]],
                [start_node["y"], end_node["y"]],
                'k--', alpha=0.5, linewidth=1)

    plt.axis('equal')
    plt.title('Trajectories and Graph Structure')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    traj1_path = "traj1_bk.json"
    traj2_path = "traj2.json"

    trajectory1 = read_trajectory_from_json(traj1_path)
    trajectory2 = read_trajectory_from_json(traj2_path)

    if trajectory1 and trajectory2:
        print(f"成功读取 {traj1_path}，包含 {len(trajectory1)} 个点")
        print(f"成功读取 {traj2_path}，包含 {len(trajectory2)} 个点")

        intersections = find_intersections(trajectory1, trajectory2)
        print(f"找到 {len(intersections)} 个交叉点")
        for idx, inter in enumerate(intersections):
            print(f"交叉点 {idx + 1}: x={inter['point']['x']:.2f}, y={inter['point']['y']:.2f}")

        graph = build_graph(trajectory1, trajectory2, intersections)
        save_graph(graph)
        print(f"图包含 {len(graph['nodes'])} 个节点和 {len(graph['edges'])} 条边")

        visualize_graph(graph, trajectory1, trajectory2, intersections)
    else:
        print("读取轨迹失败，请检查文件路径或格式")

if __name__ == "__main__":
    main()