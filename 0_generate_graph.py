import json
import os
import glob
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
    """判断两线段是否相交，并返回交点"""
    def ccw(A, B, C):
        val = (C["y"] - A["y"]) * (B["x"] - A["x"]) - (B["y"] - A["y"]) * (C["x"] - A["x"])
        return val > 1e-10

    def get_intersection_point(p1, p2, p3, p4):
        x1, y1 = p1["x"], p1["y"]
        x2, y2 = p2["x"], p2["y"]
        x3, y3 = p3["x"], p3["y"]
        x4, y4 = p4["x"], p4["y"]

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return None

        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom

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

def find_intersections(trajectory1, trajectory2, traj1_idx, traj2_idx):
    """找到两条轨迹的所有交叉点及其对应线段，记录轨迹索引"""
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
                    "seg2": (j, j + 1),
                    "traj1_idx": traj1_idx,
                    "traj2_idx": traj2_idx
                })
    return intersections

def get_t(start, end, point):
    """计算点在段上的参数t值"""
    dx = end["x"] - start["x"]
    dy = end["y"] - start["y"]
    denom = dx * dx + dy * dy
    if denom < 1e-10:
        return 0
    t = ((point["x"] - start["x"]) * dx + (point["y"] - start["y"]) * dy) / denom
    return max(0, min(1, t))

def build_graph(trajectories, all_intersections):
    """构建图结构并返回节点和边，保留每个点的轨迹索引信息"""
    def point_to_str(p):
        return f"{p['x']:.6f},{p['y']:.6f}"

    nodes = {}  # 格式: {node_id: {"x": x, "y": y, "traj_indices": [idx1, idx2, ...]}}
    edges = set()

    # 添加所有轨迹点作为节点
    for traj_idx, traj in enumerate(trajectories):
        for point in traj:
            node_id = point_to_str(point)
            if node_id not in nodes:
                nodes[node_id] = {"x": point["x"], "y": point["y"], "traj_indices": []}
            if traj_idx not in nodes[node_id]["traj_indices"]:
                nodes[node_id]["traj_indices"].append(traj_idx)

    # 添加所有交叉点作为节点
    for inter in all_intersections:
        inter_id = point_to_str(inter["point"])
        if inter_id not in nodes:
            nodes[inter_id] = {"x": inter["point"]["x"], "y": inter["point"]["y"], "traj_indices": []}
        # 添加交叉点涉及的两条轨迹索引
        for idx in [inter["traj1_idx"], inter["traj2_idx"]]:
            if idx not in nodes[inter_id]["traj_indices"]:
                nodes[inter_id]["traj_indices"].append(idx)

    # 为每条轨迹的每个段构建交叉点映射
    segment_intersections = {}
    for inter in all_intersections:
        traj1_idx = inter["traj1_idx"]
        traj2_idx = inter["traj2_idx"]
        seg1 = (traj1_idx, inter["seg1"][0], inter["seg1"][1])
        seg2 = (traj2_idx, inter["seg2"][0], inter["seg2"][1])
        if seg1 not in segment_intersections:
            segment_intersections[seg1] = []
        if seg2 not in segment_intersections:
            segment_intersections[seg2] = []
        segment_intersections[seg1].append(inter["point"])
        segment_intersections[seg2].append(inter["point"])

    # 为每条轨迹添加边
    for traj_idx, traj in enumerate(trajectories):
        for i in range(len(traj) - 1):
            start = traj[i]
            end = traj[i + 1]
            start_id = point_to_str(start)
            end_id = point_to_str(end)
            seg_key = (traj_idx, i, i + 1)

            if seg_key not in segment_intersections or not segment_intersections[seg_key]:
                if start_id != end_id:
                    edges.add((start_id, end_id))
            else:
                inter_points = segment_intersections[seg_key]
                points_with_t = [(p, get_t(start, end, p)) for p in inter_points]
                points_with_t.sort(key=lambda x: x[1])

                prev_id = start_id
                for point, t in points_with_t:
                    curr_id = point_to_str(point)
                    if curr_id != prev_id:
                        edges.add((prev_id, curr_id))
                    prev_id = curr_id

                if prev_id != end_id:
                    edges.add((prev_id, end_id))

    return {"nodes": nodes, "edges": list(edges)}

def save_graph(graph, filename="graph.json"):
    """将图保存为JSON文件"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)
    print(f"图已保存到 {filename}")

def visualize_graph(graph, trajectories, all_intersections):
    """可视化多条轨迹和图结构"""
    plt.figure(figsize=(20, 20))
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']

    for idx, traj in enumerate(trajectories):
        color = colors[idx % len(colors)]
        x = [point["x"] for point in traj]
        y = [point["y"] for point in traj]
        plt.plot(x, y, f'{color}-', label=f'Trajectory {idx + 1}', marker='o', markersize=4)
        for i in range(len(x) - 1):
            plt.arrow(x[i], y[i], x[i+1] - x[i], y[i+1] - y[i],
                      head_width=0.1, head_length=0.2, fc=color, ec=color, alpha=0.5)

    if all_intersections:
        x_inter = [inter["point"]["x"] for inter in all_intersections]
        y_inter = [inter["point"]["y"] for inter in all_intersections]
        plt.scatter(x_inter, y_inter, c='yellow', s=100, label='Intersections', zorder=5,marker='*')

    for edge in graph["edges"]:
        start_node = graph["nodes"][edge[0]]
        end_node = graph["nodes"][edge[1]]
        plt.plot([start_node["x"], end_node["x"]],
                 [start_node["y"], end_node["y"]],
                 'k--', alpha=0.5, linewidth=1)

    plt.axis('equal')
    plt.title('Multiple Trajectories and Graph Structure')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    directory_path = "trajs"
    if not os.path.exists(directory_path):
        print(f"错误：目录 {directory_path} 不存在")
        return

    trajectory_files = glob.glob(os.path.join(directory_path, "*.json"))
    if not trajectory_files:
        print(f"错误：目录 {directory_path} 中未找到JSON文件")
        return

    trajectories = []
    for filepath in trajectory_files:
        traj = read_trajectory_from_json(filepath)
        if traj:
            trajectories.append(traj)
            print(f"成功读取 {filepath}，包含 {len(traj)} 个点")
        else:
            print(f"跳过 {filepath}，读取失败")

    if len(trajectories) < 2:
        print("错误：至少需要两条轨迹来计算交叉点")
        return

    all_intersections = []
    for i in range(len(trajectories)):
        for j in range(i + 1, len(trajectories)):
            intersections = find_intersections(trajectories[i], trajectories[j], i, j)
            all_intersections.extend(intersections)

    print(f"找到 {len(all_intersections)} 个交叉点")
    for idx, inter in enumerate(all_intersections):
        print(f"交叉点 {idx + 1}: x={inter['point']['x']:.2f}, y={inter['point']['y']:.2f}, "
              f"轨迹 {inter['traj1_idx'] + 1} 和 轨迹 {inter['traj2_idx'] + 1}")

    graph = build_graph(trajectories, all_intersections)
    save_graph(graph)
    print(f"图包含 {len(graph['nodes'])} 个节点和 {len(graph['edges'])} 条边")

    visualize_graph(graph, trajectories, all_intersections)

if __name__ == "__main__":
    main()