import json
import matplotlib.pyplot as plt
import heapq
import math

def load_graph(filename="graph.json"):
    """从JSON文件加载图"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            graph = json.load(f)
            return graph
    except FileNotFoundError:
        print(f"错误：文件 {filename} 未找到")
        return None
    except json.JSONDecodeError:
        print(f"错误：文件 {filename} 不是有效的JSON格式")
        return None

def dijkstra(graph, start_id, end_id):
    """使用Dijkstra算法计算最短路径"""
    def distance(p1, p2):
        return math.sqrt((p1["x"] - p2["x"])**2 + (p1["y"] - p2["y"])**2)

    adj_list = {}
    for edge in graph["edges"]:
        u, v = edge
        if u not in adj_list:
            adj_list[u] = []
        if v not in adj_list:
            adj_list[v] = []
        dist = distance(graph["nodes"][u], graph["nodes"][v])
        adj_list[u].append((v, dist))
        adj_list[v].append((u, dist))

    distances = {node: float('inf') for node in graph["nodes"]}
    distances[start_id] = 0
    previous = {node: None for node in graph["nodes"]}
    pq = [(0, start_id)]
    visited = set()

    while pq:
        curr_dist, curr_node = heapq.heappop(pq)
        if curr_node in visited:
            continue
        visited.add(curr_node)

        if curr_node == end_id:
            break

        if curr_node in adj_list:
            for neighbor, weight in adj_list[curr_node]:
                if neighbor in visited:
                    continue
                new_dist = curr_dist + weight
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    previous[neighbor] = curr_node
                    heapq.heappush(pq, (new_dist, neighbor))

    if distances[end_id] == float('inf'):
        return None, float('inf')

    path = []
    curr_node = end_id
    while curr_node is not None:
        path.append(curr_node)
        curr_node = previous[curr_node]
    path.reverse()

    return path, distances[end_id]

def visualize_graph(graph, path=None, start_id=None, end_id=None):
    """可视化图结构和最短路径，标出起点和终点"""
    plt.figure(figsize=(10, 6))

    # 绘制图的边
    for edge in graph["edges"]:
        start_node = graph["nodes"][edge[0]]
        end_node = graph["nodes"][edge[1]]
        plt.plot([start_node["x"], end_node["x"]],
                [start_node["y"], end_node["y"]],
                'k--', alpha=0.5, linewidth=1, label='Graph Edges' if edge == graph["edges"][0] else "")

    # 绘制最短路径
    if path:
        path_x = [graph["nodes"][node]["x"] for node in path]
        path_y = [graph["nodes"][node]["y"] for node in path]
        plt.plot(path_x, path_y, 'y-', label='Shortest Path', linewidth=5, zorder=10)

    # 标出所有节点
    x_nodes = [node["x"] for node in graph["nodes"].values()]
    y_nodes = [node["y"] for node in graph["nodes"].values()]
    plt.scatter(x_nodes, y_nodes, c='blue', s=10, label='Nodes', zorder=5)

    # 标出起点和终点
    if start_id and start_id in graph["nodes"]:
        start = graph["nodes"][start_id]
        plt.scatter(start["x"], start["y"], c='purple', s=200, marker='*',
                   label='Start', zorder=15)
    if end_id and end_id in graph["nodes"]:
        end = graph["nodes"][end_id]
        plt.scatter(end["x"], end["y"], c='orange', s=200, marker='*',
                   label='End', zorder=15)

    plt.axis('equal')
    plt.title('Graph Structure and Shortest Path')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    graph = load_graph("graph.json")
    if not graph:
        return

    # 假设从轨迹1起点到轨迹2起点，需手动指定（可以从原始轨迹推断）
    # 这里示例使用图中的任意两个节点，实际使用时需根据需求指定
    nodes = list(graph["nodes"].keys())
    if len(nodes) < 2:
        print("图中节点不足，无法计算路径")
        return

    start_id = nodes[0]  # 示例：轨迹1起点
    end_id = nodes[len(nodes)//2]  # 示例：轨迹2起点
    print(f"计算从 {start_id} 到 {end_id} 的最短路径")

    path, distance = dijkstra(graph, start_id, end_id)

    if path:
        print(f"\n最短路径：")
        for node in path:
            coords = graph["nodes"][node]
            print(f"  ({coords['x']:.2f}, {coords['y']:.2f})")
        print(f"总距离：{distance:.2f}")
    else:
        print("\n没有找到路径")

    visualize_graph(graph, path, start_id, end_id)

if __name__ == "__main__":
    main()