背景

https://zhouyiqun.notion.site/1a0c553f84118093b513d3c655fd8901?pvs=4

# 场景

- 停车场充电机器人：给出多条交叉的轨迹，提取出连接关系，能够实现跨轨迹的点到点导航。无人车需要从出发点行驶到任意一条轨迹上的任意一点。
- 扫地车返航：从多条交错的轨迹中，提取出返回起点的最快路径
- 扫地车跨轨迹清扫：生成跨轨迹的清扫任务

# 层次

![image.png](attachment:5ae00133-8d94-4314-83d1-142b09876a53:image.png)

# 功能逻辑

## 1. 获得多条轨迹：

- 车辆录制
- 手动绘制

## 2. 将轨迹整合成一张路网【地图工具】

- 读取json格式的轨迹
- 保存成X格式的中间变量
- 每个场景一个地图，导航时可以快速读取
- 路网是有方向的
- 可以手动打开逆向行驶，逆向标记在路网上。
- 基于路网，能够导出json格式的综合轨迹，实现跨轨迹任务

## 3. 在路网上实现点到点导航【车端】

- 加载地图工具给出的路网文件
- 基于路网，选择两个点，能够输出最近的路线
- 输出的路线作为sweep_path，发送给导航模块
- sweep_path中保存原有的属性不变。

## 4. 交互

- 可视化路网
- 可视化当前执行的轨迹

# 关键模块

路网的数据格式：JSON

寻路算法：DIJ等

# DEMO

## 运行效果

## 第一步：导入两条轨迹 生成路网

![image.png](attachment:cac59a6c-6996-442e-be10-34c4394ed4e0:image.png)

- 代码：`0_generate_graph.py`
- 这段代码应该整合到地图工具中
    - 先载入多段轨迹，
    - 点击生成路网，能够产生路网文件
    - 对于实时变换轨迹：将路网文件拷贝到车上用于跨轨迹导航
    - (需要拓展成支持大于两条的轨迹，对数量特别多的点需要性能优化)
- 生成的中间文件：`graph.json`
    - 格式: 点+边

        ```python
        {
          "nodes": {
            "-0.034130,0.075120": {
              "x": -0.03413008153438568,
              "y": 0.07512026280164719
            },
            "0.470985,0.063333": {
              "x": 0.47098487615585327,
              "y": 0.06333254277706146
            },
            "0.997244,0.087132": {
              "x": 0.9972444772720337,
              "y": 0.08713161200284958
            },
            "1.516586,0.095518": {
              "x": 1.5165858268737793,
              "y": 0.09551813453435898
              ```
              ```
            },
          "edges": [
            [
              "-0.034130,0.075120",
              "0.470985,0.063333"
            ],
            [
              "0.470985,0.063333",
              "0.997244,0.087132"
            ],
            [
              "0.997244,0.087132",
              "1.516586,0.095518"
            ],
            [
              "1.516586,0.095518",
              "2.046854,0.061520"
        ```

    - 每个场景都应该有一个这样的路网文件

## 第二步: 基于路网 实现跨轨迹导航

![image.png](attachment:2dcd5bd9-accd-447c-a167-3b79d4426ff9:image.png)

- 代码：`1_navigation.py`
- 这段代码应该整合到车端代码中
    - 先载入路网文件
    - 根据调度需求，指定起点、终点，搜索最短路径
    - 得到跨轨迹导航路径，替换现有的sweep_path
    - 交给schedule_goal调度
- 整合到地图工具中，可以用于生成跨轨迹任务
    - 跨轨迹任务也是一个固定的json
    - 因此，可以逐段生成，每次选取一个区间，确保经由关键拐点