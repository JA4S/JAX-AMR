import jax
import jax.numpy as jnp
from jax import vmap, jit
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Rectangle
from matplotlib.path import Path


def initialize_levelset_boundary(vertices):
    '''
    初始化多边形边界
    '''
    if vertices.shape[0] < 3: 
        raise ValueError("The polygon boundary requires a list consisting of at least 3 vertices.")

    next_vertices = jnp.roll(vertices, -1, axis=0)  # 生成下一个顶点的索引（首尾连接）

    edges = jnp.stack([vertices, jnp.roll(vertices, -1, axis=0)], axis=1) # 每个边由当前顶点和下一个顶点组成，形状为 (n, 2, 2)

    # 计算面积和（用于判断缠绕顺序），鞋带公式sum(p1.x * p2.y - p2.x * p1.y)
    p1_x, p1_y = vertices[:, 0], vertices[:, 1]  # 所有顶点的x,y坐标
    p2_x, p2_y = next_vertices[:, 0], next_vertices[:, 1]  # 下一个顶点的x,y坐标
    area_sum = jnp.sum(p1_x * p2_y - p2_x * p1_y)  # 向量化求和

    is_ccw = jnp.sign(area_sum) # 如果面积和为正，则为逆时针，面积为负，则为顺时针

    return edges, is_ccw


def initialize_cell_info(nx, ny, dx, dy, cell_center):

    full_area = dx * dy
    full_edge_length = jnp.array([dx, dy, dx, dy])[:, None, None]

    # 单元几何信息，预设为流体值
    cell_info = {
        'cut_length': jnp.zeros((nx, ny)),  # 切割线段的长度
        'cut_points': jnp.empty((nx, ny, 4, 2)),  # 切割点列表，通常为2个
        'fluid_centroid': cell_center.transpose(2, 0, 1),  # 流体部分的质心，形状为(2, nx, ny)
        'cut_face_normal': jnp.zeros((2, nx, ny)),  # 切割面的法向量 (从固体指向流体)
        'fluid_area': jnp.ones((nx, ny)) * full_area,  # 流体部分的面积（体积）
        'fluid_edge_length': jnp.ones((4, nx, ny)) * full_edge_length, # 单元格各边流体面长度，顺序：下，右，上，左
        'fluid_area_fraction': jnp.ones((nx, ny)), # 流体部分面积分数
        'fluid_edge_length_fraction': jnp.ones((4, nx, ny)), # 单元格各边流体面长度分数，顺序：下，右，上，左
        'fluid_edge_center_offset': jnp.zeros((4, nx, ny)), # 切割边中心偏移百分比
        'cell_type': -jnp.ones((nx, ny)), # 单元格类型：0：切割单元，1：纯固体网格，-1：纯流体网格
        'neighbor_type': -jnp.ones((4, nx, ny)) # 0：切割边，1：纯固体边，-1：纯流体边，这里用边判断邻居类型，用于处理小体积网格通量重分配计算，仅适用于当前简单几何情况
    }

    return cell_info


def get_cell_corners_value(val):
    """
    返回单元格的四个角点的信息，按逆时针顺序排列，最后一个维度为广播维度
    """
    v0 = val # 左下
    v1 = jnp.roll(v0, -1, axis=0) # 右下
    v2 = jnp.roll(v1, -1, axis=1) # 右上
    v3 = jnp.roll(v0, -1, axis=1) # 左上

    return jnp.stack([v0, v1, v2, v3], axis=0)[:, :-1, :-1] # 需要截断最后一行和一列


def is_point_inside_polygon(polygon, points):
    """
    使用射线法生成固体区域掩模，这里使用Path.contains_points实现向量化
    """
    is_inside = polygon.contains_points(points) # 使用Path.contains_points判断点是否在路径内

    return is_inside

@jit
def point_to_segment_distance(point, edge):
    """
    向量化计算点到线段的距离
    """
    p1, p2 = edge[0], edge[1]

    seg_vec = p2 - p1  # 线段向量
    point_vec = point - p1  # 点到p1的向量

    l2 = jnp.sum(seg_vec **2) # 计算线段长度的平方（用于判断退化线段和投影参数）

    is_degenerate = (l2 == 0.0) # 处理退化线段（p1 == p2）
    dist_degenerate = jnp.linalg.norm(point_vec, axis=-1) # 距离为点到p1的距离，使用掩码避免分支判断

    # 非退化线段的计算
    t = jnp.dot(point_vec, seg_vec) / l2  # 投影参数
    t_clamped = jnp.clip(t, 0.0, 1.0)  # 用clip替代t的范围判断
    projection = p1 + t_clamped[..., None] * seg_vec  # 投影点
    dist_normal = jnp.linalg.norm(point - projection, axis=-1)

    return jnp.where(is_degenerate, dist_degenerate, dist_normal) # 退化线段用dist_degenerate，否则用dist_normal


def get_outward_normal(is_ccw, segment):
    """
    计算线段的向外法向量
    """
    dx = segment[1, 0] - segment[0, 0]
    dy = segment[1, 1] - segment[0, 1]

    normal = jnp.array([dy, -dx]) * is_ccw

    normal = normal / jnp.linalg.norm(normal)

    return jnp.nan_to_num(normal) # jnp.linalg.norm(normal)=0，则返回0


@jit
def point_to_polygon_distance(p, edges):
    """
    计算点 p 到多边形的最短距离（无符号）, edges形状为(N, 2, 2)
    """
    distances_to_edges = vmap(point_to_segment_distance, in_axes=(None, 0))(p, edges)

    distances = jnp.min(distances_to_edges)
    edge_index = jnp.argmin(distances_to_edges)

    return distances, edge_index


def gradient_phi(edges, is_ccw, points):
    """
    计算phi梯度
    """
    distances, edge_index = point_to_polygon_distance(points, edges)
    normal = get_outward_normal(is_ccw, edges[edge_index])

    return normal


def phi(polygon, edges, is_ccw, grid_points):
    """
    计算phi
    """
    distances, edge_index = vmap(point_to_polygon_distance, in_axes=(0, None))(grid_points, edges) # 点到多边形的无符号距离

    is_inside = is_point_inside_polygon(polygon, grid_points) # 点的符号

    #normal = vmap(get_outward_normal, in_axes=(None, 0))(is_ccw, edges[edge_index]) # 向量化求phi梯度

    phi_val = jnp.where(is_inside, distances, -distances) # phi>0为固体，phi<0为流体

    return phi_val


def get_cell_indices(polygon, edges, is_ccw, phi_corners):
    '''
    获取各单元格索引。phi < 0 是流体，phi > 0 是固体, phi = 0是切割单元
    '''
    # 检查单元格是完全流体还是完全固体，一边与边界重合，也认为是切割单元
    phi_sign = jnp.sign(phi_corners + EPS * jnp.ones(phi_corners.shape)) # phi > -EPS
    phi_val = phi_corners * jnp.clip(phi_sign, 0, 1) # 临近固体和固体内的点的phi，包含了边界附近的流体域角点
    num_positive_phi = jnp.sum(jnp.clip(phi_sign, 0, 1), axis=0)
    num_negative_phi = jnp.sum(jnp.clip(-phi_sign, 0, 1), axis=0)

    cond1 = (num_positive_phi == 0) # 如果所有角点都是流体 (phi < 0)
    cond2 = jnp.logical_and(num_positive_phi == 1, jnp.sum(phi_val, axis=0) < EPS) # 有且只有一个角点在边界上
    cond3 = (num_negative_phi == 0) # 如果所有角点都是固体 (phi > 0)

    is_fluid = jnp.logical_or(cond1, cond2)
    is_solid = cond3
    is_cut_cell = ~is_fluid & ~is_solid

    #fluid_indices = jnp.where(is_fluid)
    solid_indices = jnp.where(is_solid)
    cut_cell_indices = jnp.where(is_cut_cell)

    return solid_indices, cut_cell_indices # 其他索引位置为流体



def get_all_cell_info(cell_info, solid_indices, cut_cell_indices, polygon, edges, is_ccw, cell_corners, phi_corners):

    # 处理固体单元
    i, j = solid_indices
    cell_info['fluid_centroid'] = cell_info['fluid_centroid'].at[:, i, j].set(0)
    cell_info['fluid_area'] = cell_info['fluid_area'].at[i, j].set(0)
    cell_info['fluid_edge_length'] = cell_info['fluid_edge_length'].at[:, i, j].set(0)
    cell_info['fluid_area_fraction'] = cell_info['fluid_area_fraction'].at[i, j].set(0)
    cell_info['fluid_edge_length_fraction'] = cell_info['fluid_edge_length_fraction'].at[:, i, j].set(0)
    cell_info['cell_type'] = cell_info['cell_type'].at[i, j].set(1)
    cell_info['neighbor_type'] = cell_info['neighbor_type'].at[:, i, j].set(1)


    # 处理切割单元
    cell_info['cell_type'] = cell_info['cell_type'].at[cut_cell_indices].set(0)
    num_cut_cell = len(cut_cell_indices[0])
    for k in range(num_cut_cell):
        i,  j = cut_cell_indices[0][k], cut_cell_indices[1][k]
        cell_info = process_cut_cell(i, j, cell_info, edges, is_ccw, cell_corners[:, i, j], phi_corners[:, i, j])

    return cell_info


def interpolate_intersection(p1, p2, phi1, phi2):
    """
    通过线性插值计算 phi=0 的交点
    """
    val = -phi1 / (phi2 - phi1) * (p2 - p1) + p1

    return jnp.nan_to_num(val, nan=None)


def polygon_area_centroid(points):
    """
    向量化计算多边形的面积和质心（使用鞋带公式）
    """
    points = jnp.array(points) # 这是因为输入的参数是list，需要转换，后续继续优化后可删除

    next_points = jnp.roll(points, -1, axis=0)  # 形状 (n, 2)

    x1, y1 = points[:, 0], points[:, 1]    # 当前顶点坐标
    x2, y2 = next_points[:, 0], next_points[:, 1]  # 下一个顶点坐标

    cross_product = x1 * y2 - x2 * y1  # 向量化计算交叉积（替代循环中的逐个计算）

    # 累加计算面积和质心的分子部分
    area = jnp.sum(cross_product) * 0.5 # 计算面积（鞋带公式）
    cx = jnp.sum((x1 + x2) * cross_product) / (6.0 * area)
    cy = jnp.sum((y1 + y2) * cross_product) / (6.0 * area)

    # 非有效多边形（顶点数<3）和退化多边形（面积接近零），返回0
    is_vaild = (points.shape[0] > 2) & ((jnp.abs(area) - EPS) >= 0)

    if is_vaild:
        return area, jnp.array([cx, cy])
    else:
        return 0.0, jnp.array([0.0, 0.0])



def process_cut_cell(i, j, cell_info, edges, is_ccw, corners, phi_values): # 还应增加dx, dy
    """
    处理切割单元信息
    """
    # 初始化默认值
    cut_length = 0  # 切割线段的长度
    cut_points = []  # 切割点列表 (通常为2个)
    fluid_centroid = jnp.zeros(2)  # 流体部分的质心
    cut_face_normal = jnp.zeros(2)  # 切割面的法向量 (从固体指向流体)
    fluid_area = 0  # 流体部分的面积（体积）
    fluid_edge_length = jnp.array([dx, dy, dx, dy]) # 单元格各边流体面长度，顺序：下，右，上，左
    fluid_area_fraction = 0 # 流体部分面积分数
    fluid_edge_length_fraction = jnp.ones(4) # 单元格各边流体面长度分数，顺序：下，右，上，左
    fluid_edge_center_offset = jnp.zeros(4) # 切割边中心偏移百分比
    neighbor_type = -jnp.ones(4) # 单元格各边类型：0：切割边，1：纯固体边，-1：纯流体边

    is_abnormal = False # 标记cut-cell是否异常，当前程序仅处理有2个切点的情况，其他情况判断为异常，cell_info不做更改

    # 查找单元格边上的交点
    cell_edges = [(corners[0], corners[1]),  # P0-P1 (底部)
             (corners[1], corners[2]),  # P1-P2 (右侧)
             (corners[2], corners[3]),  # P2-P3 (顶部)
             (corners[3], corners[0])]  # P3-P0 (左侧)

    phi_edges = [(phi_values[0], phi_values[1]),
                 (phi_values[1], phi_values[2]),
                 (phi_values[2], phi_values[3]),
                 (phi_values[3], phi_values[0])]


    intersection_points = []
    for k in range(4): # 顺序：下，右，上，左
        p_start, p_end = cell_edges[k]
        phi_start, phi_end = phi_edges[k]

        # 检查是否存在交点：phi 值的符号不同
        # 使用一个小的 epsilon 来处理非常接近零的点
        if (phi_start * phi_end < -EPS) or \
                (abs(phi_start) < EPS and phi_end > EPS) or \
                (abs(phi_end) < EPS and phi_start > EPS):  # 一个为零，另一个为正

            # 如果一个 phi 为零，且另一个为固体，则该点是交点。
            if abs(phi_start) < EPS and phi_end > EPS: # 固体边，其他信息采用默认值
                intersection_points.append(p_start)
                fluid_edge_length = fluid_edge_length.at[k].set(0.0)
                fluid_edge_length_fraction = fluid_edge_length_fraction.at[k].set(0.0)
                neighbor_type = neighbor_type.at[k].set(1)
            elif abs(phi_end) < EPS and phi_start > EPS: # 固体边，其他信息采用默认值
                intersection_points.append(p_end)
                fluid_edge_length = fluid_edge_length.at[k].set(0.0)
                fluid_edge_length_fraction = fluid_edge_length_fraction.at[k].set(0.0)
                neighbor_type = neighbor_type.at[k].set(1)
            elif phi_start * phi_end < -EPS:  # 标准情况：穿过零
                intersection_points_cords = interpolate_intersection(p_start, p_end, phi_start, phi_end)
                intersection_points.append(intersection_points_cords)
                if phi_start < -EPS:
                    length = jnp.linalg.norm(p_start - intersection_points_cords)
                    fraction = length/jnp.linalg.norm(p_start - p_end)
                    offset = (1 - fraction) if (k == 0 or k == 1) else -(1 - fraction)

                    fluid_edge_length = fluid_edge_length.at[k].set(length)
                    fluid_edge_length_fraction = fluid_edge_length_fraction.at[k].set(fraction)
                    fluid_edge_center_offset = fluid_edge_center_offset.at[k].set(offset)
                    neighbor_type = neighbor_type.at[k].set(0)
                else:
                    length = jnp.linalg.norm(intersection_points_cords -  p_end)
                    fraction = length/jnp.linalg.norm(p_start - p_end)
                    offset = (1 - fraction) if (k == 0 or k == 1) else -(1 - fraction)

                    fluid_edge_length = fluid_edge_length.at[k].set(length)
                    fluid_edge_length_fraction = fluid_edge_length_fraction.at[k].set(fraction)
                    fluid_edge_center_offset = fluid_edge_center_offset.at[k].set(offset)
                    neighbor_type = neighbor_type.at[k].set(0)

        # 如果两个角点均与边界重合，则该边为固体边界，其他信息采用默认值
        if (abs(phi_start) < EPS and abs(phi_end) < EPS):
            intersection_points.append(p_start)
            intersection_points.append(p_end)
            fluid_edge_length = fluid_edge_length.at[k].set(0.0)
            fluid_edge_length_fraction = fluid_edge_length_fraction.at[k].set(0.0)
            neighbor_type = neighbor_type.at[k].set(1)

    def remove_all_duplicates(arr_list):
        # 将 JAX 数组转换为可哈希的 Python 数值列表，再转为元组
        tuple_list = []
        for pt in arr_list:
            if pt is not None:
                # 将 JAX 数组转为 Python 列表（元素为浮点数）
                pt_list = jnp.asarray(pt).tolist()
                # 再转为元组（可哈希）
                tuple_list.append(tuple(pt_list))

        # 统计每个元组的出现次数
        count = {}
        for t in tuple_list:
            count[t] = count.get(t, 0) + 1

        # 筛选只出现一次的元素，并转回 JAX 数组
        unique_elements = [jnp.array(t) for t in tuple_list if count[t] == 1]
        return unique_elements

    def remove_all_duplicates__(arr_list):
        tuple_list = [tuple(pt) for pt in arr_list] # 先将数组转为元组以便计数（因为数组不可哈希）

        count = {} # 统计每个元素出现的次数
        for t in tuple_list:
            count[t] = count.get(t, 0) + 1

        unique_elements = [jnp.array(t) for t in tuple_list if count[t] == 1] # 筛选出只出现一次的元素，并转回数组

        return unique_elements


    # 去除重复值，这是当前程序不完善的地方，一些情况下会产生重复值
    intersection_points = remove_all_duplicates(intersection_points)
    # 过滤掉 None 值
    cut_points = [p for p in intersection_points if p is not None]

    # --- 计算流体质心和法向量，对所有切割单元都尝试计算 ---
    if len(cut_points) == 2:
        # 对于两个切割点的情况，进行精确计算
        p_int1, p_int2 = cut_points[0], cut_points[1]

        cut_length = jnp.linalg.norm(p_int1 - p_int2)

        # 计算切割面法向量 (在切割线段的中点)
        cut_midpoint = (p_int1 + p_int2) / 2

        # 求解【实际切割面】法向量，采用levelset方法后，实际切割面与几何边界不完成重合，特别是在角点附近
        normal_1 = gradient_phi(edges, is_ccw, cut_midpoint)
        normal_2 = get_outward_normal(is_ccw, jnp.array(cut_points))
        cut_face_normal_sign = 1 if jnp.dot(normal_1, normal_2) > 0 else -1
        cut_face_normal = cut_face_normal_sign * normal_2

        # 构建流体多边形
        temp_fluid_points = []
        temp_fluid_points.append(p_int1)
        temp_fluid_points.append(p_int2)
        for k in range(4):
            if phi_values[k] < -EPS:  # 角点在流体区域或在边界上
                temp_fluid_points.append(corners[k])

        # 移除重复点
        unique_fluid_points = []
        for p in temp_fluid_points:
            is_duplicate = False
            for up in unique_fluid_points:
                if jnp.allclose(p, up):
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_fluid_points.append(p)

        if len(unique_fluid_points) >= 3:  # 能够形成有效多边形
            temp_centroid = jnp.mean(jnp.array(unique_fluid_points), axis=0)

            def angle_from_centroid(point, centroid):
                return math.atan2(point[1] - centroid[1], point[0] - centroid[0])

            fluid_polygon_points = sorted(unique_fluid_points, key=lambda p: angle_from_centroid(p, temp_centroid))

            fluid_area, fluid_centroid = polygon_area_centroid(fluid_polygon_points)

            # 确保面积为正 (如果排序结果是顺时针，则面积为负)
            if fluid_area < 0:
                fluid_polygon_points.reverse()
                fluid_area, fluid_centroid = polygon_area_centroid(fluid_polygon_points)
        else:  # 退化情况，例如只有两个点，无法形成多边形
            is_abnormal = True
            jax.debug.print(f'Warning: Degenerate case. Cannot form a polygon at cell ({i}, {j})')

    else:  # 复杂切割：切割点数量不为2 (0, 1, 3或更多)
        is_abnormal = True
        jax.debug.print(f'Warning: Complex cut cell. Cut point is not equal to 2 at cell ({i}, {j})', \
                 'Only simple cut cells is supported at this version.'
        )


    if len(cut_points) < 4: #不足4个补齐
        for k in range(4 - len(cut_points)):
            cut_points.append([jnp.nan, jnp.nan])
    else: # 若超过4个点，取前4个
        cut_points = cut_points[:4]

    fluid_area_fraction = fluid_area / dx / dy

    if ~is_abnormal:
        cell_info['cut_length'] = cell_info['cut_length'].at[i,j].set(cut_length)
        cell_info['cut_points'] = cell_info['cut_points'].at[i,j].set(cut_points)
        cell_info['fluid_centroid'] = cell_info['fluid_centroid'].at[:,i,j].set(fluid_centroid)
        cell_info['cut_face_normal'] = cell_info['cut_face_normal'].at[:,i,j].set(cut_face_normal)
        cell_info['fluid_area'] = cell_info['fluid_area'].at[i,j].set(fluid_area)
        cell_info['fluid_edge_length'] = cell_info['fluid_edge_length'].at[:,i,j].set(fluid_edge_length)
        cell_info['fluid_area_fraction'] = cell_info['fluid_area_fraction'].at[i,j].set(fluid_area_fraction)
        cell_info['fluid_edge_length_fraction'] = cell_info['fluid_edge_length_fraction'].at[:,i,j].set(fluid_edge_length_fraction)
        cell_info['fluid_edge_center_offset'] = cell_info['fluid_edge_center_offset'].at[:,i,j].set(fluid_edge_center_offset)
        cell_info['neighbor_type'] = cell_info['neighbor_type'].at[:,i,j].set(neighbor_type)

    return cell_info


def visualize_cut_cell(vertices, x_min, y_min, nx, ny, dx, dy, cut_cell_indices, cell_info, cell_corners, phi_corners):
    '''
    切割单元和信息可视化
    '''

    num_cut_cell = len(cut_cell_indices[0])

    # 打印部分切割单元的信息
    print("\n--- Part of cut cell information ---")
    for k in range(num_cut_cell):
        if k < 5 or k >= num_cut_cell - 5:
            i,  j = cut_cell_indices[0][k], cut_cell_indices[1][k]
            print(f"\n--- Cell ({i}, {j}) ---")
            print(f"cut points: {[f'({p[0]:.4f}, {p[1]:.4f})' for p in cell_info['cut_points'][i, j, 0:2]]}")
            print(f"cut length: {cell_info['cut_length'][i, j]}")
            print(f"fluid centroid: ({cell_info['fluid_centroid'][0, i, j]:.4f}, {cell_info['fluid_centroid'][1, i, j]:.4f})")
            print(f"cut face normal: ({cell_info['cut_face_normal'][0, i, j]:.4f}, {cell_info['cut_face_normal'][1, i, j]:.4f})")
            print(f"fluid area: {cell_info['fluid_area'][i, j]:.4f}")
            print(f"切割边长度分数: {cell_info['fluid_edge_length_fraction'][:, i, j]}")
            print(f"neighbor type: {cell_info['neighbor_type'][:, i, j]}")


    print("\n Visualization results are being produced...")

    fig, ax = plt.subplots(figsize=(64, 64))
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(x_min, x_min + nx*dx)
    ax.set_ylim(y_min, y_min + ny*dy)
    ax.set_title("cut cell")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # 绘制网格线
    for i in range(nx + 1):
        ax.axvline(x_min + i * dx, color='lightgray', linestyle='--', linewidth=0.5)
    for j in range(ny + 1):
        ax.axhline(y_min + j * dy, color='lightgray', linestyle='--', linewidth=0.5)

    # 绘制边界 (多边形)
    polygon_patch = MplPolygon(vertices, closed=True,
                                color='blue', fill=False, linestyle='-', linewidth=2, label='boundary')
    ax.add_patch(polygon_patch)

    # 绘制切割单元及其流体部分、质心和法向量
    for k in range(num_cut_cell):
        i,  j = cut_cell_indices[0][k], cut_cell_indices[1][k]

        # 绘制切割单元的边框 (无论其切割复杂性如何)
        rect = Rectangle((cell_corners[0, i, j, 0], cell_corners[0, i, j, 1]), dx, dy,
                            linewidth=1, edgecolor='red', facecolor='none', linestyle='-')
        ax.add_patch(rect)

        # 只有当切割点数量为2时，才绘制详细的流体多边形和切割点
        valid_cut_points = jnp.sum(~jnp.isnan(jnp.array(cell_info['cut_points'][i,j])))

        if valid_cut_points/2 == 2:
            
            corners = cell_corners[:, i, j]
            phi_values = phi_corners[:, i, j]

            temp_fluid_points = []
            temp_fluid_points.append(cell_info['cut_points'][i,j][0])
            temp_fluid_points.append(cell_info['cut_points'][i,j][1])
            for k in range(4):
                if phi_values[k] < -EPS:  # 角点在流体区域或在边界上
                    temp_fluid_points.append(corners[k])

            unique_fluid_points = []
            for p in temp_fluid_points:
                is_duplicate = False
                for up in unique_fluid_points:
                    if jnp.allclose(p, up):
                        is_duplicate = True
                        break
                if not is_duplicate:
                    unique_fluid_points.append(p)

            if len(unique_fluid_points) >= 3:  # 能够形成有效多边形
                temp_centroid_vis = jnp.mean(jnp.array(unique_fluid_points), axis=0)

                def angle_from_centroid_vis(point, centroid):
                    return math.atan2(point[1] - centroid[1], point[0] - centroid[0])


                fluid_polygon_points_vis = sorted(unique_fluid_points, key=lambda p: angle_from_centroid_vis(p, temp_centroid_vis))

                area_vis, _ = polygon_area_centroid(fluid_polygon_points_vis)
                if area_vis < 0:
                    fluid_polygon_points_vis.reverse()

                poly = MplPolygon(fluid_polygon_points_vis, closed=True, edgecolor='red',
                                    facecolor='lightgreen', alpha=0.7, linewidth=1)
                ax.add_patch(poly)

                # 绘制切割点
                ax.plot([p[0] for p in cell_info['cut_points'][i,j]], [p[1] for p in cell_info['cut_points'][i,j]], 'ko', markersize=3)

        # 绘制流体质心 (对于所有切割单元，如果其值不是默认的零向量)
        if jnp.any(cell_info['fluid_centroid'][:,i,j] != 0.0):
            ax.plot(cell_info['fluid_centroid'][0,i,j], cell_info['fluid_centroid'][1,i,j], 'bx', markersize=5)

        # 绘制切割面法向量 (对于所有切割单元，如果其值不是默认的零向量)
        if jnp.any(cell_info['cut_face_normal'][:,i,j] != 0.0):
            # 使用计算出的流体质心作为法向量的起点
            origin_for_normal = cell_info['fluid_centroid'][:,i,j]

            normal_vec = cell_info['cut_face_normal'][:,i,j] * dx * 0.5  # 缩放法向量以便可见
            ax.arrow(origin_for_normal[0], origin_for_normal[1], normal_vec[0], normal_vec[1],
                        head_width=0.01, head_length=0.01, fc='purple', ec='purple', linewidth=1)


    # 添加图例，避免重复标签
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    plt.show()


def initialize_eb(vertices, x_min, y_min, nx, ny, dx, dy, visual=False):

    # First, generate the computational grid
    xc = jnp.linspace(x_min + dx/2, x_min + nx*dx - dx/2, nx)
    yc = jnp.linspace(y_min + dy/2, y_min + ny*dy - dy/2, ny)
    Xc, Yc = jnp.meshgrid(xc, yc, indexing='ij')  # (nx, ny)网格中心点
    cell_center = jnp.stack([Xc.ravel(), Yc.ravel()], axis=1).reshape((nx, ny, 2))

    x = jnp.linspace(x_min, x_min + nx*dx, nx + 1)
    y = jnp.linspace(y_min, y_min + ny*dy, ny + 1)
    X, Y = jnp.meshgrid(x, y, indexing='ij')  # (nx+1, ny+1)网格角点
    grid_points = jnp.stack([X.ravel(), Y.ravel()], axis=1)
    cell_corners = get_cell_corners_value(grid_points.reshape(nx+1, ny+1, 2))


    # Then, intialize the embedded boundary and get cell infomation
    polygon = Path(vertices)

    edges, is_ccw = initialize_levelset_boundary(vertices)

    cell_info = initialize_cell_info(nx, ny, dx, dy, cell_center)

    phi_points = phi(polygon, edges, is_ccw, grid_points)

    phi_corners = get_cell_corners_value(phi_points.reshape((nx+1, ny+1))) 

    solid_indices, cut_cell_indices = get_cell_indices(polygon, edges, is_ccw, phi_corners)

    cell_info = get_all_cell_info(cell_info, solid_indices, cut_cell_indices, polygon, edges, is_ccw, cell_corners, phi_corners)

    print(f"\n EB initialzed. {len(cut_cell_indices[0])} cut cells are found.")

    if visual:
        visualize_cut_cell(vertices, x_min, y_min, nx, ny, dx, dy, cut_cell_indices, cell_info, cell_corners, phi_corners)

    return cell_info



