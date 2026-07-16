import numpy as np
from scipy.spatial import KDTree
from collections import Counter

class BoardTrackerVoting:
    def __init__(self, piece_ids, square_ids):
        """
        piece_ids: 棋子固定编号列表，如 ['P0','P1',...]
        square_ids: 棋盘格固定编号列表，如 ['S0','S1',...]
        """
        self.piece_ids = list(piece_ids)
        self.square_ids = list(square_ids)
        self.all_ids = self.piece_ids + self.square_ids   # 所有对象的统一ID顺序
        self.history_coords = {}      # ID -> np.array([x, y])
        self.history_array = None     # 按 all_ids 顺序排列的历史坐标数组，形状 (N,2)
        self.initialized = False

    def initialize(self, piece_coords, square_coords):
        """
        在第一帧稳定、所有对象都被检测到且棋子未移动时调用。
        piece_coords: 棋子坐标列表，顺序与 piece_ids 一一对应
        square_coords: 棋盘格坐标列表，顺序与 square_ids 一一对应
        """
        for pid, coord in zip(self.piece_ids, piece_coords):
            self.history_coords[pid] = np.array(coord, dtype=float)
        for sid, coord in zip(self.square_ids, square_coords):
            self.history_coords[sid] = np.array(coord, dtype=float)

        # 把所有历史坐标按统一顺序堆成一个数组，方便矩阵运算
        self.history_array = np.array([self.history_coords[i] for i in self.all_ids])
        self.initialized = True

    def update(self, det_piece_coords, det_square_coords, distance_thresh=5.0):
        """
        每一帧调用，处理当前检测结果。
        det_piece_coords: 当前检测到的所有棋子坐标列表（顺序任意）
        det_square_coords: 当前检测到的所有棋盘格坐标列表（可以有缺失）
        distance_thresh: 匹配距离阈值（像素）

        返回值:
            id_map          : 字典，当前检测点索引 -> 历史ID
            moved_piece     : (历史ID, 当前坐标) 或 (历史ID, None)
            missing_squares : 被遮挡的棋盘格ID列表
            inferred_coords : 字典，历史ID -> 推算出的当前位置
        """
        # 1. 合并当前检测到的所有点，并记住每个点的来源
        det_points = []     # 存放当前所有检测点的坐标 (x, y)
        det_info = []       # 存放每个点的来源：('piece', index) 或 ('square', index)
        for i, c in enumerate(det_piece_coords):
            det_points.append(c)
            det_info.append(('piece', i))
        for i, c in enumerate(det_square_coords):
            det_points.append(c)
            det_info.append(('square', i))
        det_points = np.array(det_points, dtype=float)   # 形状 (M, 2)

        # 2. 投票找出全局平移向量 T
        #    对每一对 (当前检测点, 历史点) 计算平移向量，统计出现最多的那个
        translations = []
        for det_pt in det_points:
            # 向量化减法：当前点 - 所有历史点，得到一批平移向量
            diffs = det_pt - self.history_array  # 形状 (N, 2)
            translations.extend(diffs)
        translations = np.array(translations)

        # 将平移向量量化到小数点后两位，消除微小噪声，便于计数
        quantized = np.round(translations, 2)
        # 转换成元组，用 Counter 统计出现次数
        trans_tuples = [tuple(t) for t in quantized]
        counter = Counter(trans_tuples)
        best_trans_tuple, best_votes = counter.most_common(1)[0]
        T = np.array(best_trans_tuple)
        print(f"估计的全局平移向量: {T}，得票 {best_votes}")

        # 3. 根据 T 预测所有历史对象在当前帧的位置
        predicted_coords = {}   # 历史ID -> 预测坐标 (x, y)
        predicted_array = self.history_array + T   # 形状 (N, 2)，顺序与 all_ids 一致
        for oid, coord in zip(self.all_ids, predicted_array):
            predicted_coords[oid] = coord

        # 4. 用 KDTree 快速找到每个检测点对应的最近预测点
        tree = KDTree(predicted_array)
        distances, indices = tree.query(det_points)   # distances: M个最小距离, indices: M个对应的历史数组索引

        id_map = {}
        used_hist_ids = set()
        for det_idx in range(len(det_points)):
            dist = distances[det_idx]
            hist_idx = indices[det_idx]
            if dist <= distance_thresh:
                hist_id = self.all_ids[hist_idx]
                if hist_id not in used_hist_ids:
                    id_map[det_idx] = hist_id
                    used_hist_ids.add(hist_id)

        # 5. 找出被移动的棋子
        moved_piece = None
        piece_set = set(self.piece_ids)
        matched_piece_ids = {hid for hid in id_map.values() if hid in piece_set}

        # 检查是否有检测到的棋子没有被匹配上（离预测位置太远）
        for det_idx, info in enumerate(det_info):
            if info[0] == 'piece' and det_idx not in id_map:
                # 这个棋子是移动过的
                unmatched_ids = piece_set - matched_piece_ids
                if len(unmatched_ids) == 1:
                    moved_id = list(unmatched_ids)[0]
                    moved_piece = (moved_id, det_points[det_idx])
                    id_map[det_idx] = moved_id      # 补齐匹配
                    matched_piece_ids.add(moved_id)

        # 如果还没找到，但历史棋子少了一个，说明被移出画面了
        if moved_piece is None:
            unmatched_ids = piece_set - matched_piece_ids
            if len(unmatched_ids) == 1:
                moved_piece = (list(unmatched_ids)[0], None)

        # 6. 找出被遮挡的棋盘格
        square_set = set(self.square_ids)
        matched_square_ids = {hid for hid in id_map.values() if hid in square_set}
        missing_squares = list(square_set - matched_square_ids)

        # 7. 生成所有对象的推断位置
        inferred_coords = {}
        for oid in self.all_ids:
            if oid in id_map.values():
                for det_idx, hid in id_map.items():
                    if hid == oid:
                        inferred_coords[oid] = tuple(det_points[det_idx].tolist())
                        break
            else:
                inferred_coords[oid] = tuple(predicted_coords[oid].tolist())

        return id_map, moved_piece, missing_squares, inferred_coords

"""
使用方法：
# 1. 定义你所有棋子和棋盘格的固定编号
tracker = BoardTrackerVoting(
    piece_ids=['P0','P1','P2','P3','P4','P5','P6','P7'],
    square_ids=['S0','S1','S2','S3','S4','S5','S6','S7','S8']
)

# 2. 第一帧稳定时初始化（所有对象都检测到了，棋子未移动）
init_pieces = [(120,200), (180,210), ...]   # 你的8个棋子坐标（顺序按你的编号）
init_squares = [(50,50), (150,50), ...]     # 你的9个格子坐标
tracker.initialize(init_pieces, init_squares)

# 3. 以后每一帧，传入当前检测结果
curr_pieces = [(121,201), (182,208), ...]   # 检测到的所有棋子（顺序随意）
curr_squares = [(51,51), (152,49)]          # 检测到的所有格子（可能不全）
id_map, moved, missing, inferred = tracker.update(curr_pieces, curr_squares)

print("匹配关系:", id_map)
print("被移动的棋子:", moved)
print("被遮挡的格子:", missing)
print("所有对象推断位置:", inferred)
"""