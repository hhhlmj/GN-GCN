import pandas as pd
import os

# ==================== 配置区域 ====================

# 获取当前脚本文件所在的绝对路径
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# 输入文件 (你刚刚上传的这个)
LOCGRID_FILE = os.path.join(DATA_DIR, "locgrid_one2one_level7-18.csv")

# 还需要读取数据集的时间信息
TRAIN_FILE = os.path.join(DATA_DIR, "train.txt")
VALID_FILE = os.path.join(DATA_DIR, "valid.txt")
TEST_FILE = os.path.join(DATA_DIR, "test.txt")

# 输出文件
GRID_NEIGHBOR_FILE = os.path.join(DATA_DIR, "gridneighbor.txt")
ENTITY_LOC_NEIGHBOR_FILE = os.path.join(DATA_DIR, "entityloc_neighbor.txt")

LEVEL = 7  # GeoSOT Level


# =================================================

def get_geosot_row_col(code_str):
    """将二进制 GeoSOT 码解析为 (Row, Col)"""
    # 确保是字符串格式，防止被当成数字读取
    code_binary = str(code_str).strip()

    # 如果前面丢了0，需要补齐到 14 位
    if len(code_binary) < 2 * LEVEL:
        code_binary = code_binary.zfill(2 * LEVEL)

    row_bin = []
    col_bin = []
    # 假设偶数位为 Col，奇数位为 Row (Z-order)
    for i in range(0, len(code_binary), 2):
        row_bin.append(code_binary[i])
        col_bin.append(code_binary[i + 1])

    row = int("".join(row_bin), 2)
    col = int("".join(col_bin), 2)
    return row, col


def row_col_to_geosot(row, col, level):
    """将 (Row, Col) 编码回二进制 GeoSOT"""
    row_bin = format(row, f'0{level}b')
    col_bin = format(col, f'0{level}b')
    code = []
    for i in range(level):
        code.append(row_bin[i])
        code.append(col_bin[i])
    return "".join(code)


def generate_grid_neighbor(locgrid_df):
    """
    生成 gridneighbor.txt
    """
    print("1. 生成网格邻居图 (GridNeighbor)...")

    # 确保 geosot 列是字符串类型
    locgrid_df['geosot'] = locgrid_df['geosot'].astype(str).str.zfill(2 * LEVEL)
    unique_grids = locgrid_df['geosot'].unique()
    neighbors_list = []

    max_idx = 2 ** LEVEL - 1

    for code in unique_grids:
        try:
            row, col = get_geosot_row_col(code)

            # 遍历 3x3 邻域
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0: continue

                    nr, nc = row + dr, col + dc

                    if 0 <= nr <= max_idx and 0 <= nc <= max_idx:
                        n_code = row_col_to_geosot(nr, nc, LEVEL)
                        # 这里我们记录: 当前网格 \t 邻居网格
                        neighbors_list.append(f"{code}\t{n_code}")
        except Exception as e:
            print(f"Skipping code {code}: {e}")
            continue

    with open(GRID_NEIGHBOR_FILE, 'w') as f:
        f.write("\n".join(neighbors_list))
    print(f"   已保存: {GRID_NEIGHBOR_FILE}")


def generate_entityloc_neighbor(locgrid_df):
    """
    生成 entityloc_neighbor.txt
    """
    print("2. 生成实体-位置-时间图 (EntityLoc Neighbor)...")

    locgrid_df['geosot'] = locgrid_df['geosot'].astype(str).str.zfill(2 * LEVEL)
    # 建立 ID 到 Grid 的映射
    e2g = dict(zip(locgrid_df['id'], locgrid_df['geosot']))

    output_lines = []

    # 遍历所有数据集文件，提取 (Entity, Time)
    for filepath in [TRAIN_FILE, VALID_FILE, TEST_FILE]:
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found, skipping.")
            continue

        print(f"   Processing {filepath}...")
        try:
            # 读取四元组 (s, r, o, t)
            # ICEWS 格式通常是: head  rel  tail  time ...
            df = pd.read_csv(filepath, sep='\t', header=None)
            data = df.iloc[:, :4].values

            for row in data:
                s, r, o, t = row

                # 如果头实体有位置信息
                if s in e2g:
                    # 格式: EntityID \t GridCode \t Time
                    output_lines.append(f"{s}\t{e2g[s]}\t{t}")

                # 如果尾实体有位置信息
                if o in e2g:
                    output_lines.append(f"{o}\t{e2g[o]}\t{t}")
        except Exception as e:
            print(f"Error reading {filepath}: {e}")

    # 去重
    output_lines = list(set(output_lines))

    with open(ENTITY_LOC_NEIGHBOR_FILE, 'w') as f:
        f.write("\n".join(output_lines))
    print(f"   已保存: {ENTITY_LOC_NEIGHBOR_FILE}")


if __name__ == "__main__":
    print("加载 LocGrid 文件...")
    # 强制读取为字符串，防止前导0丢失
    locgrid_df = pd.read_csv(LOCGRID_FILE, sep='\t', dtype={'geosot': str})

    generate_grid_neighbor(locgrid_df)
    generate_entityloc_neighbor(locgrid_df)

    print("\n完成！")