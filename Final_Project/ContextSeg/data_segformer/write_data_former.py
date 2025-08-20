import os
import ujson as json
import numpy as np
from PIL import Image, ImageDraw
import torch # 引入 torch
import random

# --- 路徑設定 ---
source_folder = '../SketchX-PRIS-Dataset-master/Perceptual Grouping/'
output_dir = 'data_former_pt' # 建立新的資料夾
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)

# --- 主要邏輯 ---
all_sketches = []
print("正在從 .ndjson 檔案讀取資料...")
for filename in os.listdir(source_folder):
    if filename.endswith('.ndjson'):
        filepath = os.path.join(source_folder, filename)
        with open(filepath, 'r', encoding='utf8') as fp:
            try:
                if os.path.getsize(filepath) > 0:
                    json_data = json.load(fp)
                    # 合併訓練和測試資料
                    all_sketches.extend(json_data.get("train_data", []))
                    all_sketches.extend(json_data.get("test_data", []))
            except json.JSONDecodeError:
                print(f"警告：無法解析 {filename}，已跳過。")

random.shuffle(all_sketches)

train_ratio = 0.8
split_index = int(len(all_sketches) * train_ratio)
train_data = all_sketches[:split_index]
test_data = all_sketches[split_index:]

img_size = 156

def get_bounds(data, factor=1):
  min_x, max_x, min_y, max_y = 0, 0, 0, 0
  abs_x, abs_y = 0, 0
  for i in range(len(data)):
    x, y = float(data[i][0]) / factor, float(data[i][1]) / factor
    abs_x += x; abs_y += y
    min_x, min_y = min(min_x, abs_x), min(min_y, abs_y)
    max_x, max_y = max(max_x, abs_x), max(max_y, abs_y)
  return (min_x, max_x, min_y, max_y)

def scale_bound(stroke, average_dimension=img_size):
  bounds = get_bounds(stroke, 1)
  max_dimension = max(bounds[1] - bounds[0], bounds[3] - bounds[2])
  if max_dimension == 0: return np.array(stroke)
  stroke = np.array(stroke)
  scale = (max_dimension / average_dimension)
  stroke[:,0:2] = stroke[:,0:2]/scale
  return stroke

def find_duplicate_indices(lst, num_groups):
    result = [[] for _ in range(num_groups)]
    for i, num in enumerate(lst):
        if int(num) < num_groups and int(num) >= 0:
            result[int(num)].append(i)
    return result

def strokes_to_lines(strokes):
  strokes = scale_bound(strokes)
  x, y = 0, 0
  lines, line, group_id = [], [], []
  cur_group_ip = -1
  for i in range(len(strokes)):
    if strokes[i][2] == 1:
      x += float(strokes[i][0]); y += float(strokes[i][1])
      line.append([x, y])
      group_id.append(cur_group_ip)
      lines.append(line)
      line = []
    else:
      x += float(strokes[i][0]); y += float(strokes[i][1])
      line.append([x, y])
      cur_group_ip = strokes[i][3]
  return lines, group_id

def process_data(data_subset):
    processed_list = []
    for inx, line_list in enumerate(data_subset):
        lines, group_id = strokes_to_lines(line_list)
        nb_stroke = len(group_id)
        if nb_stroke == 0: continue
        
        unique_groups = {g for g in group_id if g != -1}
        nb_group = len(unique_groups) if unique_groups else 0
        if nb_group == 0: continue
        
        group_map = {gid: i for i, gid in enumerate(sorted(list(unique_groups)))}
        
        index_group = find_duplicate_indices([group_map.get(g, -1) for g in group_id], nb_group)
        
        glabel = np.zeros((nb_group, nb_stroke), dtype=np.int64)
        for row, row_indices in enumerate(index_group):
            for col in row_indices:
                glabel[row][col] = 1
        
        temp_for_input_raw = []
        for line in lines:
            if len(line) < 2: line.append(line[0])
            img = Image.new('1', (img_size, img_size), 0)
            draw = ImageDraw.Draw(img)
            pixels = [(int(x), int(y)) for x, y in line]
            draw.line(pixels, fill=1, width=2)
            arr = np.array(img)
            
            arr_with_pad = np.zeros((256, 256))
            start_row, start_col = (256 - img_size) // 2, (256 - img_size) // 2
            arr_with_pad[start_row:start_row + img_size, start_col:start_col + img_size] = arr
            temp_for_input_raw.append(arr_with_pad)
        
        if temp_for_input_raw:
            input_raw = np.array(temp_for_input_raw, dtype=np.float32).transpose(1, 2, 0)
            processed_list.append({'img_raw': input_raw, 'glabel_raw': glabel})

    return processed_list

def write_former_pt_files(dataset, subset_name):
    print(f"正在寫入 {subset_name} 資料，共 {len(dataset)} 筆...")
    subset_path = os.path.join(output_dir, subset_name)
    for i, data in enumerate(dataset):
        data_to_save = {
            'img_raw': torch.from_numpy(data['img_raw']).float(),
            'glabel_raw': torch.from_numpy(data['glabel_raw']).long()
        }
        torch.save(data_to_save, os.path.join(subset_path, f'{i}.pt'))
    print(f"{subset_name} 寫入完成。")

print("處理訓練資料...")
train_processed = process_data(train_data)
write_former_pt_files(train_processed, 'train')

print("處理測試資料...")
test_processed = process_data(test_data)
write_former_pt_files(test_processed, 'test')
