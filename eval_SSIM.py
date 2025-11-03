import os
import json
import base64
import argparse
import logging
from tqdm import tqdm
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from concurrent.futures import ThreadPoolExecutor, as_completed

# 计算SSIM
def calculate_ssim(image_path_1, image_path_2):
    """计算两张图片的SSIM"""
    img1 = Image.open(image_path_1).convert('RGB')
    img2 = Image.open(image_path_2).convert('RGB')

    # 调整图片尺寸为相同大小（例如：256x256）
    img1 = resize_image(img1, target_size=(256, 256))
    img2 = resize_image(img2, target_size=(256, 256))

    # 转换为灰度图（SSIM通常用于灰度图像）
    img1_gray = np.array(img1.convert('L'))
    img2_gray = np.array(img2.convert('L'))

    # 计算SSIM
    ssim_value, _ = ssim(img1_gray, img2_gray, full=True)

    return ssim_value

def resize_image(image, target_size=(256, 256)):
    """调整图片为指定的大小"""
    return image.resize(target_size, Image.Resampling.LANCZOS)

def predict(image_path_1, image_path_2):
    """现在的`predict`函数计算SSIM"""
    # 直接计算SSIM
    ssim_score = calculate_ssim(image_path_1, image_path_2)
    
    # 返回SSIM分数作为评估结果
    return str(ssim_score)  # 只返回数值，不带有其他文本

def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def append_jsonl(file_path, data):
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')

def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def jsonl2json(jsonl_path, json_path):
    data = read_jsonl(jsonl_path)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def _is_true_flag(val):
    """仅当为 True 或 'true'(大小写不敏感) 时返回 True，其它情况均为 False。"""
    if isinstance(val, bool):
        return val is True
    if isinstance(val, str):
        return val.strip().lower() == "true"
    return False

def process_item(item, jsonl_path):
    try:
        # if 'output_geogebra_status' not in item:
        #     raise KeyError(f"Item missing required field 'output_geogebra_status'. ID: {item.get('id', 'unknown')}")

        # # 新增：根据 output_geogebra_status 判断是否推理
        # status = item.get('output_geogebra_status')

        # if not _is_true_flag(status):
        #     # 直接给 0，跳过模型调用
        #     item['ssim_eval_image_result'] = "0"
        #     append_jsonl(jsonl_path, item)
        #     logging.info(f"ID: {item.get('id', 'unknown')} skipped (output_geogebra_status={status}); set result=0")
        #     return item

        output_image = item['image_path']     # 模型最终图
        image_path = item['最终结果图']                    # 题目/参考图（视数据而定）

        # 使用SSIM计算两张图的相似度
        result = predict(image_path, output_image)
        item['ssim_eval_image_result'] = result
        append_jsonl(jsonl_path, item)
        return item
    except Exception as e:
        logging.error(f"ID: {item.get('id', 'unknown')} Error: {e}")
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default="E:\ECode\huitu\eval\eval_output\qwen-image_images\psnr_eval_image_result.json")

    # 注意不同模型输出到不同的文件中  (被测模型)
    parser.add_argument('--output_dir', type=str, default="E:\ECode\huitu\eval\eval_output\qwen-image_images")

    # 并发数，越大越快 推荐20
    parser.add_argument('--max_workers', type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    jsonl_path = os.path.join(args.output_dir, 'ssim_eval_image_result.jsonl')
    json_path = os.path.join(args.output_dir, 'ssim_eval_image_result.json')
    log_path = os.path.join(args.output_dir, 'ssim_eval_image_result.log')

    logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    input_data = read_json(args.input_path)

    processed_ids = set()
    if os.path.exists(jsonl_path):
        existing_items = read_jsonl(jsonl_path)
        for e in existing_items:
            if 'id' in e:
                processed_ids.add(e['id'])

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = []
        for item in input_data:
            item_id = item.get('id')
            if item_id in processed_ids:
                logging.info(f"Skip duplicate id={item_id}")
                continue
            futures.append(executor.submit(process_item, item, jsonl_path))

        for future in tqdm(as_completed(futures), total=len(futures)):
            future.result()

    jsonl2json(jsonl_path, json_path)
    print(f"Processing complete. Output saved to: {json_path}")
