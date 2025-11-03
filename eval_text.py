import os
import json
import base64
import argparse
import logging
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

def image2base64(image_path):
    with open(image_path, 'rb') as f:
        image = f.read()
    return base64.b64encode(image).decode()

# ===== CHANGED: 接受两个“文本”，而不是图像 =====
def predict(question,model, url, api_key):
    client = OpenAI(
        base_url=url,
        api_key=api_key,
    )

    # ===== CHANGED: 一次消息里放入两段文本 =====
    content = [
        {"type": "text", "text": question}
    ]

    messages = [{"role": "user", "content": content}]
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0
    )
    return completion.choices[0].message.content

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

def process_item(item, model, url, api_key, jsonl_path):
    try:
        if 'output_geogebra_status' not in item:
            raise KeyError(f"Item missing required field 'output_geogebra_status'. ID: {item.get('id', 'unknown')}")

         # 新增：根据 output_geogebra_status 判断是否推理
        status = item.get('output_geogebra_status')

        if not _is_true_flag(status):
            # 直接给 0，跳过模型调用
            item['eval_text_result'] = "0"
            append_jsonl(jsonl_path, item)
            logging.info(f"ID: {item.get('id', 'unknown')} skipped (output_geogebra_status={status}); set result=0")
            return item
        
        # ===== 这两个字段现在被当作“文本”使用 =====
        text2 = item['文本答案']  # 原“模型最终图”字段，此处作为第二段文本
        text1 = item['问题']
        text3 = item['output']               # 原“问题图”字段，此处作为第一段文本

        prompt =  f"""
【Role】
You are an evaluator of the “text steps” for geometric multimodal constructions.

【Input】

* Problem description
* Reference answer: standard construction steps + image code (image code is only for corroboration; steps take precedence)
* Model’s answer: model’s construction steps + model-generated image code (image code is only for corroboration)

【Evaluation Criteria】

1. Completeness of the reasoning chain: whether the construction order is clear, with no skipped steps, and dependencies are explicit.
2. Accuracy of steps: whether it covers the key constructions required by the problem (e.g., perpendicular lines/parallel lines/angle bisectors/circles and auxiliary circles/points of tangency and tangents/naming of intersection points/relationship annotations, etc.).
3. Consistency with geometric principles: whether each step adheres to geometric principles (collinearity/concyclicity, equality/proportionality, perpendicular/parallel/tangency, etc.).
4. Consistency with the standard solution: “equivalent construction paths” are allowed as long as the final objects and constraints satisfy the problem requirements.

【Scoring Rubric (1–5)】
5: Steps complete, logically rigorous, key constructions all present; consistent with or equivalent to the standard; no geometric errors.
4: Overall correct, with minor differences/small omissions that do not affect reproducibility or conclusions.
3: Covers most key points but lacks critical steps or contains obvious errors that are fixable.
2: Numerous logical errors or deviates from the problem intent; the chain is not coherent.
1: Seriously inconsistent, many errors, or cannot be reproduced.

【Output Requirements (Extremely Important)】

* Output only a single Arabic numeral (1–5) as the final result, with no other text, punctuation, or spaces.

Problem description: {text1}
Reference answer:{text2}
Model’s answer{text3}

""" 

        # ===== CHANGED: 传入两段文本（顺序：参考文本 -> 模型文本）=====
        result = predict(prompt, model, url, api_key)
        item['eval_text_result'] = result
        append_jsonl(jsonl_path, item)
        return item
    except Exception as e:
        logging.error(f"ID: {item.get('id', 'unknown')} Error: {e}")
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default="E:\ECode\huitu\eval\eval_output\gemini-2.5-pro\VLM_eval_image_result.json")

    # 注意不同模型输出到不同的文件中  (被测模型)
    parser.add_argument('--output_dir', type=str, default="E:\ECode\huitu\eval\eval_output\gemini-2.5-pro")
    # 裁判模型 gpt5
    parser.add_argument('--model', type=str, default="gpt-5")

    # 固定
    parser.add_argument('--url', type=str, default="http://35.220.164.252:3888/v1/")
    parser.add_argument('--api_key', type=str, default="sk-KjCTTVFCwTsTEl0Z6vNPcPA2qKEpLfKHavBgnNkBDtdSxAE2")

    # 并发数，越大越快 推荐20
    parser.add_argument('--max_workers', type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    jsonl_path = os.path.join(args.output_dir, 'eval_text_result.jsonl')
    json_path = os.path.join(args.output_dir, 'eval_text_result.json')
    log_path = os.path.join(args.output_dir, 'eval_text_result.log')

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
            futures.append(executor.submit(process_item, item, args.model, args.url, args.api_key, jsonl_path))

        for future in tqdm(as_completed(futures), total=len(futures)):
            future.result()
    # processed_items = set()
    # if os.path.exists(jsonl_path):
    #     processed_items = {json.dumps(item, sort_keys=True) for item in read_jsonl(jsonl_path)}

    # with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
    #     futures = []
    #     for item in input_data:
    #         if json.dumps(item, sort_keys=True) in processed_items:
    #             continue
    #         futures.append(executor.submit(process_item, item, args.model, args.url, args.api_key, jsonl_path))

    #     for future in tqdm(as_completed(futures), total=len(futures)):
    #         future.result()

    jsonl2json(jsonl_path, json_path)
    print(f"Processing complete. Output saved to: {json_path}")
