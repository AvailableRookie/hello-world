# import os
# import json
# import base64
# import argparse
# import logging
# import uuid  # 导入uuid库
# from tqdm import tqdm
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from openai import OpenAI

# def generate_image(prompt, model, url, api_key, output_dir, image_filename_no_ext):
#     """
#     根据提示生成图像，并将其下载为本地文件。
#     使用 image_filename_no_ext 作为不带 .png 扩展名的文件名。
#     """
#     client = OpenAI(
#         base_url=url,
#         api_key=api_key,
#     )

#     try:
#         # 调用图像生成API
#         response = client.chat.completions.create(
#             model=model,
#             messages=[{"role": "user", "content": prompt}],
#             temperature=1,
#         )

#         # 提取API返回的图像数据（base64格式）
#         image_data = response.choices[0].message.content.strip()

#         # 检查返回的数据是否以正确的base64前缀开始
#         if image_data.startswith("![image](data:image/png;base64,"):
#             # 去掉前缀，得到纯粹的base64编码字符串
#             image_data = image_data[len("![image](data:image/png;base64,"):-1]  # 移除前缀和闭合括号
#         else:
#             raise ValueError(f"ID {image_filename_no_ext} 图像生成失败，返回数据格式无效。")

#         # 解码base64字符串
#         image_content = base64.b64decode(image_data)

#         # 将图像保存到指定目录
#         images_add = os.path.join(output_dir, "images")
#         os.makedirs(images_add, exist_ok=True)
#         # 使用传入的完整文件名（不带扩展名）
#         image_path = os.path.join(images_add, f"{image_filename_no_ext}.png")

#         with open(image_path, 'wb') as img_file:
#             img_file.write(image_content)

#         # logging.info(f"ID {image_filename_no_ext}: 图像已保存到 {image_path}") # 在 process_item 中记录
#         return image_path

#     except Exception as e:
#         logging.error(f"ID {image_filename_no_ext} 在图像生成或下载时发生错误: {e}")
#         return None

# def read_json(file_path):
#     """读取JSON文件"""
#     with open(file_path, 'r', encoding='utf-8') as f:
#         return json.load(f)

# def append_jsonl(file_path, data):
#     """将数据追加到JSONL文件中"""
#     with open(file_path, 'a', encoding='utf-8') as f:
#         f.write(json.dumps(data, ensure_ascii=False) + '\n')

# def read_jsonl(file_path):
#     """读取JSONL文件"""
#     with open(file_path, 'r', encoding='utf-8') as f:
#         return [json.loads(line) for line in f]

# def jsonl2json(jsonl_path, json_path):
#     """将JSONL文件转换为JSON文件"""
#     data = read_jsonl(jsonl_path)
#     with open(json_path, 'w', encoding='utf-8') as f:
#         json.dump(data, f, ensure_ascii=False, indent=4)

# def process_item(item, model, url, api_key, output_dir, jsonl_path):
#     """
#     处理每个数据项，根据 step_num 字段生成多张累积图像。
#     """
    
#     # 优先使用 'id'，如果不存在则生成一个UUID作为图像文件名和日志记录的标识符
#     item_id_for_log_and_file = item.get('id')
#     if item_id_for_log_and_file is None:
#         item_id_for_log_and_file = str(uuid.uuid4())
#         logging.warning(f"项 {item} 没有 'id'。将使用临时ID: {item_id_for_log_and_file}")
#         # 如果没有 'id'，我们也应该给它分配一个 'id' 字段以便于跟踪
#         item['id'] = item_id_for_log_and_file

#     try:
#         step_num = item.get('step_num')

#         # 验证 step_num 是否有效
#         if not isinstance(step_num, int) or step_num <= 0:
#             logging.warning(f"ID: {item_id_for_log_and_file} 'step_num' 字段无效或缺失: {step_num}。跳过此项。")
#             return None # 跳过这个 item

#         base_prompt = "Generate an image illustrating according to the following steps.\n### **Step-by-Step Solution**\n"
#         cumulative_steps_content = ""
#         all_steps_succeeded = True

#         # 循环 N 次 (N = step_num)
#         for i in range(1, step_num + 1):
#             step_key = f'step_{i}'
#             step_content = item.get(step_key)

#             # 检查是否存在 step_i 的内容
#             if not step_content:
#                 logging.error(f"ID: {item_id_for_log_and_file} 找不到字段 {step_key} 的内容。终止此项的后续步骤。")
#                 all_steps_succeeded = False
#                 break  # 中断此 item 的循环

#             # 构建累积的 prompt 内容
#             if i == 1:
#                 cumulative_steps_content = step_content
#             else:
#                 cumulative_steps_content += f"\n+\n{step_content}"
            
#             # 最终的 prompt
#             prompt = base_prompt + cumulative_steps_content

#             # 构造文件名：id_step_i
#             image_filename = f"{item_id_for_log_and_file}_step_{i}"

#             # 调用生成图像函数并下载图像
#             image_path = generate_image(prompt, model, url, api_key, output_dir, image_filename)

#             if image_path:
#                 # 如果成功，将图像路径保存到 item 中
#                 item[f'image_path_step_{i}'] = image_path
#                 logging.info(f"ID: {item_id_for_log_and_file} 步骤 {i}/{step_num} 图像已保存: {image_path}")
#             else:
#                 # 如果失败
#                 logging.error(f"ID: {item_id_for_log_and_file} 步骤 {i}/{step_num} 图像生成失败。终止此项。")
#                 all_steps_succeeded = False
#                 break # 中断此 item 的循环

#         # 只有当所有步骤都成功时，才将 item 追加到 jsonl
#         if all_steps_succeeded:
#             append_jsonl(jsonl_path, item)
#             logging.info(f"ID: {item_id_for_log_and_file} 所有 {step_num} 个步骤均已处理并保存到 JSONL。")
#         else:
#             logging.warning(f"ID: {item_id_for_log_and_file} 未能完成所有步骤，此项未保存到 JSONL。")

#         return item
    
#     except Exception as e:
#         logging.error(f"ID: {item_id_for_log_and_file} 在处理时发生严重错误: {e}")
#         return None

# if __name__ == '__main__':
#     # 设置命令行参数
#     parser = argparse.ArgumentParser()
#     # ！！！注意：这里的输入文件现在应该是包含 step_num 和 step_i 字段的 JSON 文件
#     # ！！！例如，您在第一个请求中生成的 step_output.json
#     parser.add_argument('--input_path', type=str, default="E:\ECode\huitu\eval\eval_output\gemini-images-text\step_output.json") # ！！！确保这是包含步骤的JSON文件
#     parser.add_argument('--output_dir', type=str, default="E:\ECode\huitu\eval\eval_output\gemini-image-steps") # 建议使用新的输出目录
#     parser.add_argument('--model', type=str, default="gemini-2.5-flash-image") # 使用的图像生成模型
#     parser.add_argument('--url', type=str, default="https://api.boyuerichdata.opensphereai.com/v1") # API URL
#     parser.add_argument('--api_key', type=str, default="sk-KjCTTVFCwTsTEl0Z6vNPcPA2qKEpLfKHavBgnNkBDtdSxAE2") # API 密钥
#     parser.add_argument('--max_workers', type=int, default=1) # 并行处理线程数
#     args = parser.parse_args()

#     # 创建输出目录
#     os.makedirs(args.output_dir, exist_ok=True)
#     jsonl_path = os.path.join(args.output_dir, 'output.jsonl')
#     json_path = os.path.join(args.output_dir, 'output.json')
#     log_path = os.path.join(args.output_dir, 'process.log')

#     # 设置日志记录
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(levelname)s - %(message)s',
#         handlers=[
#             logging.FileHandler(log_path, encoding='utf-8'),
#             logging.StreamHandler() # 同时输出到控制台
#         ]
#     )

#     # 读取输入数据
#     input_data = read_json(args.input_path)
#     processed_ids = set()

#     # --- 逻辑不变：读取已处理的ID ---
#     if os.path.exists(jsonl_path):
#         try:
#             processed_data = read_jsonl(jsonl_path)
#             # 假设每个条目在jsonl中仍然包含 'id' 字段
#             processed_ids = {item.get('id') for item in processed_data if 'id' in item}
#             logging.info(f"已从 {jsonl_path} 加载 {len(processed_ids)} 个已处理的 ID。")
#         except Exception as e:
#             logging.error(f"读取或解析 {jsonl_path} 时出错: {e}。将尝试处理所有数据。")
#             processed_ids = set() # 如果文件损坏，则重置

#     # --- 逻辑不变：过滤掉已经处理过的数据项 ---
#     items_to_process = []
#     for item in input_data:
#         item_id = item.get('id')
#         if item_id is None:
#             logging.warning(f"输入数据中发现一个没有 'id' 的项: {item}。将尝试处理。")
#             items_to_process.append(item)
#         elif item_id in processed_ids:
#             # logging.info(f"跳过已处理的项，ID: {item_id}") # 注释掉以避免过多日志
#             continue
#         else:
#             items_to_process.append(item)
    
#     if len(processed_ids) > 0:
#         logging.info(f"总共 {len(input_data)} 项。已处理 {len(processed_ids)} 项。")
#     logging.info(f"本次需要处理 {len(items_to_process)} 项。")

#     # 使用线程池并行处理数据项
#     with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
#         futures = {executor.submit(process_item, item, args.model, args.url, args.api_key, args.output_dir, jsonl_path): item for item in items_to_process}

#         # 使用tqdm显示进度
#         with tqdm(total=len(items_to_process), desc="处理中") as pbar:
#             for future in as_completed(futures):
#                 item = futures[future]
#                 try:
#                     future.result()  # 获取结果（或在线程中发生的异常）
#                 except Exception as e:
#                     logging.error(f"ID: {item.get('id', 'unknown')} 在线程执行中发生未捕获的错误: {e}")
#                 pbar.update(1) # 更新进度条

#     # 将结果从JSONL转换为JSON格式
#     try:
#         if os.path.exists(jsonl_path): # 确保jsonl文件存在
#             jsonl2json(jsonl_path, json_path)
#             logging.info(f"处理完成，最终JSON结果已保存到: {json_path}")
#         else:
#             logging.info("没有生成 output.jsonl 文件，无需转换。")
#     except Exception as e:
#         logging.error(f"JSONL到JSON转换失败: {e}")

#     print(f"所有处理已完成。日志文件位于: {log_path}")
import os
import json
import base64
import argparse
import logging
import uuid
import re  # ！！！新增：导入 re 库用于正则表达式
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

def generate_image(prompt, model, url, api_key, output_dir, image_filename_no_ext):
    """
    根据提示生成图像，并将其下载为本地文件。
    使用 image_filename_no_ext 作为不带 .png 扩展名的文件名。
    """
    client = OpenAI(
        base_url=url,
        api_key=api_key,
    )

    try:
        # 调用图像生成API
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=1,
        )

        # 提取API返回的完整响应文本
        response_text = response.choices[0].message.content.strip()

        # --- ！！！修复逻辑：使用 Regex 提取 base64 数据 ---
        # 搜索 "![image](data:image/png;base64,..." 格式
        # 这个正则表达式会查找 Markdown 图像标签并捕获括号内的 base64 数据
        match = re.search(r"!\[image\]\(data:image/png;base64,([^)]+)\)", response_text)

        if match:
            # 提取第一个捕获组 (即括号内的 base64 字符串)
            image_data = match.group(1)
        else:
            # 如果在响应中找不到匹配的格式
            logging.error(f"ID {image_filename_no_ext} 无法在响应中找到 base64 图像数据。响应截断: {response_text[:200]}...")
            raise ValueError(f"ID {image_filename_no_ext} 图像生成失败，返回数据格式无效。")
        # --- 修复结束 ---

        # 解码base64字符串
        image_content = base64.b64decode(image_data)

        # 将图像保存到指定目录
        images_add = os.path.join(output_dir, "images")
        os.makedirs(images_add, exist_ok=True)
        # 使用传入的完整文件名（不带扩展名）
        image_path = os.path.join(images_add, f"{image_filename_no_ext}.png")

        with open(image_path, 'wb') as img_file:
            img_file.write(image_content)

        # logging.info(f"ID {image_filename_no_ext}: 图像已保存到 {image_path}") # 在 process_item 中记录
        return image_path

    except Exception as e:
        # 区分 base64 解码错误
        if isinstance(e, base64.binascii.Error):
            logging.error(f"ID {image_filename_no_ext} Base64 解码失败: {e}. 提取的数据可能不完整。")
            return None
        logging.error(f"ID {image_filename_no_ext} 在图像生成或下载时发生错误: {e}")
        return None

def read_json(file_path):
    """读取JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def append_jsonl(file_path, data):
    """将数据追加到JSONL文件中"""
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')

def read_jsonl(file_path):
    """读取JSONL文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def jsonl2json(jsonl_path, json_path):
    """将JSONL文件转换为JSON文件"""
    data = read_jsonl(jsonl_path)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def process_item(item, model, url, api_key, output_dir, jsonl_path):
    """
    处理每个数据项，根据 step_num 字段生成多张累积图像。
    """
    
    # 优先使用 'id'，如果不存在则生成一个UUID作为图像文件名和日志记录的标识符
    item_id_for_log_and_file = item.get('id')
    if item_id_for_log_and_file is None:
        item_id_for_log_and_file = str(uuid.uuid4())
        logging.warning(f"项 {item} 没有 'id'。将使用临时ID: {item_id_for_log_and_file}")
        # 如果没有 'id'，我们也应该给它分配一个 'id' 字段以便于跟踪
        item['id'] = item_id_for_log_and_file

    try:
        step_num = item.get('step_num')

        # 验证 step_num 是否有效
        if not isinstance(step_num, int) or step_num <= 0:
            logging.warning(f"ID: {item_id_for_log_and_file} 'step_num' 字段无效或缺失: {step_num}。跳过此项。")
            return None # 跳过这个 item

        base_prompt = """
Generate a **single, composite image** illustrating the following multi-step geometric construction.

The final image must be **cumulative**, meaning all steps are drawn on the same canvas and layered on top of each other. All construction lines, arcs, and points from **every step** must be retained and visible in the final output.
"""
        cumulative_steps_content = ""
        all_steps_succeeded = True

        # 循环 N 次 (N = step_num)
        for i in range(1, step_num + 1):
            step_key = f'step_{i}'
            step_content = item.get(step_key)

            # 检查是否存在 step_i 的内容
            if not step_content:
                logging.error(f"ID: {item_id_for_log_and_file} 找不到字段 {step_key} 的内容。终止此项的后续步骤。")
                all_steps_succeeded = False
                break  # 中断此 item 的循环

            # 构建累积的 prompt 内容
            if i == 1:
                cumulative_steps_content = step_content
            else:
                cumulative_steps_content += f"\n+\n{step_content}"
            
            # 最终的 prompt
            prompt = base_prompt + cumulative_steps_content

            # 构造文件名：id_step_i
            image_filename = f"{item_id_for_log_and_file}_step_{i}"

            # 调用生成图像函数并下载图像
            image_path = generate_image(prompt, model, url, api_key, output_dir, image_filename)

            if image_path:
                # 如果成功，将图像路径保存到 item 中
                item[f'image_path_step_{i}'] = image_path
                logging.info(f"ID: {item_id_for_log_and_file} 步骤 {i}/{step_num} 图像已保存: {image_path}")
            else:
                # 如果失败
                logging.error(f"ID: {item_id_for_log_and_file} 步骤 {i}/{step_num} 图像生成失败。终止此项。")
                all_steps_succeeded = False
                break # 中断此 item 的循环

        # 只有当所有步骤都成功时，才将 item 追加到 jsonl
        if all_steps_succeeded:
            append_jsonl(jsonl_path, item)
            logging.info(f"ID: {item_id_for_log_and_file} 所有 {step_num} 个步骤均已处理并保存到 JSONL。")
        else:
            logging.warning(f"ID: {item_id_for_log_and_file} 未能完成所有步骤，此项未保存到 JSONL。")

        return item
    
    except Exception as e:
        logging.error(f"ID: {item_id_for_log_and_file} 在处理时发生严重错误: {e}")
        return None

if __name__ == '__main__':
    # 设置命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default="E:\ECode\huitu\eval\eval_output\gemini-images-text\step_output.json") # 确保这是包含步骤的JSON文件
    parser.add_argument('--output_dir', type=str, default="E:/ECode/huitu/eval/eval_output/gemini-image-steps-new") # 建议使用新的输出目录
    parser.add_argument('--model', type=str, default="gemini-2.5-flash-image") # 使用的图像生成模型
    parser.add_argument('--url', type=str, default="https://api.boyuerichdata.opensphereai.com/v1") # API URL
    parser.add_argument('--api_key', type=str, default="sk-KjCTTVFCwTsTEl0Z6vNPcPA2qKEpLfKHavBgnNkBDtdSxAE2") # API 密钥
    parser.add_argument('--max_workers', type=int, default=5) # 并行处理线程数
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    jsonl_path = os.path.join(args.output_dir, 'output.jsonl')
    json_path = os.path.join(args.output_dir, 'output.json')
    log_path = os.path.join(args.output_dir, 'process.log')

    # 设置日志记录
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),
            logging.StreamHandler() # 同时输出到控制台
        ]
    )

    # 读取输入数据
    input_data = read_json(args.input_path)
    processed_ids = set()

    # --- 逻辑不变：读取已处理的ID ---
    if os.path.exists(jsonl_path):
        try:
            processed_data = read_jsonl(jsonl_path)
            processed_ids = {item.get('id') for item in processed_data if 'id' in item}
            logging.info(f"已从 {jsonl_path} 加载 {len(processed_ids)} 个已处理的 ID。")
        except Exception as e:
            logging.error(f"读取或解析 {jsonl_path} 时出错: {e}。将尝试处理所有数据。")
            processed_ids = set() 

    # --- 逻辑不变：过滤掉已经处理过的数据项 ---
    items_to_process = []
    for item in input_data:
        item_id = item.get('id')
        if item_id is None:
            logging.warning(f"输入数据中发现一个没有 'id' 的项: {item}。将尝试处理。")
            items_to_process.append(item)
        elif item_id in processed_ids:
            continue
        else:
            items_to_process.append(item)
    
    if len(processed_ids) > 0:
        logging.info(f"总共 {len(input_data)} 项。已处理 {len(processed_ids)} 项。")
    logging.info(f"本次需要处理 {len(items_to_process)} 项。")

    # 使用线程池并行处理数据项
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(process_item, item, args.model, args.url, args.api_key, args.output_dir, jsonl_path): item for item in items_to_process}

        with tqdm(total=len(items_to_process), desc="处理中") as pbar:
            for future in as_completed(futures):
                item = futures[future]
                try:
                    future.result() 
                except Exception as e:
                    logging.error(f"ID: {item.get('id', 'unknown')} 在线程执行中发生未捕获的错误: {e}")
                pbar.update(1) 

    # 将结果从JSONL转换为JSON格式
    try:
        if os.path.exists(jsonl_path): 
            jsonl2json(jsonl_path, json_path)
            logging.info(f"处理完成，最终JSON结果已保存到: {json_path}")
        else:
            logging.info("没有生成 output.jsonl 文件，无需转换。")
    except Exception as e:
        logging.error(f"JSONL到JSON转换失败: {e}")

    print(f"所有处理已完成。日志文件位于: {log_path}")
