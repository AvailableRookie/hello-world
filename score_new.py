import json
import pandas as pd


def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 将数据转换为pandas DataFrame
json_data = read_json("E:\ECode\huitu\eval\eval_output\qwen-image_images\ssim_eval_image_result.json")
df = pd.DataFrame(json_data)

# 统一处理“题目类型”大小写
df['题目类型'] = df['题目类型'].str.lower()

# # 过滤掉没有测试成功的样本
# df['output_geogebra_status'] = df['output_geogebra_status'].apply(lambda x: True if str(x).lower() == 'true' else False)
# df.loc[df['output_geogebra_status'] == False, ['VLM_eval_image_result', 'eval_text_result']] = 0

# 计算四个分数的均值
def calculate_scores(df):
    # 转换为float类型
    df['4o Image'] = df['VLM_eval_image_result'].astype(float)
    # df['4o Text'] = df['eval_text_result'].astype(float)
    df['LPIPS'] = df['lpips_eval_image_result'].astype(float)
    df['PSNR'] = df['psnr_eval_image_result'].astype(float)
    df['SSIM'] = df['ssim_eval_image_result'].astype(float)

    # 计算 pass@1 的数量
   # df['pass@1'] = df['output_geogebra_status'].apply(lambda x: 1 if x else 0)

    # 总分数的计算
    total_scores = {
        '4o Image Sum': df['4o Image'].sum(),
        'LPIPS Mean': df['LPIPS'].mean(),
        'PSNR Mean': df['PSNR'].mean(),
        'SSIM Mean': df['SSIM'].mean(),
        'Total Samples': len(df)
    }

    # 按不同字段进行分组计算
    group_by_fields = ['题目类型', '难度', '技能归类']
    group_results = {}

    for field in group_by_fields:
        grouped = df.groupby(field).agg(
            **{
                '4o Image Mean': ('4o Image', 'sum'),
                'LPIPS Mean': ('LPIPS', 'mean'),
                'PSNR Mean': ('PSNR', 'mean'),
                'SSIM Mean': ('SSIM', 'mean'),
                'Total Samples': ('id', 'count')
            }
        ).reset_index()
        
        group_results[field] = grouped
    
    # 处理技能归类的多选字段
    # 拆分技能归类字段，并统计每个技能出现的次数
    skills = [
        "基础几何作图 (Basic Constructions)", "圆的性质与作图 (Circle Properties & Constructions)",
        "几何变换 (Geometric Transformations)", "三角形性质与作图 (Triangle Properties & Constructions)",
        "几何定理应用 (Application of Theorems)", "多边形性质与作图 (Polygon Properties & Constructions)",
        "度量与比例 (Measurement & Ratios)", "轨迹作图 (Locus Construction)"
    ]
    
    # 为每个技能添加一列，表示是否包含该技能
    for skill in skills:
        df[skill] = df['技能归类'].apply(lambda x: 1 if skill in str(x) else 0)
    
    # 按技能统计
    skill_results = {}
    for skill in skills:
        skill_results[skill] = df.groupby(skill).agg(
            **{
                '4o Image Mean': ('4o Image', 'sum'),
                'LPIPS Mean': ('LPIPS', 'mean'),
                'PSNR Mean': ('PSNR', 'mean'),
                'SSIM Mean': ('SSIM', 'mean'),
                'Total Samples': ('id', 'count')
            }
        ).reset_index()

    return total_scores, group_results, skill_results

# 计算总分和分组分数
total_scores, group_results, skill_results = calculate_scores(df)

# 打印总分数
print("总分数:")
for key, value in total_scores.items():
    print(f"{key}: {value}")

# 打印按字段分组后的结果
for field, result in group_results.items():
    print(f"\n按 {field} 分组的结果:")
    print(result)

# 打印技能归类分组后的结果
for skill, result in skill_results.items():
    print(f"\n按 {skill} 分组的结果:")
    print(result)

# 保存到本地文件
output_file = "E:\ECode\huitu\eval\eval_output\qwen-image_images/output_results2.csv"

# 将总分数和分组结果保存为CSV文件
total_scores_df = pd.DataFrame([total_scores])
total_scores_df.to_csv(output_file, index=False, mode='w')

# 将分组结果逐个保存
for field, result in group_results.items():
    result.to_csv(output_file, index=False, mode='a', header=False)

# 将技能归类结果逐个保存
for skill, result in skill_results.items():
    result.to_csv(output_file, index=False, mode='a', header=False)

print(f"\n结果已保存到 {output_file}")
