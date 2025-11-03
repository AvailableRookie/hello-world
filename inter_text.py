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

def predict(question, model, url, api_key):

    client = OpenAI(
        base_url=url,
        api_key=api_key,
    )

    content = [
        {"type": "text", "text": question},
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

def process_item(item, model, url, api_key, jsonl_path):
    try:
        question = item['问题']
        prompt_question = f"""
Problem Title： 
{question}
"""
        prompt_sp =  r"""
Here is the translation of the provided prompt into English, keeping the content unchanged:
**According to the given drawing topic and image, create a drawing that meets the requirements, using GeoGebra language. If the problem requires the use of a ruler and compass construction, please retain the traces of ruler and compass construction. Finally, output the drawing steps and the corresponding image code for each step.**
### **Code Generation Rules**
#### **Robustness and Correctness of Code**
1. Strictly follow GeoGebra command syntax; the first letter of commands should be capitalized, and variable names should not use underscores.
2. When creating points, use the `Point({x, y})` format.
3. When creating vectors, use the `Vector((x, y))` or `Vector(<Start Point>, <End Point>)` format.
4. All auxiliary lines or auxiliary points added during the problem-solving process must be retained in the final drawing.
5. The origin of the coordinate system (0,0) is the reference for all point positioning.
6. Only one coordinate system is allowed in a single drawing.
7. For multiple tangents, use `t_1`, `t_2`, etc., to differentiate them.
8. The code block must not contain any form of comment text, and there must be no empty lines between code lines.
9. When drawing, no dynamic effects should appear; only static images should be created.

**Here is an example of the output format you can refer to:**
**Problem Title:** Given two circles of different radii that do not intersect, (c_1) (center (O_1), radius (r_1)) and (c_2) (center (O_2), radius (r_2)). Using only ruler and compass construction, draw the two external tangents to these two circles.
### **Step-by-Step Solution**
#### **Step 1: Construct the angle bisectors of ( \angle BAC ) and ( \angle ABC )**
**Construction:**
1. Construct the angle bisector (l_A) of ( \angle BAC ).
2. Construct the angle bisector (l_B) of ( \angle ABC ).
**Principle:**
The incenter of a triangle (the center of the incircle) is the intersection of its three angle bisectors. According to the angle bisector theorem, any point on an angle bisector is equidistant from the two sides of the angle. Therefore, the intersection of the two angle bisectors will be equidistant from the three sides of the triangle.
**GeoGebra Code:**

```geogebra
ShowAxes(false)
ShowGrid(false)
A = Point({1, 5})
SetCaption(A, "A")
SetColor(A, "#34495E")
SetPointStyle(A, 0)
SetPointSize(A, 5)
B = Point({0, 1})
SetCaption(B, "B")
SetColor(B, "#34495E")
SetPointStyle(B, 0)
SetPointSize(B, 5)
C = Point({7, 2})
SetCaption(C, "C")
SetColor(C, "#34495E")
SetPointStyle(C, 0)
SetPointSize(C, 5)
triangleABC = Polygon(A, B, C)
SetColor(triangleABC, "#3498DB")
SetLineThickness(triangleABC, 2)
SetFilling(triangleABC, 0.1)
bisectorA = AngleBisector(C, A, B)
SetLineStyle(bisectorA, 2)
SetColor(bisectorA, "#E74C3C")
bisectorB = AngleBisector(A, B, C)
SetLineStyle(bisectorB, 2)
SetColor(bisectorB, "#E74C3C")
ZoomIn(0, 0, 8, 6)
```

#### **Step 2: Find the Incenter (I)**
**Construction:**
1. Find the intersection of the angle bisectors (l_A) and (l_B), and call it (I).
**Principle:**
Point (I) is the incenter of triangle (ABC), i.e., the center of the incircle.
**GeoGebra Code:**

```geogebra
ShowAxes(false)
ShowGrid(false)
A = Point({1, 5})
SetCaption(A, "A")
SetColor(A, "#34495E")
SetPointStyle(A, 0)
SetPointSize(A, 5)
B = Point({0, 1})
SetCaption(B, "B")
SetColor(B, "#34495E")
SetPointStyle(B, 0)
SetPointSize(B, 5)
C = Point({7, 2})
SetCaption(C, "C")
SetColor(C, "#34495E")
SetPointStyle(C, 0)
SetPointSize(C, 5)
triangleABC = Polygon(A, B, C)
SetColor(triangleABC, "#3498DB")
SetLineThickness(triangleABC, 2)
SetFilling(triangleABC, 0.1)
bisectorA = AngleBisector(C, A, B)
SetLineStyle(bisectorA, 2)
SetColor(bisectorA, "#E74C3C")
bisectorB = AngleBisector(A, B, C)
SetLineStyle(bisectorB, 2)
SetColor(bisectorB, "#E74C3C")
I = Intersect(bisectorA, bisectorB)
SetCaption(I, "I")
SetColor(I, "#9B59B6")
SetPointStyle(I, 0)
SetPointSize(I, 6)
ZoomIn(0, 0, 8, 6)
```

#### **Step 3: Determine the Inradius**

**Construction:**
1. Draw the perpendicular line (l_{\perp}) from the incenter (I) to any side (e.g., (AB)).
2. The intersection of (l_{\perp}) with side (AB) is point (D).
3. The length of segment (ID) is the inradius.
**Principle:**
The necessary and sufficient condition for a circle to be tangent to a line is that the distance from the circle's center to the line equals the radius. This distance is defined by the length of the perpendicular segment.
**GeoGebra Code:**

```geogebra
ShowAxes(false)
ShowGrid(false)
A = Point({1, 5})
SetCaption(A, "A")
SetColor(A, "#34495E")
SetPointStyle(A, 0)
SetPointSize(A, 5)
B = Point({0, 1})
SetCaption(B, "B")
SetColor(B, "#34495E")
SetPointStyle(B, 0)
SetPointSize(B, 5)
C = Point({7, 2})
SetCaption(C, "C")
SetColor(C, "#34495E")
SetPointStyle(C, 0)
SetPointSize(C, 5)
triangleABC = Polygon(A, B, C)
SetColor(triangleABC, "#3498DB")
SetLineThickness(triangleABC, 2)
SetFilling(triangleABC, 0.1)
sideAB = Segment(A, B)
bisectorA = AngleBisector(C, A, B)
SetLineStyle(bisectorA, 2)
SetColor(bisectorA, "#E74C3C")
bisectorB = AngleBisector(A, B, C)
SetLineStyle(bisectorB, 2)
SetColor(bisectorB, "#E74C3C")
I = Intersect(bisectorA, bisectorB)
SetCaption(I, "I")
SetColor(I, "#9B59B6")
SetPointStyle(I, 0)
SetPointSize(I, 6)
perpLine = PerpendicularLine(I, sideAB)
SetLineStyle(perpLine, 2)
SetColor(perpLine, "#F1C40F")
D = Intersect(perpLine, sideAB)
SetCaption(D, "D")
SetColor(D, "#F39C12")
SetPointStyle(D, 2)
SetPointSize(D, 5)
radiusID = Segment(I, D)
SetColor(radiusID, "#F39C12")
SetLineThickness(radiusID, 2)
ZoomIn(0, 0, 8, 6)
```

#### **Step 4: Construct the Incircle**

**Construction:**
1. Construct a circle with center (I) and radius (ID).
**Principle:**
The circle constructed in this manner will be tangent to all three sides of the triangle, as the distance from its center to each side is equal to the radius.
**GeoGebra Code:**

```geogebra
ShowAxes(false)
ShowGrid(false)
A = Point({1, 5})
SetCaption(A, "A")
SetColor(A, "#34495E")
SetPointStyle(A, 0)
SetPointSize(A, 5)
B = Point({0, 1})
SetCaption(B, "B")
SetColor(B, "#34495E")
SetPointStyle(B, 0)
SetPointSize(B, 5)
C = Point({7, 2})
SetCaption(C, "C")
SetColor(C, "#34495E")
SetPointStyle(C, 0)
SetPointSize(C, 5)
triangleABC = Polygon(A, B, C)
SetColor(triangleABC, "#3498DB")
SetLineThickness(triangleABC, 3)
SetFilling(triangleABC, 0.1)
sideAB = Segment(A, B)
bisectorA = AngleBisector(C, A, B)
SetLineStyle(bisectorA, 2)
SetColor(bisectorA, "#BDC3C7")
bisectorB = AngleBisector(A, B, C)
SetLineStyle(bisectorB, 2)
SetColor(bisectorB, "#BDC3C7")
I = Intersect(bisectorA, bisectorB)
SetCaption(I, "I")
SetColor(I, "#9B59B6")
SetPointStyle(I, 0)
SetPointSize(I, 6)
perpLine = PerpendicularLine(I, sideAB)
SetLineStyle(perpLine, 2)
SetColor(perpLine, "#BDC3C7")
D = Intersect(perpLine, sideAB)
SetCaption(D, "D")
SetColor(D, "#F39C12")
SetPointStyle(D, 0)
SetPointSize(D, 4)
radiusID = Segment(I, D)
SetColor(radiusID, "#F39C12")
SetLineThickness(radiusID, 2)
incircle = Circle(I, D)
SetColor(incircle, "#E67E22")
SetLineThickness(incircle, 3)
SetFilling(incircle, 0.2)
ZoomIn(0, 0, 8, 6)
```
""" 
        prompt = prompt_question + prompt_sp

        result = predict(prompt, model, url, api_key)
        item['output'] = result
        append_jsonl(jsonl_path, item)
        return item
    except Exception as e:
        logging.error(f"ID: {item.get('id', 'unknown')} Error: {e}")
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default="E:\ECode\huitu\eval\GeoCraft_dataset.json")



    # 注意不同模型输出到不同的文件中  
    parser.add_argument('--output_dir', type=str, default="E:\ECode\huitu\eval\eval_output\gemini-images-text")
    # 修改模型名字，测哪个改哪个
    parser.add_argument('--model', type=str, default="gemini-2.5-flash-image")



    #固定
    parser.add_argument('--url', type=str, default="http://35.220.164.252:3888/v1/")
    parser.add_argument('--api_key', type=str, default="sk-KjCTTVFCwTsTEl0Z6vNPcPA2qKEpLfKHavBgnNkBDtdSxAE2")

    #并发数，越大越快 推荐20
    parser.add_argument('--max_workers', type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    jsonl_path = os.path.join(args.output_dir, 'output.jsonl')
    json_path = os.path.join(args.output_dir, 'output.json')
    log_path = os.path.join(args.output_dir, 'process.log')

    logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    input_data = read_json(args.input_path)
    processed_items = set()
    if os.path.exists(jsonl_path):
        processed_items = {json.dumps(item, sort_keys=True) for item in read_jsonl(jsonl_path)}

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = []
        for item in input_data:
            if json.dumps(item, sort_keys=True) in processed_items:
                continue
            futures.append(executor.submit(process_item, item, args.model, args.url, args.api_key, jsonl_path))

        for future in tqdm(as_completed(futures), total=len(futures)):
            future.result()

    jsonl2json(jsonl_path, json_path)
    print(f"Processing complete. Output saved to: {json_path}")
