import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import pandas as pd

# 禁用GPU（确保使用CPU）
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 配置参数
model_path = './output/model/best_model.keras'  # 替换为你的模型路径
image_dir = 'test'  # 替换为你的图片目录路径
output_csv = 'predictions.csv'  # 预测结果保存路径
img_size = (456, 456)  # 与训练时相同的尺寸
class_names_path = 'class.txt'  # 类别名称文件路径（可选）

# 定义Lambda层使用的函数
def cast_to_float32(x):
    return tf.cast(x, tf.float32)

# 加载模型
print("正在加载模型...")
try:
    custom_objects = {'cast_to_float32': cast_to_float32}
    model = load_model(model_path, compile=False, custom_objects=custom_objects)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("模型加载成功！")
    model.summary()
except Exception as e:
    print(f"模型加载失败: {str(e)}")
    exit(1)

# 加载类别名称（如果可用）
if os.path.exists(class_names_path):
    with open(class_names_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    print(f"已加载 {len(class_names)} 个类别名称")
else:
    class_names = None
    print("未找到类别名称文件，将使用数字标签")

# 支持的图片格式
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']

# 获取图片文件列表
image_files = []
for root, _, files in os.walk(image_dir):
    for file in files:
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(root, file))

if not image_files:
    print(f"在目录 {image_dir} 中未找到图片文件")
    exit(1)

print(f"找到 {len(image_files)} 张待预测图片")

# 创建结果列表
results = []

# 批量处理图片
for i, img_path in enumerate(image_files):
    # 加载和预处理图片
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # 添加批次维度
    
    # 预测
    predictions = model.predict(img_array, verbose=0)[0]
    top_idx = np.argmax(predictions)
    confidence = predictions[top_idx]
    
    # 获取类别名称
    if class_names and len(class_names) > top_idx:
        predicted_class = class_names[top_idx]
    else:
        predicted_class = str(top_idx)
    
    # 获取top3预测结果
    top3_indices = np.argsort(predictions)[::-1][:3]
    top3_classes = []
    top3_confidences = []
    
    for idx in top3_indices:
        if class_names and len(class_names) > idx:
            top3_classes.append(class_names[idx])
        else:
            top3_classes.append(str(idx))
        top3_confidences.append(float(predictions[idx]))
    
    # 添加到结果
    result = {
        'file_path': img_path,
        'predicted_class': predicted_class,
        'confidence': confidence,
        'top1_class': top3_classes[0],
        'top1_confidence': top3_confidences[0],
        'top2_class': top3_classes[1],
        'top2_confidence': top3_confidences[1],
        'top3_class': top3_classes[2],
        'top3_confidence': top3_confidences[2]
    }
    
    results.append(result)
    
    # 打印进度
    if (i + 1) % 10 == 0 or (i + 1) == len(image_files):
        print(f"已处理 {i+1}/{len(image_files)} 张图片")

# 保存结果到CSV
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False, encoding='utf-8-sig')  # 使用utf-8-sig支持中文

print(f"\n预测完成！结果已保存至: {output_csv}")
print("="*50)
print("结果预览:")
print(df[['file_path', 'predicted_class', 'confidence']].head())