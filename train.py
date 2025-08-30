import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB6
from tensorflow.keras.models import Model, save_model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler, Callback
from tensorflow.keras.regularizers import l2
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import random
import math

# 启用混合精度训练以加速（GPU兼容）
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# 设置随机种子确保可复现性
tf.random.set_seed(42)
np.random.seed(42)

# 自定义回调用于记录学习率
class LRTracker(Callback):
    def __init__(self):
        super(LRTracker, self).__init__()
        self.lr_history = []
    
    def on_epoch_end(self, epoch, logs=None):
        current_lr = tf.keras.backend.get_value(self.model.optimizer.lr)
        self.lr_history.append(current_lr)
        logs = logs or {}
        logs['lr'] = current_lr  # 确保学习率被记录到history中

# 参数配置
data_dir = '/kaggle/input/animals/Animal/Animal'  # 数据集路径
num_classes = 100
img_size = (456, 456)  # B6模型推荐尺寸
batch_size = 12  # 增加批次大小以提高GPU利用率
epochs = 20  # 减少轮数以适应12小时限制
patience = 5  # 早停等待轮数
l2_reg = 1e-4  # L2正则化系数
dropout_rate = 0.3  # Dropout比率
initial_lr = 1e-4  # 初始学习率

# 创建数据增强
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.25,
    height_shift_range=0.25,
    shear_range=0.2,
    zoom_range=0.3,
    brightness_range=[0.7, 1.3],
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # 80%训练，20%验证
)

# 训练集生成器
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42
)

# 验证集生成器
val_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False,  # 验证集不洗牌，确保一致性
    seed=42
)

# 获取类别名称
class_names = list(train_generator.class_indices.keys())
print(f"训练样本数: {train_generator.samples}, 验证样本数: {val_generator.samples}")

# 加速学习率调度
def accelerated_lr_schedule(epoch):
    """更激进的学习率调度以快速收敛"""
    warmup_epochs = 3
    decay_start = 10
    
    if epoch < warmup_epochs:
        # 学习率预热
        return initial_lr * (epoch + 1) / warmup_epochs
    elif epoch < decay_start:
        # 保持高学习率
        return initial_lr
    else:
        # 指数衰减
        return initial_lr * math.exp(0.1 * (decay_start - epoch))

# 构建模型
def create_model():
    base_model = EfficientNetB6(
        include_top=False,
        weights='imagenet',
        input_shape=(img_size[0], img_size[1], 3),
        pooling=None
    )
    
    # 冻结底层，微调上层
    base_model.trainable = True
    for layer in base_model.layers[:200]:
        layer.trainable = False
    
    # 添加自定义顶层
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu', kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(512, activation='relu', kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate/2)(x)
    
    # 修复：使用Lambda层进行类型转换
    x = Lambda(lambda t: tf.cast(t, tf.float32))(x)
    predictions = Dense(num_classes, activation='softmax', dtype=tf.float32)(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(
        optimizer=Adam(learning_rate=initial_lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

model = create_model()
model.summary()

# 创建学习率跟踪器
lr_tracker = LRTracker()

# 回调函数
callbacks = [
    EarlyStopping(
        monitor='val_loss', 
        patience=patience, 
        verbose=1, 
        restore_best_weights=True,
        min_delta=0.001
    ),
    ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=2, 
        min_lr=1e-6, 
        verbose=1
    ),
    ModelCheckpoint(
        '/kaggle/working/best_model.keras', 
        monitor='val_loss', 
        save_best_only=True, 
        verbose=1
    ),
    LearningRateScheduler(accelerated_lr_schedule, verbose=1),
    lr_tracker  # 添加学习率跟踪器
]

# 训练模型
print("开始训练模型...")
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // batch_size,
    epochs=epochs,
    callbacks=callbacks,
    verbose=1
)

# 确保学习率被记录到历史中
if 'lr' not in history.history:
    history.history['lr'] = lr_tracker.lr_history

# 保存训练历史
history_df = pd.DataFrame(history.history)
history_df.to_excel('/kaggle/working/training_history.xlsx', index=False)

# 绘制训练曲线
plt.figure(figsize=(12, 10))

# 损失曲线
plt.subplot(2, 2, 1)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Training & Validation Loss')
plt.legend()

# 准确率曲线
plt.subplot(2, 2, 2)
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.title('Training & Validation Accuracy')
plt.legend()

# 学习率曲线 - 现在使用记录的学习率数据
plt.subplot(2, 2, 3)
plt.plot(history.history['lr'], label='learning rate')
plt.title('Learning Rate Schedule')
plt.yscale('log')
plt.legend()

# 训练/验证样本数量
plt.subplot(2, 2, 4)
plt.bar(['train', 'val'], [train_generator.samples, val_generator.samples], color=['blue', 'orange'])
plt.title('Dataset Distribution')

plt.tight_layout()
plt.savefig('/kaggle/working/training_metrics.png', dpi=150)
plt.close()

# 加载最佳模型
model = load_model('/kaggle/working/best_model.keras')

# 在验证集上评估
print("\n在验证集上评估模型...")
val_loss, val_acc = model.evaluate(val_generator)
print(f"验证集准确率: {val_acc:.4f}")

# 生成预测结果
print("\n生成预测结果...")
y_true = val_generator.classes
y_pred = model.predict(val_generator, verbose=1).argmax(axis=1)

# 计算混淆矩阵
cm = confusion_matrix(y_true, y_pred)

# 分析每个类别的准确率和错误情况
class_report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
class_metrics = []

for i, class_name in enumerate(class_names):
    # 类别准确率
    precision = class_report[class_name]['precision']
    recall = class_report[class_name]['recall']
    f1 = class_report[class_name]['f1-score']
    support = class_report[class_name]['support']
    
    # 主要错误类别
    class_errors = cm[i].copy()
    class_errors[i] = 0  # 忽略正确预测
    if np.sum(class_errors) > 0:
        main_error_idx = np.argmax(class_errors)
        main_error_class = class_names[main_error_idx]
        error_count = class_errors[main_error_idx]
        error_percent = error_count / np.sum(class_errors) * 100
    else:
        main_error_class = "无错误"
        error_count = 0
        error_percent = 0
    
    class_metrics.append({
        '类别': class_name,
        '精确率': precision,
        '召回率': recall,
        'F1分数': f1,
        '样本数': support,
        '主要误判类别': main_error_class,
        '误判数量': error_count,
        '误判占比(%)': error_percent
    })

# 保存类别准确率结果
metrics_df = pd.DataFrame(class_metrics)
metrics_df.to_excel('/kaggle/working/class_accuracy_report.xlsx', index=False)

# 绘制混淆矩阵热力图（简化版，只显示前20类）
plt.figure(figsize=(15, 13))
sns.heatmap(cm[:20, :20], annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names[:20], yticklabels=class_names[:20])
plt.title('Confusion Matrix (Top 20 Classes)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks(rotation=45, fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.savefig('/kaggle/working/confusion_matrix_top20.png', dpi=150)
plt.close()

# 绘制各类别F1分数分布
plt.figure(figsize=(15, 8))
metrics_df_sorted = metrics_df.sort_values(by='F1分数', ascending=False)
plt.bar(metrics_df_sorted['类别'], metrics_df_sorted['F1分数'], color='skyblue')
plt.xticks(rotation=90, fontsize=8)
plt.axhline(y=metrics_df_sorted['F1分数'].mean(), color='r', linestyle='--', label='Average F1 Score')
plt.title('F1 Score per Class')
plt.ylabel('F1 Score')
plt.legend()
plt.tight_layout()
plt.savefig('/kaggle/working/class_f1_scores.png', dpi=150)
plt.close()

# 随机抽取12个验证样本可视化（节省空间）
print("\n随机抽取验证样本可视化...")
sample_indices = random.sample(range(len(val_generator.filepaths)), 12)
sample_images = []
sample_true_labels = []
sample_pred_labels = []
sample_probs = []

for idx in sample_indices:
    img_path = val_generator.filepaths[idx]
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=img_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    sample_images.append(img_array)
    sample_true_labels.append(class_names[y_true[idx]])
    
    # 获取预测概率
    pred_probs = model.predict(np.expand_dims(img_array, axis=0), verbose=0)[0]
    pred_idx = np.argmax(pred_probs)
    sample_pred_labels.append(class_names[pred_idx])
    sample_probs.append(pred_probs[pred_idx])

# 可视化结果
plt.figure(figsize=(15, 12))
for i in range(12):
    plt.subplot(3, 4, i+1)
    plt.imshow(sample_images[i])
    color = 'green' if sample_true_labels[i] == sample_pred_labels[i] else 'red'
    title = f"True: {sample_true_labels[i]}\nPred: {sample_pred_labels[i]}\nProb: {sample_probs[i]:.2f}"
    plt.title(title, fontsize=10, color=color)
    plt.axis('off')
plt.tight_layout()
plt.savefig('/kaggle/working/sample_predictions.png', dpi=150)
plt.close()

# 计算实际训练时间
total_minutes = len(history.history['loss']) * 34
hours = int(total_minutes // 60)
minutes = int(total_minutes % 60)

print("="*50)
print("所有操作已完成！")
print(f"最佳模型已保存为: best_model.keras")
print(f"训练历史已保存为: training_history.xlsx")
print(f"类别准确率报告已保存为: class_accuracy_report.xlsx")
print(f"验证集准确率: {val_acc:.4f}")
print(f"总训练时间: {hours}小时 {minutes}分钟")
print("="*50)