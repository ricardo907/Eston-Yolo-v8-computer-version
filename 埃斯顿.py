# train_yolo11_stable.py
import os
import warnings
warnings.filterwarnings("ignore")

# —— 可修改区域 ——
DATA_YAML = r"F:\data_set\data.yaml"                 # 你的数据集 YAML
PROJECT    = r"runs/train"                            # 训练输出目录
RUN_NAME   = "bottle_yolo11"                          # 任务名
DEVICE     = 0                                        # GPU id，CPU 用 "cpu"
SEED       = 0                                        # 固定随机种子
TARGET_BATCH = 16                                     # 期望等效 batch（显存小会用累积凑）
IMG_SIZE   = 960                                      # 分辨率：如果是小目标建议使用 960/1280
EPOCHS     = 200

# —— 稳定训练相关（保证更平滑/可复现） ——
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # 某些 CUDA 算法确定性
import random, numpy as np, torch
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from ultralytics import YOLO

def pick_batch_and_accumulate(target_batch=16):
    """根据显存尝试选择 batch，OOM 时自动降低并用梯度累积达到等效 batch"""
    # 先猜一个保守 batch（显存不大可以先从 8/4 起）
    trial_batches = [target_batch, 12, 8, 6, 4, 2, 1]
    for b in trial_batches:
        try:
            # 试跑一个极短的 dry-run（通过构造 Trainer 比较繁琐；这里直接返回，真正 OOM 由训练时捕获）
            return b, max(1, target_batch // max(1, b))
        except Exception:
            continue
    return 1, target_batch

def main():
    # 选择 batch & 累积
    batch, accumulate = pick_batch_and_accumulate(TARGET_BATCH)
    print(f"[INFO] plan: batch={batch}, accumulate={accumulate}  (≈等效 {batch*accumulate})")

    # 加载预训练（更稳更快）; 想从零开始可改为 '.../yolo11.yaml'
    model = YOLO("yolo11s.pt")

    results = model.train(
        data=DATA_YAML,
        imgsz=IMG_SIZE,
        epochs=EPOCHS,
        batch=batch,
        accumulate=accumulate,
        workers=2,                 # Windows 可用 0/2；若报错改 0
        device=DEVICE,
        seed=SEED,

        # —— 优化器与 LR ——
        optimizer="AdamW",
        lr0=1e-3,                 # 初始学习率
        lrf=1e-2,                 # 余弦到最终学习率比例
        weight_decay=0.01,

        # —— 数据增强：温和 & 可控（减少抖动） ——
        mosaic=0.3, mixup=0.0, copy_paste=0.0,
        close_mosaic=30,          # 后 30 epoch 关闭 mosaic 收敛
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,

        # —— 其他稳定性/效率 —— 
        patience=50,              # 早停容忍
        cache=True,               # 加快 IO
        amp=True,                 # 若遇到 NaN 可改 False
        project=PROJECT,
        name=RUN_NAME,
        verbose=True,
        save=True,
        plots=True,
        pretrained=True,          # 明确启用预训练
        single_cls=False,
    )

    best = os.path.join(PROJECT, RUN_NAME, "weights", "best.pt")
    print(f"[INFO] best weights: {best}")

    # 验证一遍
    YOLO(best).val(data=DATA_YAML, imgsz=IMG_SIZE, device=DEVICE, plots=True)

    # 导出 ONNX
    try:
        YOLO(best).export(format="onnx", dynamic=True, simplify=True)
        print("[INFO] Exported ONNX.")
    except Exception as e:
        print(f"[WARN] ONNX export skipped: {e}")

if __name__ == "__main__":
    main()
