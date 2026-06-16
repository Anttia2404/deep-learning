import matplotlib.pyplot as plt
import json

# ============================================================
# HƯỚNG DẪN: Copy các dòng JSON kết quả in ra ở Cell 10 trên Kaggle
# và dán vào chuỗi `log_text` dưới đây.
# ============================================================
log_text = """
{"epoch": 1, "train_loss": 0.4512, "train_acc": 78.20, "val_loss": 0.3210, "val_acc": 85.40, "lr": 0.0002}
{"epoch": 2, "train_loss": 0.2845, "train_acc": 89.10, "val_loss": 0.1842, "val_acc": 93.10, "lr": 0.0002}
{"epoch": 3, "train_loss": 0.1766, "train_acc": 93.61, "val_loss": 0.0629, "val_acc": 99.28, "lr": 0.0002}
"""

def plot_from_logs(text_data, title="Training History"):
    epochs = []
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    
    for line in text_data.strip().split('\n'):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
            epochs.append(data["epoch"])
            train_loss.append(data["train_loss"])
            val_loss.append(data["val_loss"])
            train_acc.append(data["train_acc"])
            val_acc.append(data["val_acc"])
        except Exception as e:
            print(f"Bỏ qua dòng lỗi: {line} ({e})")
            
    if not epochs:
        print("Không tìm thấy dữ liệu hợp lệ!")
        return

    # Khởi tạo đồ thị
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Đồ thị Loss (Hàm mất mát)
    ax1.plot(epochs, train_loss, 'o-', color='#3b82f6', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_loss, 'o-', color='#ef4444', label='Val Loss', linewidth=2)
    ax1.set_title("Training & Validation Loss", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Loss", fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()
    
    # 2. Đồ thị Accuracy (Độ chính xác)
    ax2.plot(epochs, train_acc, 'o-', color='#10b981', label='Train Acc', linewidth=2)
    ax2.plot(epochs, val_acc, 'o-', color='#f59e0b', label='Val Acc', linewidth=2)
    ax2.set_title("Training & Validation Accuracy", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Accuracy (%)", fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Lưu đồ thị ra file ảnh
    output_path = "training_history.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Đã vẽ đồ thị thành công! Ảnh lưu tại: {output_path}")
    plt.show()

if __name__ == "__main__":
    plot_from_logs(log_text, "Face2Face Detector Training History")
