import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ==========================================
# 0. Reproducibility — fix seed ทุกจุด
# ==========================================
# สำคัญสำหรับงานประกวด: ถ้ากรรมการถามว่า "รันซ้ำได้ผลเหมือนเดิมไหม"
# ต้องตอบได้ว่าใช่ เพราะ fix seed ไว้แล้ว
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ==========================================
# 1. ตั้งค่าพื้นฐานและที่อยู่โฟลเดอร์รูปภาพ
# ==========================================
# เปลี่ยน path ตรงนี้ให้ตรงกับโฟลเดอร์ Dataset ของคุณ
# สมมติว่าโฟลเดอร์ 1_Dataset อยู่ระดับเดียวกับ 2_Model_Training
train_dir = r"..\1_Dataset\train"

# ถ้ายังไม่มีโฟลเดอร์ val ให้ดึงจาก train มาทดสอบก่อน
val_dir = r"..\1_Dataset\val"
has_real_val = os.path.exists(val_dir)
if not has_real_val:
    val_dir = train_dir
    print("⚠️  ไม่พบโฟลเดอร์ val แยก — ใช้ train ทดสอบไปพลางก่อน "
          "(ตัวเลข accuracy ที่ได้จะดูดีเกินจริง เพราะโมเดลเคยเห็นข้อมูลนี้มาแล้ว "
          "ควรแยกชุด val จริงก่อนสรุปผลในรายงาน)")

num_classes = 2 # ปรับเหลือ 2 คลาส (awake, sleepy)
batch_size = 16
epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"กำลังใช้ Device: {device} ในการเทรนโมเดล {num_classes} คลาส")

# ==========================================
# 2. เตรียมข้อมูลรูปภาพ
# ==========================================
transform_train = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

transform_val = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

try:
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform_train)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=transform_val)
except FileNotFoundError:
    print(f"\n❌ Error: หาโฟลเดอร์รูปภาพไม่เจอ กรุณาตรวจสอบว่ามีรูปใน {train_dir}")
    exit()

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
print(f"✅ คลาสที่ระบบค้นพบ: {train_dataset.classes}  (index 0={train_dataset.classes[0]}, 1={train_dataset.classes[1]})")

# เช็คสัดส่วนข้อมูลแต่ละคลาส — ถ้าคลาสใดคลาสหนึ่งมีน้อยกว่าอีกคลาสมาก
# โมเดลจะเอนเอียงไปทายคลาสที่มีเยอะกว่า ควรถ่ายรูปเพิ่มให้จำนวนใกล้เคียงกัน
counts = np.bincount(train_dataset.targets)
for cls_name, cnt in zip(train_dataset.classes, counts):
    print(f"   - {cls_name}: {cnt} ภาพ")
if max(counts) / max(min(counts), 1) > 1.5:
    print("⚠️  ข้อมูลแต่ละคลาสไม่สมดุลกันมาก (ต่างกันเกิน 1.5 เท่า) "
          "ควรเก็บข้อมูลเพิ่มให้จำนวนใกล้เคียงกันเพื่อไม่ให้โมเดล bias")

# ==========================================
# 3. สร้างโมเดล CNN แบบ 2 คลาส
# ==========================================
class EyeClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(EyeClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes) # ส่งออก 2 คลาส
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

model = EyeClassifier(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def evaluate(model, loader):
    """รันโมเดลบนชุด validation แล้วคืนค่า loss เฉลี่ย, accuracy,
    และ (labels จริง, labels ที่ทาย) ไว้ทำ confusion matrix"""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_labels, all_preds = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    return total_loss / len(loader), correct / total, all_labels, all_preds


# ==========================================
# 4. เริ่มการเทรน (พร้อม validate ทุก epoch + เก็บโมเดลที่ดีที่สุด)
# ==========================================
print("\n🚀 เริ่มการเทรนโมเดล...")
best_val_acc = 0.0
best_state = None

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    val_loss, val_acc, _, _ = evaluate(model, val_loader)
    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} "
          f"| Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

    if val_acc >= best_val_acc:
        best_val_acc = val_acc
        best_state = {k: v.clone() for k, v in model.state_dict().items()}

print(f"\n🎉 เทรนเสร็จสิ้น! Best Val Accuracy = {best_val_acc*100:.2f}%")

# โหลด weight ที่ดีที่สุดกลับมาก่อนบันทึก/export
# (กัน epoch สุดท้ายบังเอิญแย่กว่ากลางๆ)
if best_state is not None:
    model.load_state_dict(best_state)

torch.save(model.state_dict(), "eye_model_best.pth")
print("✅ บันทึก checkpoint ไว้ที่ eye_model_best.pth (ไว้เทรนต่อ/ทดสอบใน Python ภายหลัง)")

# ==========================================
# 4.1 Confusion Matrix + Classification Report — เอาไปใส่รายงานโครงงาน
# ==========================================
try:
    from sklearn.metrics import confusion_matrix, classification_report
    import matplotlib
    matplotlib.use("Agg")  # ไม่ต้องเปิดหน้าต่างแสดงผล แค่บันทึกเป็นไฟล์ภาพ
    import matplotlib.pyplot as plt
    import numpy as np

    _, _, y_true, y_pred = evaluate(model, val_loader)
    class_names = train_dataset.classes  # เช่น ['awake', 'sleepy']
    cm = confusion_matrix(y_true, y_pred)

    print("\n📊 Confusion Matrix (แถว=จริง, คอลัมน์=ทาย) "
          f"ลำดับคลาส {class_names}:")
    print(cm)
    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # ---------- วาดเป็นภาพ PNG พร้อมใช้ในรายงาน (ภาพที่ 5) ----------
    NAVY = "#16324A"
    TEAL = "#1F8A83"

    fig, ax = plt.subplots(figsize=(5.2, 4.6), dpi=220)
    im = ax.imshow(cm, cmap="Blues")

    # ตัวเลขในแต่ละช่อง (สีขาวถ้าพื้นเข้ม, สีเข้มถ้าพื้นอ่อน อ่านง่ายทั้งคู่)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                     fontsize=15, fontweight="bold",
                     color="white" if cm[i, j] > thresh else NAVY)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=11)
    ax.set_yticklabels(class_names, fontsize=11)
    ax.set_xlabel("ผลที่โมเดลทาย (Predicted)", fontsize=11, labelpad=8)
    ax.set_ylabel("คลาสจริง (True Label)", fontsize=11, labelpad=8)
    ax.set_title(f"Confusion Matrix (Val Accuracy = {best_val_acc*100:.2f}%)",
                 fontsize=12.5, fontweight="bold", color=NAVY, pad=12)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close()
    print("✅ บันทึกภาพ Confusion Matrix ไว้ที่ confusion_matrix.png "
          "(เอาไปใส่แทนภาพที่ 5 ในรายงานได้เลย)")

except ImportError as e:
    print(f"\n(ยังขาดไลบรารี: {e}. ติดตั้งด้วย: "
          "pip install scikit-learn matplotlib)")

# ==========================================
# 5. แปลงและส่งออกโมเดลเป็น ONNX สำหรับใช้งานบนมือถือ
# ==========================================
model.eval()
dummy_input = torch.randn(1, 3, 64, 64).to(device)
onnx_path = "eye_model_2classes.onnx"

torch.onnx.export(
    model, 
    dummy_input, 
    onnx_path, 
    export_params=True, 
    opset_version=11, 
    do_constant_folding=True, 
    input_names=['input'], 
    output_names=['output'], 
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    # หมายเหตุ: เอา use_external_data_format ออกแล้ว เพราะ PyTorch เวอร์ชันใหม่
    # ลบพารามิเตอร์นี้ทิ้งไปแล้ว (เรียกจะ error ทันที) ไม่ต้องกังวลเรื่องไฟล์
    # แยก .onnx.data เพราะขั้นตอนถัดไปข้างล่างจะบังคับรวมเป็นไฟล์เดียวให้เองอยู่แล้ว
)

# ==========================================
# 5.1 กันเหนียว: บังคับรวมเป็นไฟล์เดียวอีกรอบ
# ==========================================
# PyTorch บางเวอร์ชันจะเมิน use_external_data_format=False และแยกน้ำหนัก
# โมเดลไปเป็นไฟล์ .onnx.data อยู่ดี ซึ่งเบราว์เซอร์ (onnxruntime-web) โหลด
# ไม่ได้ ขั้นตอนนี้จึงโหลดโมเดลกลับมาพร้อมข้อมูลภายนอก (ถ้ามี) แล้วบันทึก
# ทับเป็นไฟล์เดียวแบบสมบูรณ์อีกครั้ง เพื่อการันตีว่าไฟล์ที่ได้ใช้กับเว็บได้แน่นอน
import onnx as _onnx
_model = _onnx.load(onnx_path, load_external_data=True)
_onnx.save_model(_model, onnx_path, save_as_external_data=False)

# ลบไฟล์ .onnx.data เก่าทิ้งถ้ามันถูกสร้างขึ้นมา (ไม่ต้องใช้แล้ว)
stray_data_file = onnx_path + ".data"
if os.path.exists(stray_data_file):
    os.remove(stray_data_file)
    print(f"🧹 ลบไฟล์ {stray_data_file} ที่ไม่จำเป็นแล้วออก")

print(f"✅ บันทึกโมเดล ONNX ไฟล์เดียวสำเร็จ! ({onnx_path})")