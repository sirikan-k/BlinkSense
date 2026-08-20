"""
merge_onnx.py
=============
แก้ปัญหา "Failed to load external data file ...onnx.data" บนเบราว์เซอร์

สาเหตุ: torch.onnx.export() บางเวอร์ชันของ PyTorch จะแยกน้ำหนักโมเดล
ไปเก็บในไฟล์ .onnx.data แยกต่างหาก แม้จะตั้ง use_external_data_format=False
ไว้แล้วก็ตาม ทำให้เบราว์เซอร์ (onnxruntime-web) โหลดไม่ได้ เพราะไม่รองรับ
การโหลดไฟล์ข้อมูลภายนอกแบบนี้ในสภาพแวดล้อม WASM

สคริปต์นี้จะรวมโครงสร้าง (.onnx) และน้ำหนัก (.onnx.data) กลับเป็นไฟล์
เดียวแบบสมบูรณ์ (self-contained) — ไม่ต้องเทรนโมเดลใหม่

วิธีใช้:
  1) วางสคริปต์นี้ไว้โฟลเดอร์เดียวกับ eye_model_2classes.onnx
     (และ eye_model_2classes.onnx.data ถ้ามี — เช็คในโฟลเดอร์ที่รันคำสั่ง
     train_eye_model.py ค้นหาไฟล์ที่ลงท้ายด้วย .data หรือ .onnx.data)
  2) ติดตั้ง: pip install onnx --break-system-packages
  3) รัน: python merge_onnx.py
  4) จะได้ไฟล์ eye_model_2classes_fixed.onnx ที่เป็นไฟล์เดียวจบ
     เอาไฟล์นี้ไปแทนที่ eye_model_2classes.onnx เดิมบน GitHub
     (ลบไฟล์ .onnx.data ตัวเก่าออกจาก repo ได้เลย ไม่ต้องใช้แล้ว)
"""

import onnx
import os

INPUT_MODEL = "eye_model_2classes.onnx"
OUTPUT_MODEL = "eye_model_2classes_fixed.onnx"

if not os.path.exists(INPUT_MODEL):
    print(f"❌ ไม่พบไฟล์ {INPUT_MODEL} ในโฟลเดอร์นี้ — ย้ายสคริปต์ไปวางในโฟลเดอร์ที่มีไฟล์โมเดลก่อน")
    exit()

# โหลดโมเดลพร้อมดึงข้อมูล external data (ถ้ามีไฟล์ .onnx.data อยู่ด้วย
# มันจะถูกรวมเข้ามาในหน่วยความจำตรงนี้อัตโนมัติ)
model = onnx.load(INPUT_MODEL, load_external_data=True)

# บันทึกใหม่แบบบังคับไม่ให้แยกไฟล์ข้อมูลออกอีก -> ได้ไฟล์เดียวจบแน่นอน
onnx.save_model(
    model,
    OUTPUT_MODEL,
    save_as_external_data=False,
)

size_kb = os.path.getsize(OUTPUT_MODEL) / 1024
print(f"✅ รวมไฟล์สำเร็จ: {OUTPUT_MODEL} ({size_kb:.1f} KB)")
print("   เอาไฟล์นี้ไปแทนที่ eye_model_2classes.onnx บน GitHub แล้ว push ใหม่ได้เลย")