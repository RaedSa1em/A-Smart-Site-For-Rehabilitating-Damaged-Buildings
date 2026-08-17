import os
import time
import requests
import trimesh

# ==========================================
# 1. الإعدادات واختيار الصورة
# ==========================================
# يُقرأ المفتاح من متغيّر البيئة، حتى لا يبقى مكتوباً داخل الشيفرة
API_KEY = os.environ.get("TRIPO_API_KEY")

if not API_KEY:
    print("[ERROR] Environment variable TRIPO_API_KEY is not set.")
    print('        PowerShell: $env:TRIPO_API_KEY = "tsk_..."')
    print('        Bash:       export TRIPO_API_KEY="tsk_..."')
    exit()

# ضع اسم صورتك هنا
IMAGE_PATH = r"C:\Users\RAED\Desktop\image_generation\middle-size-building-damaged-free-png.png"

# ==========================================
# 2. تحديد اسم ملف النتيجة تلقائياً بحسب اسم الصورة
# ==========================================
if not os.path.exists(IMAGE_PATH):
    print(f"[ERROR] Image file '{IMAGE_PATH}' not found!")
    exit()

# استخراج اسم الصورة بدون الامتداد (مثلاً: damaged_building)
image_name_without_ext = os.path.splitext(os.path.basename(IMAGE_PATH))[0]

# إنشاء اسم النتيجة تلقائياً
OUTPUT_GLB = f"{image_name_without_ext}_3d.glb"

print(f"[INFO] Input Image : {IMAGE_PATH}")
print(f"[INFO] Output Model: {OUTPUT_GLB}")

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# ==========================================
# 3. رفع الصورة إلى السيرفر
# ==========================================
print("\n[1/4] Uploading image to Tripo AI...")
with open(IMAGE_PATH, "rb") as f:
    upload_res = requests.post(
        "https://api.tripo3d.ai/v2/openapi/upload",
        headers={"Authorization": f"Bearer {API_KEY}"},
        files={"file": f}
    ).json()

if upload_res.get("code") != 0:
    print(f"[ERROR Upload Failed]: {upload_res}")
    exit()

image_token = upload_res["data"]["image_token"]
print("[SUCCESS] Image uploaded successfully.")

# ==========================================
# 4. طلب إنشاء نموذج عالي الدقة والألوان
# ==========================================
print("[2/4] Generating High-Resolution 3D Model with PBR Textures...")
payload = {
    "type": "image_to_model",
    "model_version": "v2.5-20250123",
    "texture": True,
    "pbr": True,
    "file": {
        "type": os.path.splitext(IMAGE_PATH)[1].replace(".", "").lower(),
        "file_token": image_token
    }
}

task_res = requests.post(
    "https://api.tripo3d.ai/v2/openapi/task",
    headers=headers,
    json=payload
).json()

if task_res.get("code") != 0:
    print(f"[ERROR Task Creation Failed]: {task_res}")
    exit()

task_id = task_res["data"]["task_id"]

# ==========================================
# 5. الانتظار والتنزيل
# ==========================================
print("[3/4] Processing 3D Reconstruction...")
while True:
    status_res = requests.get(
        f"https://api.tripo3d.ai/v2/openapi/task/{task_id}",
        headers={"Authorization": f"Bearer {API_KEY}"}
    ).json()

    if status_res.get("code") != 0:
        print(f"[ERROR Status Check Failed]: {status_res}")
        exit()

    status = status_res["data"]["status"]
    if status == "success":
        output_data = status_res["data"]["output"]
        model_url = output_data.get("pbr_model") or output_data.get("model")
        print("[SUCCESS] 3D Model generated successfully!")
        break
    elif status == "failed":
        print(f"[ERROR Generation Failed]: {status_res}")
        exit()

    time.sleep(3)

# حفظ ملف الـ GLB بالاسم الديناميكي المخصص
model_bytes = requests.get(model_url).content
with open(OUTPUT_GLB, "wb") as f:
    f.write(model_bytes)

print(f"[SUCCESS] Model saved as: '{OUTPUT_GLB}'")

# ==========================================
# 6. العرض التفاعلي بـ Trimesh Viewer
# ==========================================
print("[4/4] Opening 3D Viewer...")
scene = trimesh.load(OUTPUT_GLB)
scene.show()