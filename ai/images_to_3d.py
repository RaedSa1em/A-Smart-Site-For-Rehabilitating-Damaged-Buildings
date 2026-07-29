import cv2
import numpy as np
import open3d as o3d
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


# 1. التجهيز والتحقق من جهاز المعالجة
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using processing device: {device}")


# 2. تحميل نموذج Depth Anything V2 الرسمي
print("[INFO] Loading Depth Anything V2 model from Hugging Face...")
model_id = "depth-anything/Depth-Anything-V2-Small-hf"
image_processor = AutoImageProcessor.from_pretrained(model_id)
model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device)


# 3. قراءة وتحضير صورة المبنى المتضرر
image_path = "C:\\Users\\RAED\\Desktop\\1\\inputs\\middle-size-building-damaged-free-png.png"  # تأكد من وجود الصورة بنفس الاسم في المجلد
try:
    raw_image = Image.open(image_path).convert("RGB")
except FileNotFoundError:
    print(f"[ERROR] Could not find the image '{image_path}'. Please check the path.")
    exit()

w, h = raw_image.size
print(f"[INFO] Image loaded successfully ({w}x{h}).")


# 4. استخراج ومعالجة خريطة العمق (Depth Map)
print("[INFO] Inferring depth map using AI...")
inputs = image_processor(images=raw_image, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth

# إعادة تحجيم خريطة العمق لتطابق مقاس الصورة الأصلية
depth = torch.nn.functional.interpolate(
    predicted_depth.unsqueeze(1),
    size=(h, w),
    mode="bicubic",
    align_corners=False,
)
depth_np = depth.squeeze().cpu().numpy()

# تطبيع القيم (Normalization) لبيانات Float32 متوافقة مع Open3D
depth_normalized = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min())
depth_float32 = depth_normalized.astype(np.float32)


# 5. بناء وتوليد المجسم ثلاثي الأبعاد
print("[INFO] Constructing 3D Point Cloud...")

color_raw = o3d.geometry.Image(np.array(raw_image))
depth_raw = o3d.geometry.Image(depth_float32)

# دمج الألوان والعمق
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color_raw,
    depth_raw,
    depth_scale=1.0,
    depth_trunc=1000.0,
    convert_rgb_to_intensity=False
)

# إعدادات الكاميرا الافتراضية
fx, fy = float(w), float(h)
cx, cy = float(w) / 2.0, float(h) / 2.0
pinhole_camera = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)

# إنشاء سحابة النقاط الأولية
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, pinhole_camera)


# 6. تصفية وعزل الخلفية البيضاء الشاذة برمجياً
print("[INFO] Cleaning background and isolated artifacts...")

points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)

# قناع لعزل البكسلات البيضاء النقية الخاصة بالخلفية المعزولة
background_mask = (colors[:, 0] > 0.92) & (colors[:, 1] > 0.92) & (colors[:, 2] > 0.92)
building_mask = ~background_mask

# تطبيق القناع واستخراج المبنى فقط
pcd_clean = o3d.geometry.PointCloud()
pcd_clean.points = o3d.utility.Vector3dVector(points[building_mask])
pcd_clean.colors = o3d.utility.Vector3dVector(colors[building_mask])

# تنظيف النقاط المتناثرة والشواذ العشوائية (Statistical Outlier Removal)
cl, ind = pcd_clean.remove_statistical_outlier(nb_neighbors=25, std_ratio=1.8)
pcd_final = pcd_clean.select_by_index(ind)

# تصحيح اتجاه المجسم في الفراغ
pcd_final.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])


# 7. الحفظ والتصدير والعرض
output_path = "cleaned_building_3d.ply"
o3d.io.write_point_cloud(output_path, pcd_final)
print(f"[SUCCESS] Cleaned 3D model exported successfully to: {output_path}")

print("[INFO] Opening interactive 3D window... (Close window to exit)")
o3d.visualization.draw_geometries([pcd_final])