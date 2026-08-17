import cv2
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel

# دالة قراءة الصور التي تدعم اللغة العربية والترميز في المسار
def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    try:
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
    except Exception as e:
        print(f"خطأ في قراءة الصورة: {e}")
        return None

# ==========================================
# الخطوة 1: إنشاء القناع (توسيع القناع لأعلى لتغطية السطح)
# ==========================================
def create_damage_mask(img_path, target_size=(512, 512)):
    img = imread_unicode(img_path)
    if img is None:
        raise FileNotFoundError(f"لم يتم العثور على الصورة في المسار: {img_path}")
        
    img_resized = cv2.resize(img, target_size)
    h, w, _ = img_resized.shape
    
    mask = np.zeros((h, w), dtype=np.uint8)
    # رفع القناع ليصل لقمة المبنى (من 0.02 إلى 0.85) لتغطية الانحدار بالكامل
    mask[int(h*0.02):int(h*0.85), int(w*0.20):int(w*0.80)] = 255
    
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    _, text_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    final_mask = cv2.bitwise_and(mask, mask, mask=text_mask)
    
    _, buf = cv2.imencode('.png', final_mask)
    buf.tofile("generated_mask_512.png")
    return "generated_mask_512.png"

# ==========================================
# الخطوة 2: استخراج حواف ControlNet وإصلاح خط السطح
# ==========================================
def extract_control_edges_reconstructed(img_path, mask_path, target_size=(512, 512)):
    img = imread_unicode(img_path)
    img_resized = cv2.resize(img, target_size)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # 1. استخراج الحواف الأصلية
    edges = cv2.Canny(gray, 100, 200)
    
    # 2. تفريغ منطقة الدمار داخل القناع
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask_resized = cv2.resize(mask, target_size)
    edges[mask_resized > 128] = 0
    
    # 3. رسم خط أفق مستقيم للسطح لإجبار النموذج على إكمال السطح أفقياً
    # يتم رسم خط مستقيم عند مستوى أعلى النقطة السليمة للسطح (تقريباً الارتفاع 10% من أعلى)
    roof_level = int(target_size[1] * 0.10)
    cv2.line(edges, (int(target_size[0]*0.20), roof_level), (int(target_size[0]*0.80), roof_level), 255, 2)

    edges_3ch = np.stack([edges] * 3, axis=-1)
    _, buf = cv2.imencode('.png', edges_3ch)
    buf.tofile("control_edges_512.png")
    
    return Image.fromarray(edges_3ch)

# ==========================================
# الخطوة 3: التوليد الموجه للسطح والواجهة
# ==========================================
def run_local_inpainting_low_vram(img_path, mask_path):
    target_size = (512, 512)
    
    init_image = Image.open(img_path).convert("RGB").resize(target_size)
    mask_image = Image.open(mask_path).convert("RGB").resize(target_size)
    control_image = extract_control_edges_reconstructed(img_path, mask_path, target_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny",
        torch_dtype=dtype
    )

    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        controlnet=controlnet,
        torch_dtype=dtype
    )

    pipe.to(device)
    pipe.enable_attention_slicing("max")
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()

    # نص موجه يشدد على السطح الأفقي المستوي والواجهة الخرسانية المكتملة
    prompt = "straight flat horizontal roof, complete modern concrete building facade, fully restored structure, clean uniform grid windows, intact flat top roof, photorealistic architecture, 8k photo"
    negative_prompt = "sloped roof, slanted roof, broken roof, ruins, cracks, damage, debris, hole, distorted architecture, blurry, transparent"

    output_image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=init_image,
        mask_image=mask_image,
        control_image=control_image,
        height=512,
        width=512,
        num_inference_steps=30,
        guidance_scale=9.5,                 # توجيه قوي جداً للنص للتخلص من أي انحدار
        controlnet_conditioning_scale=0.5
    ).images[0]

    output_image.save("ai_generated_512.png")
    return "ai_generated_512.png"

# ==========================================
# الخطوة 4: دمج الصورة النهائية
# ==========================================
def post_process_blend_highres(original_path, generated_512_path, mask_512_path):
    original = imread_unicode(original_path)
    h, w, _ = original.shape

    generated = cv2.imread(generated_512_path)
    mask = cv2.imread(mask_512_path, cv2.IMREAD_GRAYSCALE)

    generated_upscaled = cv2.resize(generated, (w, h), interpolation=cv2.INTER_CUBIC)
    mask_upscaled = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)

    mask_blurred = cv2.GaussianBlur(mask_upscaled, (15, 15), 0)
    mask_3ch = cv2.cvtColor(mask_blurred, cv2.COLOR_GRAY2BGR) / 255.0

    final_blended = (generated_upscaled * mask_3ch) + (original * (1.0 - mask_3ch))
    
    _, buf = cv2.imencode('.png', final_blended)
    buf.tofile("final_building_restored_gtx1650.png")
    print("تم إعادة ترميم المبنى والسطح بنجاح!")

# --- المسار الخاص بالصورة ---
image_path = r"C:\Users\RAED\Desktop\A-Smart-Site-For-Rehabilitating-Damaged-Buildings\ai\inputs\photo_2026-08-09_12-04-01.jpg"

# --- تشغيل المسار ---
mask_file = create_damage_mask(image_path)
generated_file = run_local_inpainting_low_vram(image_path, mask_file)
post_process_blend_highres(image_path, generated_file, mask_file)