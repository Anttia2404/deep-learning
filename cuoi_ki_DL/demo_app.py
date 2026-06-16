import streamlit as st
import cv2
import numpy as np
import torch
import tempfile
from pathlib import Path
import os
import time

# Import our codebase elements
import importlib, sys as _sys
# Tránh xung đột với thư viện datasets của Hugging Face
for _k in list(_sys.modules.keys()):
    if _k == 'datasets' or _k.startswith('datasets.'):
        del _sys.modules[_k]
from datasets.transforms import build_transforms
from inference.face_detector import FaceDetectionConfig, FaceDetector
from inference.predict_image import load_model_from_checkpoint
from training.trainer import resolve_device

# Set Page Config
st.set_page_config(
    page_title="Deepfake Detection Hub - Xception vs MoLD-ViT",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (CSS) for premium dark theme, glassmorphism and glowing effects
st.markdown("""
<style>
    /* Dark theme wrapper */
    .stApp {
        background: radial-gradient(circle, #0f172a 0%, #020617 100%);
        color: #f8fafc;
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    
    /* Header card styling */
    .header-card {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        backdrop-filter: blur(12px);
        margin-bottom: 24px;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    .header-title {
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.8rem;
        margin-bottom: 8px;
    }
    
    /* Status dots */
    .status-dot {
        height: 12px;
        width: 12px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
    }
    .status-online { background-color: #10b981; box-shadow: 0 0 8px #10b981; }
    .status-offline { background-color: #ef4444; box-shadow: 0 0 8px #ef4444; }
    
    /* Model Card styling */
    .model-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        transition: all 0.3s ease;
    }
    .model-card:hover {
        border: 1px solid rgba(56, 189, 248, 0.4);
        transform: translateY(-2px);
    }
    
    /* Glowing Verdict Cards */
    .verdict-box {
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 20px;
    }
    .verdict-real {
        background: rgba(16, 185, 129, 0.15);
        border: 2px solid #10b981;
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.3);
    }
    .verdict-fake {
        background: rgba(239, 68, 68, 0.15);
        border: 2px solid #ef4444;
        box-shadow: 0 0 20px rgba(239, 68, 68, 0.3);
    }
    .verdict-title {
        font-size: 2.2rem;
        font-weight: 900;
        margin-top: 10px;
        letter-spacing: 1px;
    }
    
    /* Custom buttons */
    div.stButton > button {
        background: linear-gradient(135deg, #0284c7 0%, #4f46e5 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        font-weight: bold !important;
        transition: all 0.3s ease !important;
        width: 100%;
        box-shadow: 0 4px 15px rgba(79, 70, 229, 0.4) !important;
    }
    div.stButton > button:hover {
        transform: scale(1.02) !important;
        box-shadow: 0 6px 20px rgba(79, 70, 229, 0.6) !important;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- APP HEADER -----------------
st.markdown("""
<div class="header-card">
    <h1 class="header-title">🛡️ Deepfake Detection Hub</h1>
    <p style="color: #94a3b8; font-size: 1.1rem; max-width: 900px; margin: 0 auto;">
        Hệ thống so sánh đối chiếu đa kiến trúc: <b>Xception (PyTorch)</b> vs <b>MoLD-ViT (Keras/TensorFlow)</b>.
        Hỗ trợ so sánh trực quan hiệu năng nhận diện giả mạo thời gian thực.
    </p>
</div>
""", unsafe_allow_html=True)

# ----------------- PATH & CHECKPOINT CHECKS -----------------
CHECKPOINT_DIR = Path("checkpoints")
MANIPULATIONS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]

# Map model filenames
ckpt_paths = {
    manip: CHECKPOINT_DIR / f"best_{manip}_c23.pth" for manip in MANIPULATIONS
}
KERAS_WEIGHTS_PATH = CHECKPOINT_DIR / "best_pure_mold.weights.h5"

# ----------------- SIDEBAR CONFIGURATION -----------------
st.sidebar.markdown("### ⚙️ Cấu hình bộ quét")

run_mode = st.sidebar.selectbox(
    "Chế độ chạy Demo",
    ["So sánh song song (Side-by-Side)", "Xception (PyTorch) - 4 Classes", "MoLD-ViT (Keras) - 1 Class"],
    index=0
)

# Base architecture select for Xception
model_arch = st.sidebar.selectbox(
    "Kiến trúc (dành cho PyTorch)",
    ["xception", "mesonet", "mesoinception4"],
    index=0
)

# Kiểm tra xem facenet-pytorch đã được cài chưa
try:
    from facenet_pytorch import MTCNN as _MTCNN
    _has_mtcnn = True
except ImportError:
    _has_mtcnn = False

_backend_options = ["mtcnn", "haar"] if _has_mtcnn else ["haar", "mtcnn"]
detector_backend = st.sidebar.selectbox(
    "Bộ phát hiện khuôn mặt (Backend)",
    _backend_options,
    index=0,
    help="haar: Không cần cài thêm thư viện. mtcnn: Cần cài facenet-pytorch, chính xác hơn."
)

num_frames = st.sidebar.slider(
    "Số frame phân tích (đối với Video)",
    min_value=4,
    max_value=128,
    value=32,
    step=4
)

threshold = st.sidebar.slider(
    "Ngưỡng quyết định (Threshold)",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05
)

# Select GPU if available
device_options = ["cpu"]
if torch.cuda.is_available():
    device_options.insert(0, "cuda")
device = st.sidebar.selectbox("Thiết bị tính toán (Device)", device_options, index=0)

# ----------------- MODEL STATUS ON SIDEBAR -----------------
st.sidebar.markdown("### 📁 Trạng thái Checkpoints")

# 1. PyTorch models status
st.sidebar.markdown("<p style='font-size:0.9rem; font-weight:bold; color:#38bdf8;'>1. PyTorch Models:</p>", unsafe_allow_html=True)
for manip, path in ckpt_paths.items():
    exists = path.exists()
    status_html = f'<span class="status-dot status-online"></span> Sẵn sàng' if exists else f'<span class="status-dot status-offline"></span> Thiếu file'
    st.sidebar.markdown(f"""
    <div class="model-card">
        <strong style="color: #e2e8f0;">{manip} Detector</strong><br/>
        <span style="font-size: 0.8rem; color: #94a3b8;">{path.name}</span><br/>
        <span style="font-size: 0.8rem;">{status_html}</span>
    </div>
    """, unsafe_allow_html=True)

# 2. Keras model status
st.sidebar.markdown("<p style='font-size:0.9rem; font-weight:bold; color:#818cf8;'>2. Keras Model (MoLD-ViT):</p>", unsafe_allow_html=True)
keras_exists = KERAS_WEIGHTS_PATH.exists()
keras_status_html = f'<span class="status-dot status-online"></span> Sẵn sàng' if keras_exists else f'<span class="status-dot status-offline"></span> Thiếu file'
st.sidebar.markdown(f"""
<div class="model-card">
    <strong style="color: #e2e8f0;">MoLD-ViT Classifier</strong><br/>
    <span style="font-size: 0.8rem; color: #94a3b8;">best_pure_mold.weights.h5</span><br/>
    <span style="font-size: 0.8rem;">{keras_status_html}</span>
</div>
""", unsafe_allow_html=True)


# ----------------- FUNCTION TO RUN INFERENCE (PYTORCH) -----------------
def run_single_inference(checkpoint_path, input_path, is_video):
    torch_device = resolve_device(device)
    
    # Load model
    try:
        model, checkpoint = load_model_from_checkpoint(checkpoint_path, model_arch, torch_device)
    except Exception as e:
        return {"error": f"Không thể load checkpoint: {str(e)}"}
        
    input_size = checkpoint.get("config", {}).get("input_size", 299 if model_arch == "xception" else 256)
    
    # Init face detector
    detector = FaceDetector(FaceDetectionConfig(
        backend=detector_backend,
        device=torch_device.type if torch_device.type == "cuda" else None,
        output_size=input_size
    ))
    transform = build_transforms(model_arch)["test"]
    
    # Face crop cache for display
    sample_face = None
    
    if is_video:
        cap = cv2.VideoCapture(str(input_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, max(total_frames - 1, 0), num_frames, dtype=int)
        probs = []
        
        with torch.no_grad():
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                success, frame = cap.read()
                if not success:
                    continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face = detector.detect_and_crop(rgb)
                if face is None:
                    continue
                
                # Cache first face found to show to user
                if sample_face is None:
                    sample_face = face
                    
                tensor = transform(face).unsqueeze(0).to(torch_device)
                logits = model(tensor)
                prob = torch.softmax(logits, dim=1)[0, 1].item()
                probs.append(prob)
        cap.release()
        
        if not probs:
            return {"label": "uncertain", "reason": "Không phát hiện thấy khuôn mặt", "prob_fake": 0.0, "sample_face": None}
            
        score = float(np.mean(probs))
        
    else: # Image
        image = cv2.imread(str(input_path))
        if image is None:
            return {"error": "Không đọc được file ảnh."}
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face = detector.detect_and_crop(rgb)
        if face is None:
            return {"label": "uncertain", "reason": "Không phát hiện thấy khuôn mặt", "prob_fake": 0.0, "sample_face": None}
            
        sample_face = face
        tensor = transform(face).unsqueeze(0).to(torch_device)
        with torch.no_grad():
            logits = model(tensor)
            score = torch.softmax(logits, dim=1)[0, 1].item()
            
    label = "FAKE" if score > threshold else "REAL"
    confidence = score if score > threshold else 1.0 - score
    
    return {
        "label": label,
        "prob_fake": score,
        "confidence": confidence,
        "sample_face": sample_face,
        "val_acc": checkpoint.get("val_acc", 0.0)
    }


# ----------------- FUNCTION TO RUN INFERENCE (KERAS / MOLD-VIT) -----------------
def run_keras_inference(weights_path, input_path, is_video):
    from inference.predict_keras import KerasModelManager
    import tensorflow as tf
    
    try:
        model = KerasModelManager.get_model(str(weights_path))
    except Exception as e:
        return {"error": f"Không thể load Keras model: {str(e)}"}
        
    input_size = 256
    torch_device = resolve_device(device)
    
    detector = FaceDetector(FaceDetectionConfig(
        backend=detector_backend,
        device=torch_device.type if torch_device.type == "cuda" else None,
        output_size=input_size
    ))
    
    sample_face = None
    
    if is_video:
        cap = cv2.VideoCapture(str(input_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, max(total_frames - 1, 0), num_frames, dtype=int)
        probs = []
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            success, frame = cap.read()
            if not success:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face = detector.detect_and_crop(rgb)
            if face is None:
                continue
                
            if sample_face is None:
                sample_face = face
                
            # Preprocess to float32 and divide by 255.0
            face_norm = face.astype(np.float32) / 255.0
            face_input = np.expand_dims(face_norm, axis=0)
            
            preds = model(face_input, training=False)
            prob = float(preds[0, 1].numpy())
            probs.append(prob)
        cap.release()
        
        if not probs:
            return {"label": "uncertain", "reason": "Không phát hiện thấy khuôn mặt", "prob_fake": 0.0, "sample_face": None}
            
        score = float(np.mean(probs))
        
    else: # Image
        image = cv2.imread(str(input_path))
        if image is None:
            return {"error": "Không đọc được file ảnh."}
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face = detector.detect_and_crop(rgb)
        if face is None:
            return {"label": "uncertain", "reason": "Không phát hiện thấy khuôn mặt", "prob_fake": 0.0, "sample_face": None}
            
        sample_face = face
        face_norm = face.astype(np.float32) / 255.0
        face_input = np.expand_dims(face_norm, axis=0)
        
        preds = model(face_input, training=False)
        score = float(preds[0, 1].numpy())
        
    label = "FAKE" if score > threshold else "REAL"
    confidence = score if score > threshold else 1.0 - score
    
    return {
        "label": label,
        "prob_fake": score,
        "confidence": confidence,
        "sample_face": sample_face,
        "val_acc": 95.25 # Accuracy on test set from paper
    }


# ----------------- MAIN LAYOUT -----------------
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("### 📥 Tải lên Tệp dữ liệu cần quét")
    uploaded_file = st.file_uploader(
        "Chọn Video (.mp4, .avi, .mov) hoặc Ảnh (.jpg, .png)",
        type=["mp4", "avi", "mov", "jpg", "png", "jpeg"]
    )
    
    if uploaded_file is not None:
        file_suffix = Path(uploaded_file.name).suffix
        temp_dir = tempfile.gettempdir()
        temp_path = Path(temp_dir) / f"temp_upload{file_suffix}"
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        is_video = file_suffix.lower() in [".mp4", ".avi", ".mov"]
        
        if is_video:
            st.video(open(temp_path, "rb").read())
            st.info(f"Video: {uploaded_file.name}")
        else:
            st.image(str(temp_path), use_column_width=True)
            st.info(f"Ảnh: {uploaded_file.name}")
            
        run_scan = st.button("🚀 Bắt đầu phân tích")

with col_right:
    st.markdown("### 📊 Kết quả phân tích")
    
    if uploaded_file is None:
        st.info("Đang chờ tải tệp lên từ bảng điều khiển bên trái...")
    elif uploaded_file is not None and not run_scan:
        st.success("Tải tệp thành công! Hãy bấm 'Bắt đầu phân tích' bên trái để quét.")
    elif uploaded_file is not None and run_scan:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # ----------------- SIDE-BY-SIDE MODE -----------------
        if run_mode == "So sánh song song (Side-by-Side)":
            active_pytorch = {m: p for m, p in ckpt_paths.items() if p.exists()}
            
            # 1. Run PyTorch Xception Scan
            pytorch_results = {}
            sample_face_to_show = None
            if active_pytorch:
                status_text.text("🔍 Đang chạy mô hình PyTorch Xception...")
                for idx, (manip, path) in enumerate(active_pytorch.items()):
                    res = run_single_inference(path, temp_path, is_video)
                    if "error" not in res:
                        pytorch_results[manip] = res
                        if res.get("sample_face") is not None:
                            sample_face_to_show = res["sample_face"]
                progress_bar.progress(50)
            
            # 2. Run Keras MoLD-ViT Scan
            keras_result = None
            if KERAS_WEIGHTS_PATH.exists():
                status_text.text("🔍 Đang chạy mô hình Keras MoLD-ViT...")
                keras_result = run_keras_inference(KERAS_WEIGHTS_PATH, temp_path, is_video)
                if keras_result.get("sample_face") is not None and sample_face_to_show is None:
                    sample_face_to_show = keras_result["sample_face"]
                progress_bar.progress(100)
            
            status_text.text("✅ Quá trình so sánh hoàn tất!")
            progress_bar.empty()
            
            # Display Side-by-side results
            c_py, c_keras = st.columns(2)
            
            # Col 1: PyTorch Results
            with c_py:
                st.markdown("<h4 style='color:#38bdf8; text-align:center;'>PyTorch (Xception)</h4>", unsafe_allow_html=True)
                if pytorch_results:
                    # Calculate overall PyTorch label
                    max_fake_score = max([r["prob_fake"] for r in pytorch_results.values()])
                    is_fake = max_fake_score > threshold
                    
                    if is_fake:
                        st.markdown("""<div class="verdict-box verdict-fake" style="padding:15px;"><div class="verdict-title" style="font-size:1.5rem;">⚠️ FAKE</div></div>""", unsafe_allow_html=True)
                    else:
                        st.markdown("""<div class="verdict-box verdict-real" style="padding:15px;"><div class="verdict-title" style="font-size:1.5rem;">✅ REAL</div></div>""", unsafe_allow_html=True)
                    
                    # Probabilities breakdown
                    for manip, res in pytorch_results.items():
                        score_pct = res["prob_fake"] * 100
                        st.write(f"**{manip}**: {score_pct:.1f}%")
                        st.progress(res["prob_fake"])
                else:
                    st.warning("Không tìm thấy checkpoint PyTorch (.pth).")
                    
            # Col 2: Keras Results
            with c_keras:
                st.markdown("<h4 style='color:#818cf8; text-align:center;'>Keras (MoLD-ViT)</h4>", unsafe_allow_html=True)
                if keras_result and "error" not in keras_result:
                    is_fake = keras_result["prob_fake"] > threshold
                    if is_fake:
                        st.markdown("""<div class="verdict-box verdict-fake" style="padding:15px;"><div class="verdict-title" style="font-size:1.5rem;">⚠️ FAKE</div></div>""", unsafe_allow_html=True)
                    else:
                        st.markdown("""<div class="verdict-box verdict-real" style="padding:15px;"><div class="verdict-title" style="font-size:1.5rem;">✅ REAL</div></div>""", unsafe_allow_html=True)
                        
                    score_pct = keras_result["prob_fake"] * 100
                    st.write(f"**Độ tương thích MoLD-ViT**: {score_pct:.1f}%")
                    st.progress(keras_result["prob_fake"])
                else:
                    if keras_result and "error" in keras_result:
                        st.error(keras_result["error"])
                    else:
                        st.warning("Không tìm thấy checkpoint Keras (best_pure_mold.weights.h5).")
            
            # Show shared Face Crop
            if sample_face_to_show is not None:
                st.markdown("<hr/>", unsafe_allow_html=True)
                st.markdown("<p style='font-size: 1rem; color: #94a3b8; font-weight: bold; text-align:center;'>Khuôn mặt phân tích:</p>", unsafe_allow_html=True)
                st.image(sample_face_to_show, width=150, caption="Detected Face", use_column_width=False)
                
        # ----------------- PYTORCH ONLY MODE -----------------
        elif run_mode == "Xception (PyTorch) - 4 Classes":
            active_paths = {m: p for m, p in ckpt_paths.items() if p.exists()}
            if not active_paths:
                st.error("Không tìm thấy tệp checkpoint PyTorch (.pth) nào.")
            else:
                results = {}
                sample_face_to_show = None
                total_active = len(active_paths)
                for i, (manip, ckpt_path) in enumerate(active_paths.items()):
                    status_text.text(f"🔍 Đang quét bằng: {manip}...")
                    res = run_single_inference(ckpt_path, temp_path, is_video)
                    if "error" not in res:
                        results[manip] = res
                        if res.get("sample_face") is not None:
                            sample_face_to_show = res["sample_face"]
                    progress_bar.progress(int((i + 1) / total_active * 100))
                    
                status_text.text("✅ Quá trình quét hoàn tất!")
                progress_bar.empty()
                
                if results:
                    fake_scores = [r["prob_fake"] for r in results.values()]
                    max_fake_score = max(fake_scores)
                    max_manip = [m for m, r in results.items() if r["prob_fake"] == max_fake_score][0]
                    is_fake = max_fake_score > threshold
                    
                    if is_fake:
                        st.markdown(f"""
                        <div class="verdict-box verdict-fake">
                            <span style="font-size: 1.1rem; color: #fca5a5;">CẢNH BÁO</span>
                            <div class="verdict-title">⚠️ FAKE DETECTED</div>
                            <p style="margin-top: 5px; color: #fecaca;">Tương thích cao nhất với <b>{max_manip} ({max_fake_score*100:.1f}%)</b>.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="verdict-box verdict-real">
                            <span style="font-size: 1.1rem; color: #6ee7b7;">AN TOÀN</span>
                            <div class="verdict-title">✅ REAL FACE</div>
                            <p style="margin-top: 5px; color: #a7f3d0;">Khuôn mặt tự nhiên (Xác suất giả tối đa: {max_fake_score*100:.1f}%).</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    col_face, col_chart = st.columns([1, 2])
                    with col_face:
                        if sample_face_to_show is not None:
                            st.image(sample_face_to_show, use_column_width=True, caption="Face Crop")
                    with col_chart:
                        for manip in MANIPULATIONS:
                            if manip in results:
                                res = results[manip]
                                st.write(f"**{manip}**: {res['prob_fake']*100:.1f}%")
                                st.progress(res["prob_fake"])
                                
        # ----------------- KERAS ONLY MODE -----------------
        else:
            if not KERAS_WEIGHTS_PATH.exists():
                st.error("Không tìm thấy checkpoint Keras (best_pure_mold.weights.h5).")
            else:
                status_text.text("🔍 Đang chạy phân tích Keras MoLD-ViT...")
                res = run_keras_inference(KERAS_WEIGHTS_PATH, temp_path, is_video)
                progress_bar.progress(100)
                status_text.text("✅ Hoàn tất!")
                progress_bar.empty()
                
                if "error" in res:
                    st.error(res["error"])
                else:
                    is_fake = res["prob_fake"] > threshold
                    if is_fake:
                        st.markdown(f"""
                        <div class="verdict-box verdict-fake">
                            <span style="font-size: 1.1rem; color: #fca5a5;">CẢNH BÁO</span>
                            <div class="verdict-title">⚠️ FAKE DETECTED</div>
                            <p style="margin-top: 5px; color: #fecaca;">Mô hình MoLD-ViT dự đoán: <b>{res['prob_fake']*100:.1f}% FAKE</b>.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="verdict-box verdict-real">
                            <span style="font-size: 1.1rem; color: #6ee7b7;">AN TOÀN</span>
                            <div class="verdict-title">✅ REAL FACE</div>
                            <p style="margin-top: 5px; color: #a7f3d0;">Mô hình MoLD-ViT dự đoán: <b>{(1.0-res['prob_fake'])*100:.1f}% REAL</b>.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    col_face, col_chart = st.columns([1, 2])
                    with col_face:
                        if res.get("sample_face") is not None:
                            st.image(res["sample_face"], use_column_width=True, caption="Face Crop")
                    with col_chart:
                        st.write(f"**Xác suất giả mạo**: {res['prob_fake']*100:.1f}%")
                        st.progress(res["prob_fake"])
                        st.write(f"**Độ chính xác Test Set (Paper)**: 95.25%")
                        
        # Cleanup temp file
        if temp_path.exists():
            os.remove(temp_path)
