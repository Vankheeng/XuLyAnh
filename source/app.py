# app.py - PHIÊN BẢN HOÀN HẢO 2025
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="AI Nhận Diện Chữ & Hình Dạng", layout="wide")
st.title("Nhận Diện Chữ Viết  & Hình Dạng Đơn Giản")

# ================= LOAD MODELS =================
@st.cache_resource
def load_models():
    with st.spinner("Đang tải mô hình chữ viết ..."):
        model_chuviet = tf.keras.models.load_model('models\handwriting_model.h5', compile=False)
    with st.spinner("Đang tải mô hình hình dạng..."):
        model_hinhdang = tf.keras.models.load_model('models\shape_detector_3classes.h5')
    return model_chuviet, model_hinhdang

model_chuviet, model_hinhdang = load_models()

# ================= TAB CHỮ VIẾT  =================
tab1, tab2 = st.tabs(["Chữ Viết", "Hình Dạng"])

with tab1:
    st.header("Viết chữ bằng tay hoặc tải ảnh lên")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.write("Viết chữ ở đây ")
        canvas = st_canvas(
            fill_color="white",
            stroke_width=18,
            stroke_color="black",
            background_color="white",
            height=220,
            width=680,
            drawing_mode="freedraw",
            key="canvas_chu",
            display_toolbar=True,
        )

        uploaded = st.file_uploader(
            "Hoặc tải ảnh chữ viết lên",
            type=["png", "jpg", "jpeg"],
            key="upload_chu"
        )

        if st.button("Xóa vùng vẽ chữ", use_container_width=True):
            st.rerun()

    with col_right:
        st.write("### Kết quả nhận diện")
        result_placeholder = st.empty()
        img_display = st.empty()

        if canvas.image_data is not None or uploaded:
            # Lấy ảnh
            if uploaded:
                img = np.array(Image.open(uploaded).convert("RGB"))
            else:
                img = canvas.image_data.astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

            # Hiển thị ảnh gốc
            img_display.image(img, caption="Ảnh đầu vào", use_column_width=True)

            # === PREPROCESS ===
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
            coords = cv2.findNonZero(thresh)
            if coords is not None:
                x, y, w, h = cv2.boundingRect(coords)
                gray = gray[max(0, y-10):y+h+10, max(0, x-10):x+w+10]

            resized = cv2.resize(gray, (256, 64))
            rotated = np.rot90(resized, k=1)  # ĐÚNG SHAPE (256, 64)
            normalized = rotated.astype("float32") / 255.0
            final_input = np.expand_dims(np.expand_dims(normalized, -1), 0)

            # === PREDICT ===
            pred = model_chuviet.predict(final_input, verbose=0)

            # === DECODE CTC ===
            alphabets = "ABCDEFGHIJKLMNOPQRSTUVWXYZ-' "
            decoded = tf.keras.backend.ctc_decode(pred, input_length=np.ones(1)*pred.shape[1], greedy=True)[0][0]
            text = ""
            prev = -1
            for i in decoded[0]:
                idx = i.numpy()
                if idx != -1 and idx != prev and idx < len(alphabets):
                    text += alphabets[idx]
                prev = idx

            result_placeholder.success(f"**Kết quả: {text.upper() if text else 'Không nhận diện được'}**")

# ================= TAB HÌNH DẠNG (ĐÃ SỬA HOÀN TOÀN) =================
with tab2:
    st.header("Vẽ hoặc tải lên hình tròn, vuông, tam giác")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.write("**Vẽ hình ở đây**")
        
        # canvas_shape = st_canvas(
        #     fill_color="white",
        #     stroke_width=12,
        #     stroke_color="black",
        #     background_color="white",
        #     height=300,
        #     width=300,
        #     drawing_mode="freedraw",
        #     key="canvas_shape_real",  # key khác để tránh xung đột
        #     display_toolbar=True,
        # )
        canvas_shape = st_canvas(
            fill_color="white",
            stroke_width=18,
            stroke_color="black",
            background_color="white",
            height=220,
            width=680,
            drawing_mode="freedraw",
            key="canvas_shape",
            display_toolbar=True,
        )

        uploaded_shape = st.file_uploader(
            "Tải ảnh hình dạng",
            type=["png", "jpg", "jpeg"],
            key="upload_shape"
        )

        if st.button("Xóa hình vẽ", use_container_width=True):
            st.rerun()

    with col_right:
        st.write("### Kết quả phân tích")
        shape_img = st.empty()
        shape_result = st.empty()

        if canvas_shape.image_data is not None or uploaded_shape:
            # Lấy ảnh
            if uploaded_shape:
                img = np.array(Image.open(uploaded_shape).convert("RGB"))
            else:
                raw = canvas_shape.image_data.astype(np.uint8)
                img = cv2.cvtColor(raw, cv2.COLOR_RGBA2RGB)

            # Resize đúng input model
            img_resized = cv2.resize(img, (128, 128))
            img_input = img_resized.astype("float32") / 255.0
            img_input = np.expand_dims(img_input, axis=0)

            # Predict
            pred = model_hinhdang.predict(img_input, verbose=0)
            classes = ['Circle', 'Square', 'Triangle']
            idx = np.argmax(pred[0])
            confidence = pred[0][idx]

            # Hiển thị
            shape_img.image(img, caption=f"Ảnh được nhận diện", use_column_width=True)
            shape_result.success(f"**Hình dạng: {classes[idx]}**\n\nĐộ tin cậy: {confidence:.1%}")

# ================= FOOTER =================
st.markdown("---")
st.caption("Website by Nhóm 7 - D22CNPM02 - AI Nhận Diện Chữ Viết & Hình Dạng Đơn Giản ")