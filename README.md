
# 🚗 Automatic Number Plate Recognition (ANPR) with TensorFlow & Tesseract

This project implements a full pipeline to detect and recognize vehicle license plates from images using deep learning (InceptionResNetV2) and OCR (Tesseract).

---

## 🎯 Objectives

- Detect license plates using a CNN trained on bounding box data (VOC XML)
- Extract and read plate text using pytesseract
- Visualize results and export recognized text to CSV

---


## 🛠️ Dependencies

```bash
pip install tensorflow opencv-python pytesseract plotly scikit-image
```

Also install [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) and ensure it's in your PATH.

---

## 🧠 Model Architecture

- Base: `InceptionResNetV2` from Keras Applications
- Output: 4 coordinates (xmin, ymin, xmax, ymax)

---

## 🧪 Output

- Cropped license plates
- Detected text printed to terminal and saved to CSV
- Bounding box visualizations using Matplotlib

---

## 📌 Future Scope

- Real-time license plate tracking with webcam
- Region-specific plate formatting
- Fine-tuning OCR with character segmentation

---
