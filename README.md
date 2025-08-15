
# 🚗 Automatic Number Plate Recognition (ANPR) with TensorFlow & Tesseract

This project implements a full pipeline to detect and recognize vehicle license plates from images using deep learning (InceptionResNetV2) and OCR (Tesseract).

---

## 🎯 Objectives

- Detect license plates using a CNN trained on bounding box data (VOC XML)
- Extract and read plate text using pytesseract
- Visualize results and export recognized text to CSV

---

## 📂 File Structure

```
ANPR/
├── images/                   # Input images & annotation XMLs
├── data_preprocessing.ipynb  # Extract bounding boxes to CSV
├── model_training.ipynb      # Train detector
├── plate_ocr.py              # Apply OCR
├── labels.csv                # Bounding box info
└── README.md
```

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

## 🖥️ How to Use

```bash
# Step 1: Convert XML labels to CSV
python data_preprocessing.py

# Step 2: Train the detector
python train_detector.py

# Step 3: Run OCR on detected plates
python plate_ocr.py
```

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
