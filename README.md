# Smart Photo Manager  
### AI-assisted Photo Understanding & Organization System

---

##  Overview

Smart Photo Manager is an AI-assisted image management system that integrates **computer vision, image quality assessment, and metadata analysis** to enable intelligent photo organization and user-centered visual exploration.

The system is designed to go beyond traditional file-based photo management by introducing **semantic understanding of images** through multiple perception channels, including:

- Cloud-based image recognition (Google Vision API)
- Aesthetic quality estimation (MUSIQ model)
- Low-level image feature extraction (OpenCV)
- EXIF metadata analysis
- Rule-based + AI-assisted recommendation system

The goal is to explore how multimodal visual signals can be combined into a unified framework for **intelligent personal media understanding**.

---

##  Motivation

With the rapid increase in digital photography, users face several challenges:

- Difficulty organizing large-scale personal image collections
- Lack of semantic understanding of photos
- Inefficient retrieval based on filenames or manual folders
- Absence of intelligent feedback for photography quality

This project explores a lightweight approach to **human-centered visual data organization**, combining AI perception with traditional image processing techniques.

---

##  System Architecture

The system follows a multi-stage processing pipeline:

1. **Image Ingestion**
   - Batch import from local directories
   - SQLite-based persistent storage

2. **Feature Extraction**
   - EXIF metadata parsing (ISO, aperture, shutter speed, focal length)
   - OpenCV-based brightness and sharpness estimation
   - Color distribution analysis

3. **AI Understanding Layer**
   - Google Vision API for semantic labeling
   - Scene classification (portrait, landscape, night, etc.)

4. **Aesthetic Evaluation**
   - MUSIQ model for perceptual quality estimation

5. **Recommendation Engine**
   - Rule-based photography suggestions
   - AI-assisted improvement feedback

6. **Visualization & Interaction**
   - Tkinter-based GUI system
   - Image gallery browsing interface
   - Histogram and feature visualization
   - Basic image editing tools (before/after comparison)

---

##  Key Features

###  Image Understanding
- Automatic image labeling via Google Vision API
- Scene recognition and categorization

###  Image Quality Analysis
- Sharpness estimation (Laplacian variance)
- Brightness / exposure evaluation
- Aesthetic scoring via deep learning model (MUSIQ)

###  Intelligent Suggestions
- Photography improvement recommendations
- Scene-aware guidance (lighting, exposure, composition)

###  Smart Library System
- Semantic image filtering
- Thumbnail-based browsing
- SQLite-based indexing system

###  Image Editing Tools
- Exposure, contrast, saturation adjustment
- Crop, rotate, sharpen, denoise
- Before/after comparison view

---

##  Tech Stack

- Python 3
- Tkinter (GUI)
- OpenCV
- Pillow (PIL)
- NumPy
- SQLite
- Google Cloud Vision API
- TensorFlow / TensorFlow Hub (MUSIQ)
- Matplotlib

---

##  Example Use Cases

- Intelligent organization of personal photo libraries
- AI-based photography quality evaluation
- Visual analytics for image datasets
- Exploration of multimodal image understanding systems

---

## bash

pip install -r requirements.txt
python smart_photo_manager.py

---

##Environment Setup

Set your Google Vision API key:
```bash
export GOOGLE_API_KEY="your_api_key_here"
```
Or configure it in the application settings UI.

---

## Future Work
- Replace rule-based logic with learned deep learning models
- Integrate CLIP-based multimodal retrieval system
- Improve scene classification accuracy
- Extend system to video understanding
- Add similarity-based image retrieval functionality

---

## Research Perspective
This project can be viewed as a lightweight exploration of:

Human-centered multimodal image understanding for personal media organization

It integrates concepts from:

- Computer vision
- Aesthetic modeling
- Human-AI interaction
- Interactive visualization systems

---

## Author
Zhang Jiayi

BSc Computer Science (First Class Honours)

Coventry University
