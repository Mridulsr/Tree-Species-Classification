# 🌲 Tree Species Classification & Recommendation System

An AI-powered environmental analytics dashboard designed to identify tree species from images and recommend suitable flora based on geographical and physical attributes. Built using **Streamlit**, this multi-faceted application bridges the gap between complex machine learning workflows and real-world ecological management.

## 🚀 Live Demo
🔗 **Explore the live dashboard here:** [Tree Intelligence Assistant Live Link](https://tree-species-classificationgit-4afua3w5mzkzgsnzsgk57j.streamlit.app/)

---

## 📌 Problem Statement & Solution
### The Challenge
* **Identification Bottlenecks:** Non-experts struggle to accurately identify tree species, hindering local gardening, urban planning, and micro-ecological studies.
* **Fragmented Location Guidance:** Determining which tree species thrive in a specific geographical location or analyzing species distribution patterns manually is incredibly labor-intensive.

### The Solution: "Tree Intelligence Assistant"
1. **Recommend Trees by Location:** Leverages spatial coordinate modeling to suggest species likely to flourish in a given region based on latitude, longitude, and baseline tree dimensions.
2. **Find Locations for a Tree:** Allows users to query individual species to dynamically aggregate and track their density patterns across major urban cities and states.
3. **Identify Tree from Image:** Implements a deep learning engine to instantly classify a tree's species from an uploaded mobile or drone capture, supplying top-3 confidence scores mapped directly to regional growth profiles.

---

## ✨ Core Features
* **Dual-Model Processing Engine:** Blends traditional data-driven spatial algorithms with state-of-the-art computer vision inside a unified user interface.
* **High-Performance Caching:** Utilizes Streamlit's `@st.cache_data` and `@st.cache_resource` protocols to execute lightweight, lightning-fast application boots by freezing datasets and deep neural network layers in application memory.
* **Robust Error Scopes:** Implemented structured try-except layers to smoothly intercept internal pipeline dependencies or missing weights without disrupting user session states.

---

## 🛠️ Tech Stack & Dependencies
* **Core Language:** Python
* **Web Architecture:** Streamlit (Interactive Frontend)
* **Data & Math Processors:** Pandas, NumPy, Collections (Counter)
* **Machine Learning Runtime:** Scikit-Learn (NearestNeighbors, StandardScaler) via Joblib
* **Deep Learning Engine:** TensorFlow / Keras (Convolutional Neural Networks)
* **Image Processing Engine:** PIL (Pillow Image Layering)

---

## 📂 Project Structure
```text
├── .devcontainer/                    # Standardized isolated development container workspace
├── Tree_Species_Dataset/             # Core tracking folder for datasets and serialized training files
├── Code/                             # Source modules, testing assets, and modeling scripts
├── app.py                            # Streamlit frontend gateway and workflow controller
├── Tree_Species_Classification.ipynb # Google Colab pipeline used to train the CNN & KNN systems
├── Tree Species Classification.pptx  # Multi-page project slideshow and summary presentation
├── requirements.txt                  # Consolidated python dependency requirements
└── README.md                         # Current project documentation
