# 🚀 DeepSeek-R1-Distill-Qwen-1.5B Offline Setup

This repository provides a step-by-step guide to setting up and running **DeepSeek-R1-Distill-Qwen-1.5B** offline on a local machine. The model is useful for NLP tasks such as text generation, code completion, and knowledge inference while maintaining full control over data privacy.

---

## 📌 Features
- 🔹 **Runs Fully Offline**: No internet connection required after setup.
- 🔹 **NLP-Powered AI**: Generates human-like text responses.
- 🔹 **Fast & Secure**: Runs locally, ensuring data security.
- 🔹 **Supports GPU Acceleration**: Uses CUDA for faster inference (if available).

---

## 🛠 Prerequisites
Before setting up, ensure your system meets the following requirements:

### **System Requirements**
- **CPU**: Intel i5/i7 or AMD Ryzen 5/7 (or better)
- **RAM**: Minimum 16GB (32GB+ recommended for heavy workloads)
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)
- **Storage**: At least 50GB free space
- **OS**: Linux / macOS / Windows

### **Dependencies**
Make sure you have **Python 3.8+** installed. Then, install required dependencies:
```bash
pip install torch torchvision torchaudio transformers accelerate
```
(For GPU acceleration, install the compatible PyTorch version for CUDA.)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 📦 Installation & Setup
### **Step 1: Clone the Repository**
```bash
git clone https://github.com/your-username/deepseek-offline.git
cd deepseek-offline
```

### **Step 2: Create a Virtual Environment**
```bash
python -m venv deepseek_env
source deepseek_env/bin/activate  # For Linux/macOS
# On Windows: deepseek_env\Scripts\activate
```

### **Step 3: Download the Model Offline**
```bash
pip install huggingface_hub
huggingface-cli login
huggingface-cli download deepseek-ai/deepseek-r1-distill-qwen-1.5b --local-dir deepseek_model
```

### **Step 4: Run the Model**
Execute the script to generate responses:
```bash
python run_deepseek.py
```

---

## 🏃 Usage
### **Running the Model in Python**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "deepseek_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to("cuda")

def generate_text(prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(output[0], skip_special_tokens=True)

response = generate_text("Explain the Pythagorean theorem in simple terms.")
print(response)
```

---

## 🤝 Contributing
Contributions are welcome! Feel free to submit pull requests or report issues.

---

## 📜 License
This project is licensed under the MIT License.

---

## 🌟 Acknowledgments
Special thanks to **DeepSeek AI** and **Hugging Face** for their amazing contributions to the AI community.

---

