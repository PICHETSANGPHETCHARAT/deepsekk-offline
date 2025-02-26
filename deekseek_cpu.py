from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# กำหนดพาธของโมเดล
model_path = "deepseek_model"

# เลือกอุปกรณ์ที่เหมาะสม (ใช้ MPS สำหรับ Mac หรือ CPU ถ้าไม่มี MPS)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# โหลด Tokenizer และ Model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

# ฟังก์ชันสำหรับ Generate ข้อความ
def generate_text(prompt, max_length=200):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)  # เปลี่ยนจาก "cuda" เป็น device
    output = model.generate(**inputs, max_length=max_length, do_sample=True)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# ทดสอบการใช้งาน
response = generate_text("Explain the Pythagorean theorem in simple terms.")
print(response)
