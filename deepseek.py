from transformers import AutoModelForCausalLM, AutoTokenizer

# กำหนดพาธของโมเดลที่ดาวน์โหลด
model_path = "deepseek_model"

# โหลด Tokenizer และ Model จากไฟล์ที่ดาวน์โหลด
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to("cuda")  # ใช้ GPU ถ้ามี
print("Model loaded successfully.")

# ฟังก์ชันทดสอบให้โมเดลสร้างข้อความ

def generate_text(prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_length=max_length)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# ทดสอบใช้งาน
if __name__ == "__main__":
    test_prompt = "Explain the Pythagorean theorem in simple terms."
    print("Generating response...")
    response = generate_text(test_prompt)
    print("Response:")
    print(response)
