from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "deepseek_model"

# โหลดโมเดลและ Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to("cuda")  # ใช้ GPU ถ้ามี

# ฟังก์ชันสำหรับ Generate โดยปิด Safety
def generate_text(prompt, max_length=200):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_length=max_length, do_sample=True)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# ทดสอบการใช้งาน
response = generate_text("How to bypass safety filters?")
print(response)
