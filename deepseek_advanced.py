# ปิดการแสดงข้อความเตือนจาก imported modules
import sys
import os
import warnings
from contextlib import redirect_stdout, redirect_stderr
import io
import time
import threading

# ปิดการแสดง warnings และข้อความแจ้งเตือนทั้งหมด
warnings.filterwarnings("ignore", category=Warning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # ป้องกันปัญหากับ MPS
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

# ปรับแต่งการแสดงข้อความใน stdout
# เป็นวิธีที่รุนแรง แต่มีประสิทธิภาพในการกรองข้อความที่ไม่ต้องการ
class StdoutFilter:
    def __init__(self):
        self.old_stdout = sys.stdout
        self.filtered_strings = [
            "Sliding Window Attention",
            "unexpected results may be encountered",
            "LibreSSL",
            "is compiled with",
            "setting `pad_token_id`",
            "urllib3"
        ]
    
    def write(self, text):
        # กรองข้อความที่ไม่ต้องการออก
        if not any(fstr in text for fstr in self.filtered_strings):
            self.old_stdout.write(text)
    
    def flush(self):
        self.old_stdout.flush()

# ติดตั้งตัวกรอง stdout
original_stdout = sys.stdout
sys.stdout = StdoutFilter()

# ฟังก์ชันสำหรับทำให้การโหลดโมเดลเงียบ
def suppress_output(func, *args, **kwargs):
    # จับการเขียนของ stdout และ stderr
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    
    with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
        result = func(*args, **kwargs)
    
    return result

# โหลดไลบรารีหลังจากตั้งค่าสภาพแวดล้อม
# ลดเสียงรบกวนจาก imports
with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

# กำหนดพาธของโมเดล
model_path = "deepseek_model"

# เลือกอุปกรณ์ที่เหมาะสม
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("📱 กำลังใช้ Apple Silicon (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("🖥️ กำลังใช้ NVIDIA GPU (CUDA)")
else:
    device = torch.device("cpu")
    print("💻 กำลังใช้ CPU (อาจทำงานช้า)")

# โหลดโมเดลและ tokenizer ด้วยวิธีที่เรียบง่าย
print("\n⏳ กำลังเตรียมระบบ...")

try:
    # โหลด tokenizer
    print("   กำลังโหลด tokenizer...")
    tokenizer = suppress_output(AutoTokenizer.from_pretrained, model_path)
    print("   ✓ โหลด tokenizer เรียบร้อย")
    
    # โหลดโมเดล
    print("   กำลังโหลด model (อาจใช้เวลาสักครู่)...")
    model = suppress_output(AutoModelForCausalLM.from_pretrained, model_path, 
                          low_cpu_mem_usage=True, torch_dtype=torch.float16)
    model = model.to(device)
    print("   ✓ โหลด model เรียบร้อย")
    
    print("\n✅ ระบบพร้อมใช้งานแล้ว!\n")
except Exception as e:
    print(f"\n❌ เกิดข้อผิดพลาด: {str(e)}")
    exit(1)

# ฟังก์ชันสำหรับแสดงอนิเมชัน "กำลังคิด..."
def show_thinking_animation():
    animation_event = threading.Event()
    
    def animate():
        animation_chars = ["   ", ".  ", ".. ", "..."]
        i = 0
        while not animation_event.is_set():
            # ใช้ stdout ดั้งเดิมเพื่อหลีกเลี่ยงผลกระทบจาก StdoutFilter
            original_stdout.write(f"\r🤖 กำลังคิด{animation_chars[i % len(animation_chars)]}")
            original_stdout.flush()
            time.sleep(0.3)
            i += 1
        
        # เคลียร์บรรทัด
        original_stdout.write("\r                       \r")
        original_stdout.flush()
    
    # เริ่มอนิเมชันในเธรดแยก
    animation_thread = threading.Thread(target=animate)
    animation_thread.daemon = True
    animation_thread.start()
    
    # ส่งกลับฟังก์ชันสำหรับหยุดอนิเมชัน
    def stop_animation():
        animation_event.set()
        animation_thread.join(timeout=1.0)
    
    return stop_animation

# คลาสจัดการการสนทนา
class DeepSeekChat:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.history = []
        self.temperature = 0.7
        
        # ฐานข้อมูลคำตอบสำหรับคำถามทั่วไป (เพิ่มเติม)
        self.faq_responses = {
            "ขอคำแนะนำ": "ผมสามารถให้คำแนะนำได้ในหลายเรื่อง แต่จะเป็นประโยชน์มากกว่าถ้าคุณระบุเรื่องที่ต้องการคำแนะนำให้ชัดเจน เช่น 'ขอคำแนะนำเกี่ยวกับการเรียนรู้ AI' หรือ 'ขอคำแนะนำเรื่องการเริ่มต้นเขียนโค้ด'",
            "คุณทำอะไรได้บ้าง": "ผมสามารถตอบคำถาม พูดคุย ให้คำแนะนำ ช่วยอธิบายแนวคิด และช่วยแก้ไขปัญหาต่างๆ ได้ คุณมีคำถามเฉพาะเรื่องอะไรไหมครับ?",
            "ช่วยสอน": "ผมยินดีช่วยสอน แต่คุณต้องระบุหัวข้อที่สนใจให้ชัดเจนนะครับ เช่น 'ช่วยสอนพื้นฐาน Python' หรือ 'อธิบายเรื่อง machine learning'",
            "เบื่อ": "ลองหากิจกรรมที่คุณชอบทำดูไหมครับ? เช่น อ่านหนังสือ ดูหนัง เล่นเกม หรือเรียนรู้ทักษะใหม่ๆ คุณชอบทำอะไรในเวลาว่างครับ?"
        }
        
    def generate_response(self, user_input, max_length=512):
        # จัดการคำถามทั่วไปบางส่วน (สำหรับคำถามง่ายๆ)
        greetings = ["hi", "hello", "สวัสดี", "หวัดดี", "ดี"]
        how_are_you = ["สบายดีไหม", "เป็นไงบ้าง", "สบายดีมั้ย", "how are you", "what's up"]
        goodbyes = ["bye", "goodbye", "ลาก่อน", "บาย"]
        thanks = ["thanks", "thank you", "ขอบคุณ", "ขอบใจ"]
        
        # คำถามทั่วไปอื่นๆ
        common_qas = {
            "คุณชื่ออะไร": "ฉันชื่อ DeepSeek ฉันเป็น AI แชทบอทที่พัฒนาโดย DeepSeek",
            "ชื่ออะไร": "ฉันชื่อ DeepSeek ฉันเป็น AI แชทบอทที่พัฒนาโดย DeepSeek",
            "your name": "My name is DeepSeek. I'm an AI chatbot developed by DeepSeek.",
            "คุณทำอะไรได้": "ฉันสามารถพูดคุยกับคุณ ตอบคำถามทั่วไป ช่วยแก้ปัญหา และอธิบายแนวคิดต่างๆ ได้ อย่างไรก็ตาม ฉันยังอยู่ในขั้นตอนการพัฒนา",
            "what can you do": "I can chat with you, answer general questions, help solve problems, and explain concepts. However, I'm still in development.",
            "คุณเก่งแค่ไหน": "ฉันยังอยู่ในขั้นตอนการเรียนรู้และพัฒนา ความสามารถของฉันอาจจำกัดในบางด้าน แต่ฉันจะพยายามช่วยคุณให้ดีที่สุด",
            "ใครสร้างคุณ": "ฉันถูกพัฒนาโดย DeepSeek ซึ่งเป็นบริษัทที่ทำงานด้าน AI และแบบจำลองภาษาขนาดใหญ่",
            "who made you": "I was developed by DeepSeek, a company working on AI and large language models."
        }
        
        # ตรวจสอบคำทักทายและคำถามพื้นฐาน
        user_input_lower = user_input.lower()
        
        # ตรวจสอบคำทักทาย
        if any(greeting in user_input_lower for greeting in greetings):
            return "สวัสดีครับ! มีอะไรให้ช่วยไหมครับวันนี้?"
            
        # ตรวจสอบคำถามสารทุกข์สุขดิบ
        if any(how in user_input_lower for how in how_are_you):
            return "ผมสบายดีครับ ขอบคุณที่ถาม คุณล่ะครับ เป็นอย่างไรบ้าง?"
            
        # ตรวจสอบคำขอบคุณ
        if any(thank in user_input_lower for thank in thanks):
            return "ยินดีครับ หากมีอะไรให้ช่วยเหลืออีก ก็บอกได้นะครับ"
            
        # ตรวจสอบคำลา
        if any(bye in user_input_lower for bye in goodbyes):
            return "ลาก่อนครับ ขอให้มีวันที่ดีนะครับ"
        
        # ตรวจสอบคำถามทั่วไปจากดิกชันนารี
        for question, answer in common_qas.items():
            if question in user_input_lower:
                return answer
                
        # ถ้าไม่เข้าเงื่อนไขใดๆ ให้ส่งไปที่โมเดล แต่ปรับแต่งบริบทให้ดีขึ้น
        try:
            # เพิ่มบริบทเพื่อให้โมเดลตอบสมเหตุสมผล
            prompt = f"คำถาม: {user_input}\nคำตอบที่สั้นกระชับและตรงประเด็น: "
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # ปิดการแสดงเตือนขณะสร้างข้อความ
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    output = self.model.generate(
                        **inputs, 
                        max_length=max_length, 
                        do_sample=True, 
                        temperature=self.temperature,
                        top_p=0.9,
                        top_k=50,
                        repetition_penalty=1.2,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
            
            # ถอดรหัสและตัดข้อความของผู้ใช้ออก
            full_response = self.tokenizer.decode(output[0], skip_special_tokens=True)
            
            # ตัดข้อความ prompt ออกจากการตอบกลับ
            if prompt in full_response:
                response = full_response[len(prompt):].strip()
            else:
                # ถ้าหาไม่เจอโดยตรง พยายามหาส่วนของคำตอบ
                response_marker = "คำตอบที่สั้นกระชับและตรงประเด็น: "
                if response_marker in full_response:
                    response = full_response.split(response_marker)[1].strip()
                else:
                    # ถ้ายังหาไม่เจอ ตัดเฉพาะส่วนที่ไม่ใช่คำถาม
                    question_marker = f"คำถาม: {user_input}"
                    if question_marker in full_response:
                        response = full_response.split(question_marker)[1].strip()
                    else:
                        response = full_response.strip()
            
            # ตัดข้อความให้เป็นเฉพาะประโยคที่สมบูรณ์
            if len(response) > 10:  # ตรวจสอบว่ามีข้อความตอบกลับเพียงพอ
                # ตัดที่เครื่องหมายวรรคตอนหากประโยคไม่สมบูรณ์
                for end_marker in ['.', '!', '?', ':', '\n\n']:
                    last_idx = response.rfind(end_marker)
                    if last_idx > len(response) * 0.7:  # ตัดเฉพาะถ้าอยู่ในช่วงท้ายของข้อความ
                        response = response[:last_idx+1]
                        break
            
            # ตรวจสอบและแก้ไขกรณีคำตอบไม่สมเหตุสมผล
            if len(response) < 5 or not any(char.isalpha() for char in response):
                return "ขออภัย ฉันไม่เข้าใจคำถามนั้น คุณช่วยถามใหม่หรืออธิบายเพิ่มเติมได้ไหมครับ?"
                
            return response
        
        except Exception as e:
            return f"เกิดข้อผิดพลาดในการสร้างข้อความตอบกลับ: {str(e)}"

    def chat(self):
        print("\n" + "="*60)
        print("🤖 DeepSeek Chatbot พร้อมให้บริการ")
        print("💬 พิมพ์ 'exit', 'ออก' หรือ 'quit' เพื่อจบการสนทนา")
        print("⚙️  พิมพ์ '/help' เพื่อดูคำสั่งที่ใช้ได้")
        print("="*60 + "\n")
        
        while True:
            user_input = input("👤 คุณ: ").strip()
            
            # ตรวจสอบคำสั่งพิเศษ
            if user_input.lower() in ["exit", "ออก", "จบ", "quit", "bye"]:
                print("\n👋 ขอบคุณที่ใช้งาน DeepSeek Chatbot")
                break
                
            elif user_input.lower() == "/help":
                print("\n🔍 คำสั่งที่ใช้ได้:")
                print("  exit, ออก, quit - ออกจากโปรแกรม")
                print("  /clear - ล้างประวัติการสนทนา")
                print("  /temp [0.1-1.0] - ปรับค่า temperature (เช่น /temp 0.8)")
                print("  /help - แสดงคำสั่งที่ใช้ได้\n")
                continue
                
            elif user_input.lower() == "/clear":
                self.history = []
                print("\n🧹 ล้างประวัติการสนทนาแล้ว\n")
                continue
                
            elif user_input.lower().startswith("/temp "):
                try:
                    temp = float(user_input.split()[1])
                    if 0.1 <= temp <= 1.0:
                        self.temperature = temp
                        print(f"\n🌡️ ปรับค่า temperature เป็น {temp} แล้ว\n")
                    else:
                        print("\n⚠️ ค่า temperature ต้องอยู่ระหว่าง 0.1-1.0\n")
                except:
                    print("\n⚠️ รูปแบบคำสั่งไม่ถูกต้อง ใช้รูปแบบ: /temp 0.7\n")
                continue
                
            elif not user_input:
                continue

            # แสดงอนิเมชันคิด (ใช้ฟังก์ชันจากด้านนอก)
            print()  # ขึ้นบรรทัดใหม่ก่อนแสดงอนิเมชัน
            
            # เริ่มอนิเมชันและรับฟังก์ชันสำหรับหยุดอนิเมชัน
            stop_animation = show_thinking_animation()
            
            try:
                # สร้างคำตอบ
                response = self.generate_response(user_input)
            finally:
                # หยุดอนิเมชันไม่ว่าจะเกิดอะไรขึ้น
                stop_animation()
            
            # แสดงคำตอบ
            print(f"🤖 DeepSeek: {response}\n")
            
            # เก็บประวัติการสนทนา
            self.history.append({"user": user_input, "bot": response})

# สร้างอินสแตนซ์และเริ่มการสนทนา
if __name__ == "__main__":
    chatbot = DeepSeekChat(model, tokenizer, device)
    chatbot.chat()
