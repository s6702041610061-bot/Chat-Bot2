# แชทบอทหลักสูตรเทคโนโลยีคอมพิวเตอร์ พ.ศ. 2569

ใช้ `TCT69_Curriculum.md` ทั้งฉบับเป็นฐานความรู้ แทนข้อมูลแรงงานใน Excel
ค้นจาก Gemini embedding และ BM25 (คำสำคัญ/ตัวเลข/รหัสวิชา) รวมอันดับด้วย RRF
รายการใน `synonyms.json` ใช้ขยายคำค้น ไม่ใช่แหล่งข้อเท็จจริง
ประวัติสนทนาใช้เติมบริบทคำถามต่อเนื่อง แล้วค้นหลักฐานใหม่ทุกครั้ง

## Query vector cache และการค้นสำรอง

คำถามหลังตีความจะผ่านการ normalize และจับ alias ก่อนค้น seed vector ที่จัดส่ง
พร้อมแอป: 50 คำถามมาตรฐาน รวม 100 รูปแบบ อยู่ใน `seed_questions.json`
และ `embeddings/query_embeddings.npz` พร้อม manifest `query_metadata.json`
สร้างหรืออัปเดต seed ด้วย `python workaw/rag.py --build-query-cache` โดยใช้
Gemini แบบ batch 16 ข้อ เก็บแต่ละ batch เพื่อทำต่อได้เมื่อบริการขัดข้อง
คำสั่งนี้มีค่าใช้จ่าย API แต่การเปิดแอปหรือ deploy ไม่เรียกสร้าง seed อัตโนมัติ

คำถามใหม่เก็บผ่าน `st.cache_data(persist="disk", max_entries=2000)` โดย key
ประกอบด้วยรุ่น cache โมเดล มิติ task และข้อความหลังขยาย synonyms ไม่รวมคีย์ API
เปลี่ยนคีย์จึงใช้ vector เดิมต่อได้ ผลล้มเหลวไม่ถูก cache คำถามคนละภาคเรียน
และเงื่อนไขไม่ถูกรวมเข้าด้วยกัน การแก้ synonyms ทำให้ข้อความ embedding ใหม่
ไม่ตรง cache เดิม และ seed เก่าจะถูกข้ามจนสร้างใหม่

บน Streamlit Cloud runtime cache อาจหายเมื่อ redeploy/ย้าย instance แต่ seed
ที่ commit พร้อมแอปยังอยู่ ขีดจำกัด 2000 รายการใช้กับ cache ในหน่วยความจำ
ไม่ใช่การรับประกันขนาดไฟล์ cache บนดิสก์ ไม่มีการเก็บประวัติผู้ใช้ใน seed

เมื่อ cache miss เรียก Gemini เพียงครั้งเดียว ตั้ง timeout 4 วินาทีและปิด retry
ถ้าโควตาหมด คีย์ผิด เครือข่ายล้มเหลว หรือ vector ไม่สมบูรณ์ จะค้นด้วย BM25 ต่อ
DeepSeek ยังอ่านหลักฐานใหม่ทุกครั้ง ไม่ cache คำตอบ และไม่แต่งข้อมูลเมื่อค้นไม่พบ
การไม่มี Gemini key ไม่ปิดแชต ส่วน DeepSeek key ยังต้องตั้งค่าใน Cloud Secrets
หรือ `.env` Log แสดง source (`seed`, `runtime`, `gemini`, `bm25_fallback`) และเวลา
โดยไม่แสดงคีย์หรือข้อความคำถาม

ตรวจสอบการตอบจริงโดยไม่เรียก Gemini ได้ด้วย
`python workaw/tests/verify_live.py --no-embedding` (ยังเรียก DeepSeek และใช้เครดิต)

## เริ่มใช้งาน

เปิด `run_local.bat` จากโฟลเดอร์โครงการ ซึ่งใช้ environment `worker` เดิม
หรือติดตั้ง dependencies และเปิดจาก terminal:

```powershell
python -m pip install -r workaw/requirements.txt
python -m streamlit run workaw/app.py
```

สร้างคีย์ในหน้า API keys ของ DeepSeek Platform แล้วตั้งค่าใน `workaw/.env`
ตาม `.env.example` (เก็บคีย์จริงไว้ในเครื่อง):

```dotenv
DEEPSEEK_API_KEY=ใส่คีย์_DeepSeek_ที่นี่
DEEPSEEK_MODEL=deepseek-flash
GEMINI_API_KEY=คีย์_Gemini_เดิม
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
```

DeepSeek ตีความคำถามต่อเนื่องและตอบจากหลักฐาน โดยเรียก
`https://api.deepseek.com/chat/completions` ปิด thinking และใช้ JSON Output
คำถามทั่วไปเรียกสร้างคำตอบหนึ่งครั้ง ส่วนคำถามต่อเนื่องอาจเรียกตีความเพิ่มอีกหนึ่งครั้ง
Gemini ใช้เฉพาะ embedding เอกสารและ embedding คำถามสำหรับค้นข้อมูล
`GEMINI_EMBEDDING_MODEL` ค่าเริ่มต้นคือ
`gemini-embedding-001` ขนาด 768 มิติ ต้องใช้โมเดลเดียวกันกับการค้นคำถาม
การสร้าง embedding จะส่งข้อความเอกสารให้ Gemini API
การเปลี่ยนคีย์ DeepSeek ไม่ทำให้ดัชนี embedding ถูกสร้างใหม่
`GEMINI_MODEL` และ `GEMINI_FALLBACK_MODEL` ไม่ใช้สำหรับตอบคำถามแล้ว
หาก DeepSeek แจ้งเครดิตไม่เพียงพอ ให้ตรวจยอดใน DeepSeek Platform
หลังแก้ `.env` ให้หยุดและเปิดแชทบอทใหม่

สำหรับตัวเปิด `run_local.bat` ให้ติดตั้ง dependencies ใน environment ที่ใช้จริง:

```bat
conda activate worker
cd /d "C:\Users\Phoow\Documents\workaw_chatbot"
python -m pip install -r workaw\requirements.txt
python -m streamlit run workaw\app.py
```

## ฐานความรู้และ embedding

สร้างล่วงหน้าได้ด้วย:

```powershell
python workaw/rag.py
python workaw/rag.py --query "เรียนทั้งหมดกี่หน่วยกิต"
```

เก็บเวกเตอร์และ metadata สำหรับรันจริงใน `workaw/.rag_cache/` โดยใช้ SHA-256 ของเอกสาร
ชื่อโมเดล ขนาดเวกเตอร์ และรุ่น parser เป็นตัวระบุ cache
เปลี่ยนหลักสูตรหรือโมเดลแล้วระบบจะสร้าง index ใหม่เอง
การเปลี่ยนคำพ้องจะสร้างดัชนีคำสำคัญใหม่โดยไม่ต้องสร้าง embedding เอกสารซ้ำ
เก็บแต่ละ batch เพื่อทำต่อได้เมื่อเครือข่ายหรือโควตาขัดข้อง
ค่าเริ่มต้นเว้น `RAG_EMBED_BATCH_INTERVAL=13` วินาทีต่อ batch เพื่อรองรับโควตา
ปรับลดได้เมื่อบัญชีมีโควตาสูงกว่าเดิม การค้นคำถามไม่ต้องรอช่วงเวลานี้
ห้ามส่ง `.env` หรือ commit cache เข้าระบบควบคุมรุ่น

ไฟล์ที่ตรวจสอบได้โดยตรงคือ `workaw/embeddings/curriculum_embeddings.npz`
และ `workaw/embeddings/curriculum_metadata.json` ซึ่งเป็นสำเนาที่ระบบอัปเดต
จาก cache ล่าสุด ไฟล์เวกเตอร์ปัจจุบันมี 629 แถว × 768 มิติ และ metadata เก็บข้อความ
แต่ละส่วน เลขหน้า หมวด และ SHA-256 ของเอกสารต้นทาง
หาก `.rag_cache` หาย ระบบกู้ดัชนีจากไฟล์สองไฟล์นี้เมื่อ SHA-256 โมเดล ขนาดเวกเตอร์
รุ่น parser และรายการข้อความตรงกัน โดยไม่เรียกสร้าง embedding เอกสารซ้ำ

## โครงสร้าง

- `app.py`: หน้าแชท การตั้งค่า session และการแสดงแหล่งอ้างอิง
- `chatbot.py`: เติมบริบทคำถามต่อเนื่องและสร้างคำตอบพร้อมรหัสแหล่งอ้างอิง
- `rag.py`: อ่านเอกสาร แบ่งข้อมูล สร้าง/โหลด embedding และค้นแบบผสม
- `prompt.py`: กติกาตอบจากหลักฐานและกติกาตีความคำถาม
- `synonyms.json`: คำย่อ คำพ้อง และคำค้นที่เกี่ยวข้อง

คำตอบไม่แสดงรหัสภายใน เช่น `[S1]` และมีแถบดูแหล่งอ้างอิงที่ระบุชื่อไฟล์ หัวข้อ
หน้าในเล่ม หน้า PDF และข้อความหลักฐานจริง
ไม่ใช้ปกและสารบัญเป็นหลักฐาน และไม่ค้นข้อมูลเปรียบเทียบ 2564/2569
สำหรับคำถามทั่วไปจนกว่าผู้ใช้จะถามถึงฉบับเดิมหรือการเปรียบเทียบ
เอกสารแนบและประวัติสนทนาเป็นข้อมูล ไม่ใช่คำสั่งของระบบ

ข้อจำกัด: ภาพในโฟลเดอร์ assets ยังไม่ถูกแปลงเป็น embedding ของภาพ
ระบบค้นจากข้อความที่อยู่ใน Markdown รวมถึง OCR ที่ถอดไว้เท่านั้น
ระเบียบและประกาศที่แนบเป็นข้อมูลตามฉบับเอกสาร ไม่ใช่การยืนยันสถานะปัจจุบัน

## ตรวจสอบ

```powershell
python -m unittest discover -s workaw/tests -v
```

เมื่อใส่คีย์ทั้งสองบริการและมีเครดิต DeepSeek แล้ว ตรวจคำถามทั่วไปและคำถามต่อเนื่องได้ด้วย:

```powershell
python workaw/tests/verify_live.py --smoke
```

การตรวจสอบจริงเรียก API และคิดค่าใช้จ่ายตามการใช้งาน ส่วนชุด unittest ใช้บริการจำลอง

ตรวจแบบเรียก API จริงหลังสร้าง index ครบด้วย `python workaw/tests/verify_live.py`
ซึ่งตรวจหน่วยกิต คำถามต่อเนื่อง แผนเรียน คุณสมบัติผู้สมัคร การเปลี่ยนเรื่อง
และคำถามที่ไม่มีข้อมูลในเอกสาร การตรวจนี้ใช้โควตา API
