# Event Extraction using LLMs (GPT-4o, Qwen 2.5, Llama 3)
# Doc jsonl cau, goi LLM theo tung batch de gan nhan FinTech theme + event, luu ra JSONL.

import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

from openai import OpenAI

# =================================
# Cau hinh
# =================================
INPUT_FILE = Path("data.jsonl")
OUTPUT_FILE = Path("preannotated_events.jsonl")
CHECKPOINT_FILE = Path("llm_checkpoint_ids.json")

# DEMO: SAMPLE_SIZE = 50 -> chi chay 50 cau de test.
# PRODUCTION: doi SAMPLE_SIZE = None -> chay toan bo file (~50k dong), logic giu nguyen.
SAMPLE_SIZE: Optional[int] = 50
RANDOM_SEED = 42
BATCH_SIZE = 10
MAX_RETRIES = 3

MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")
BASE_URL = os.environ.get("OPENAI_BASE_URL")
API_KEY = os.environ.get("OPENAI_API_KEY")

REQUIRED_FIELDS = {"sentence_id", "sentence", "is_fintech", "fintech_themes", "events"}

SYSTEM_INSTRUCTION = """
Ban la chuyen gia NLP phu trach gan nhan su kien va chu de Tai chinh Cong nghe (FinTech) trong tin tuc tieng Viet.

Nhiem vu:
1. Xac dinh cau co lien quan den FinTech hay khong.
2. Xac dinh mot hoac nhieu linh vuc FinTech duoc de cap.
3. Xac dinh mot hoac nhieu su kien duoc the hien trong cau.
4. Xac dinh trigger cua tung su kien.
5. Gan moi su kien vao linh vuc FinTech tuong ung.

1. FINTECH THEMES
- "Banking": Ngan hang, tai khoan, tien gui, tiet kiem, the, ngan hang so, challenger/neobank va ha tang ngan hang.
- "Crowdfunding": Goi von cong dong qua nen tang truc tuyen.
- "Digital Assets": Crypto, token, stablecoin, NFT, RWA va cac dich vu tai chinh lay blockchain/DLT lam trong tam.
- "Insurance": Bao hiem so, phan phoi bao hiem, tham dinh, boi thuong va cac nen tang InsurTech.
- "Investment": Dau tu va giao dich truc tiep, moi gioi chung khoan, nen tang giao dich, VC va PE.
- "Lending": Khoan vay va tin dung, P2P/marketplace lending, BNPL, embedded credit va cac giai phap ho tro tin dung.
- "Payments": Thanh toan va chuyen tien, vi dien tu, payment gateway, QR/NFC, POS, remittance va thanh toan xuyen bien gioi.
- "Wealth Management": Quan ly tai san, tai chinh ca nhan, hoach dinh tai chinh, robo-advisory va toi uu danh muc.
- "Other": Cac linh vuc FinTech khac khong thuoc cac nhom tren, nhu RegTech va SupTech.

2. EVENT TYPES
- "Funding": Huy dong, nhan hoac rot von; dau tu, IPO, trai phieu va tai tro.
- "Partnership": Hop tac, lien minh, lien doanh, MOU, phan phoi, tich hop API hoac ket noi nen tang.
- "Merger/Acquisition": Sap nhap, mua lai, ban tai san, thoai von hoac thay doi quyen so huu/kiem soat.
- "Product/Service/Technology": Ra mat, cap nhat, mo rong, trien khai, nang cap hoac ngung san pham, dich vu va cong nghe.
- "Regulating": Quy dinh, chinh sach, sandbox, cap/thu hoi giay phep, xu phat va thu tuc phap ly.
- "Business restructuring": Thay doi lanh dao, nhan su, co cau to chuc, thuong hieu, bao cao tai chinh, giai the hoac pha san.
- "Recognizing & Branding": Marketing, quang cao, giai thuong, chung nhan, tai tro, ESG/CSR.
- "Disrupting": Su co he thong, gian doan dich vu, tan cong mang, ro ri du lieu, gian lan, vo no hoac no xau tang manh.

3. QUY TAC
- "is_fintech" = true neu cau thuoc it nhat 1 FinTech Theme, nguoc lai = false.
- Neu is_fintech = false -> fintech_themes: [] va events: [].
- Co the gan nhieu themes/events cho 1 cau. Chi gan event khi cau thuc su the hien su kien.
- "evidence" va "trigger" phai la doan van ban trich nguyen van tu cau, khong dich/sua/viet lai.
- Chi dung cac nhan duoc liet ke o tren.

4. OUTPUT
Tra ve DUY NHAT 1 JSON array, moi phan tu ung voi dung 1 cau dau vao, theo dung thu tu:
{"sentence_id": "", "sentence": "", "is_fintech": true/false,
 "fintech_themes": [{"theme": "", "evidence": ""}],
 "events": [{"event_type": "", "trigger": "", "theme": ""}]}
Khong them giai thich hay van ban ngoai JSON.
"""


def load_data(jsonl_path: Path, checkpoint_path: Path, n: Optional[int] = None) -> List[Dict]:
    """Doc jsonl, bo qua id da co trong checkpoint, tuy chon random sample n dong (demo)."""
    if checkpoint_path.exists():
        done_ids = set(json.loads(checkpoint_path.read_text(encoding="utf-8")))
    else:
        done_ids = set()

    data = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    data = [item for item in data if item.get("id") not in done_ids]

    if n is not None:
        data = random.Random(RANDOM_SEED).sample(data, min(n, len(data)))
    return data


def make_batches(data: List[Dict], batch_size: int = BATCH_SIZE):
    """Chia data thanh cac batch nho de goi LLM tung batch."""
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


def build_prompt(batch: List[Dict]) -> str:
    """Ghep cac cau trong batch thanh prompt, moi cau kem sentence_id."""
    sentences = [{"sentence_id": item["id"], "sentence": item["snippet"]} for item in batch]
    return json.dumps(sentences, ensure_ascii=False)


def call_llm(client: OpenAI, prompt: str) -> List[Dict]:
    """Goi LLM, yeu cau tra ve JSON array. Retry don gian neu loi."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            completion = client.chat.completions.create(
                model=MODEL,
                temperature=0,
                messages=[
                    {"role": "system", "content": SYSTEM_INSTRUCTION},
                    {"role": "user", "content": prompt},
                ],
            )
            return json.loads(completion.choices[0].message.content)
        except Exception as e:
            print(f"  [retry {attempt}/{MAX_RETRIES}] call_llm error: {e}")
            if attempt == MAX_RETRIES:
                raise
            time.sleep(2 * attempt)


def validate_result(result: List[Dict], batch: List[Dict]) -> List[Dict]:
    """Kiem tra: la list, du so luong, id khop input, khong trung, du field bat buoc."""
    input_ids = {str(item["id"]) for item in batch}
    result_ids = [str(r.get("sentence_id")) for r in result]

    assert isinstance(result, list), "Ket qua khong phai JSON array"
    assert len(result) == len(batch), f"So luong ket qua ({len(result)}) != so cau ({len(batch)})"
    assert len(set(result_ids)) == len(result_ids), "sentence_id bi trung"
    assert set(result_ids) == input_ids, f"sentence_id khong khop: {input_ids ^ set(result_ids)}"
    for r in result:
        assert REQUIRED_FIELDS <= r.keys(), f"Thieu field o sentence_id={r.get('sentence_id')}"
    return result


def save_results(results: List[Dict], output_path: Path = OUTPUT_FILE) -> None:
    """Append ket qua da validate vao file JSONL output."""
    with open(output_path, "a", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def update_checkpoint(new_ids: List, checkpoint_path: Path = CHECKPOINT_FILE) -> None:
    """Chi goi sau khi save_results thanh cong, de danh dau cac id da xu ly."""
    done_ids = set(json.loads(checkpoint_path.read_text(encoding="utf-8"))) if checkpoint_path.exists() else set()
    done_ids.update(str(i) for i in new_ids)
    checkpoint_path.write_text(json.dumps(sorted(done_ids), ensure_ascii=False), encoding="utf-8")


def main(sample_size: Optional[int] = SAMPLE_SIZE) -> None:
    """sample_size=50 -> demo; sample_size=None -> chay toan bo file con lai."""
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    data = load_data(INPUT_FILE, CHECKPOINT_FILE, n=sample_size)
    print(f"Se xu ly {len(data)} cau (sample_size={sample_size})")

    for i, batch in enumerate(make_batches(data), start=1):
        try:
            result = call_llm(client, build_prompt(batch))
            result = validate_result(result, batch)
            save_results(result)
            update_checkpoint([item["id"] for item in batch])
            print(f"Batch {i}: {len(batch)} cau - OK")
        except Exception as e:
            print(f"Batch {i}: {len(batch)} cau - FAILED: {e}")
            continue  # bo qua batch loi, khong checkpoint -> tu retry o lan chay sau


if __name__ == "__main__":
    main()
