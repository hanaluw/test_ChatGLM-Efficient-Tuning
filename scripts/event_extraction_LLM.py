# Event Extraction using LLMs (GPT-4o, Qwen 2.5, Llama 3)
# Get snippet_sentences_corrected.jsonl, extract events from the snippet field using an LLM.

# Import
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# =================================
# 0. Set up environment variables and OpenAI client
# =================================
INPUT_FILE = Path("data.jsonl")
OUTPUT_FILE = Path("preannotated_events.jsonl")
CHECKPOINT_FILE = Path("llm_checkpoint_ids.json")

# DEMO: chỉ chạy 50 dòng đầu để test. PRODUCTION: đặt SAMPLE_SIZE = None để chạy toàn bộ 50k dòng.
SAMPLE_SIZE: Optional[int] = 50
RANDOM_SEED = 42
BATCH_SIZE = 10

MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")
BASE_URL = os.environ.get("OPENAI_BASE_URL")  # optional: for OpenAI-compatible providers
API_KEY = os.environ.get("OPENAI_API_KEY")

MAX_RETRIES = 3
RETRY_DELAY = 3  # seconds, nhân đôi sau mỗi lần retry (exponential backoff)


# =================================
# 1. Load the snippet_sentences_corrected.jsonl file
# =================================
def load_data(
    jsonl_path: str,
    checkpoint_path: str,
    n: Optional[int] = None,
    seed: int = RANDOM_SEED,
) -> List[Dict]:
    """Đọc file jsonl, bỏ qua các id đã xử lý (checkpoint), tuỳ chọn lấy mẫu n dòng."""
    try:
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            checkpoint_ids = set(json.load(f))
    except FileNotFoundError:
        checkpoint_ids = set()

    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                if item.get("id") not in checkpoint_ids:
                    data.append(item)

    if n is not None:
        rng = random.Random(seed)
        data = rng.sample(data, min(n, len(data)))

    logger.info("Loaded %d unprocessed rows (sample_size=%s)", len(data), n)
    return data


# =================================
# 2. Batch the data into smaller chunks for processing
# =================================
def make_batches(data: List[Dict], batch_size: int = BATCH_SIZE) -> Iterator[List[Dict]]:
    """Chia data thành các batch nhỏ, mỗi batch batch_size câu, để build 1 request/batch."""
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]


# =================================
# 3.1. Event extraction schema / system instruction
# =================================
SYSTEM_INSTRUCTION = """
Bạn là chuyên gia NLP phụ trách gán nhãn sự kiện và chủ đề Tài chính Công nghệ (FinTech) trong tin tức tiếng Việt.

Nhiệm vụ:
1. Xác định câu có liên quan đến FinTech hay không.
2. Xác định một hoặc nhiều lĩnh vực FinTech được đề cập.
3. Xác định một hoặc nhiều sự kiện được thể hiện trong câu.
4. Xác định trigger của từng sự kiện.
5. Gán mỗi sự kiện vào lĩnh vực FinTech tương ứng.

1. FINTECH THEMES

- "Banking": Ngân hàng, tài khoản, tiền gửi, tiết kiệm, thẻ, ngân hàng số, challenger/neobank và hạ tầng ngân hàng.
- "Crowdfunding": Gọi vốn cộng đồng qua nền tảng trực tuyến.
- "Digital Assets": Crypto, token, stablecoin, NFT, RWA và các dịch vụ tài chính lấy blockchain/DLT làm trọng tâm.
- "Insurance": Bảo hiểm số, phân phối bảo hiểm, thẩm định, bồi thường và các nền tảng InsurTech.
- "Investment": Đầu tư và giao dịch trực tiếp, môi giới chứng khoán, nền tảng giao dịch, VC và PE.
- "Lending": Khoản vay và tín dụng, P2P/marketplace lending, BNPL, embedded credit và các giải pháp hỗ trợ tín dụng.
- "Payments": Thanh toán và chuyển tiền, ví điện tử, payment gateway, QR/NFC, POS, remittance và thanh toán xuyên biên giới.
- "Wealth Management": Quản lý tài sản, tài chính cá nhân, hoạch định tài chính, robo-advisory và tối ưu danh mục.
- "Other": Các lĩnh vực FinTech khác không thuộc các nhóm trên, như RegTech và SupTech.

BOUNDARIES:
- Dịch vụ thanh toán độc lập → "Payments"; tài khoản/tiền gửi ngân hàng → "Banking".
- P2P lending/marketplace lending → "Lending"; crowdfunding → "Crowdfunding".
- Crypto/blockchain-centric financial services → "Digital Assets".
- Thực hiện đầu tư/giao dịch → "Investment"; quản lý tài sản và hoạch định tài chính dài hạn → "Wealth Management".
- Khoản vay, tín dụng hoặc nợ → "Lending".

2. EVENT TYPES

- "Funding": Huy động, nhận hoặc rót vốn; đầu tư, IPO, trái phiếu và tài trợ.
- "Partnership": Hợp tác, liên minh, liên doanh, MOU, phân phối, tích hợp API hoặc kết nối nền tảng.
- "Merger/Acquisition": Sáp nhập, mua lại, bán tài sản, thoái vốn hoặc thay đổi quyền sở hữu/kiểm soát.
- "Product/Service/Technology": Ra mắt, cập nhật, mở rộng, triển khai, nâng cấp hoặc ngừng sản phẩm, dịch vụ và công nghệ.
- "Regulating": Quy định, chính sách, sandbox, cấp/thu hồi giấy phép, xử phạt và thủ tục pháp lý từ cơ quan có thẩm quyền.
- "Business restructuring": Thay đổi lãnh đạo, nhân sự, cơ cấu tổ chức, thương hiệu, báo cáo tài chính, giải thể hoặc phá sản.
- "Recognizing & Branding": Marketing, quảng cáo, giải thưởng, chứng nhận, tài trợ, ESG/CSR và các sự kiện ảnh hưởng đến nhận diện hoặc uy tín thương hiệu.
- "Disrupting": Sự cố hệ thống, gián đoạn dịch vụ, tấn công mạng, rò rỉ dữ liệu, gian lận, vỡ nợ hoặc nợ xấu tăng mạnh.

3. ANNOTATION RULES

- "is_fintech" = true nếu câu chứa nội dung thuộc ít nhất một FinTech Theme; ngược lại = false.
- Nếu "is_fintech" = false → "fintech_themes": [] và "events": [].
- Có thể gán nhiều FinTech Themes cho một câu.
- Chỉ gán event khi câu thực sự thể hiện một sự kiện, không suy diễn từ việc chỉ đề cập đến doanh nghiệp, sản phẩm, công nghệ hoặc lĩnh vực.
- Có thể gán nhiều events cho một câu.
- "evidence" phải là đoạn văn bản được trích nguyên văn từ câu làm căn cứ cho theme.
- "trigger" phải là từ/cụm từ xuất hiện nguyên văn trong câu và trực tiếp biểu thị event.
- "theme" trong event là FinTech Theme tương ứng trực tiếp với event đó.
- Không dịch, sửa, chuẩn hóa hoặc viết lại "evidence" và "trigger".
- Chỉ sử dụng các nhãn được liệt kê ở trên.

4. OUTPUT

Trả về một JSON array, mỗi phần tử tương ứng với đúng 1 câu trong batch đầu vào, theo đúng thứ tự, có dạng:
{
    "sentence_id": "",
    "sentence": "",
    "is_fintech": true/false,
    "fintech_themes": [
        {"theme": "", "evidence": ""}
    ],
    "events": [
        {"event_type": "", "trigger": "", "theme": ""}
    ]
}

Trả về duy nhất JSON hợp lệ (JSON array). Không thêm giải thích, nhận xét hoặc văn bản bên ngoài JSON.
"""


def build_prompt(batch: List[Dict]) -> str:
    """Ghép các câu trong batch thành 1 prompt: mỗi câu kèm sentence_id để model bám theo khi trả lời."""
    lines = [
        json.dumps({"sentence_id": item["id"], "sentence": item["snippet"]}, ensure_ascii=False)
        for item in batch
    ]
    return "Hãy gán nhãn cho các câu sau (mỗi dòng là 1 câu JSON):\n" + "\n".join(lines)


# =================================
# 3.2. Call the LLM API to extract events from the snippet field
# =================================
def call_llm(client: OpenAI, prompt: str) -> List[Dict]:
    """Gọi LLM với 1 batch prompt, yêu cầu JSON array, retry với exponential backoff nếu lỗi/timeout."""
    last_error = None
    delay = RETRY_DELAY
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            completion = client.chat.completions.create(
                model=MODEL,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_INSTRUCTION},
                    {"role": "user", "content": prompt},
                ],
            )
            text = completion.choices[0].message.content
            parsed = json.loads(text)
            # Model có thể bọc array trong 1 key, ví dụ {"results": [...]}
            if isinstance(parsed, dict):
                parsed = next((v for v in parsed.values() if isinstance(v, list)), [])
            return parsed
        except (json.JSONDecodeError, Exception) as e:  # noqa: BLE001
            last_error = e
            logger.warning("call_llm attempt %d/%d failed: %s", attempt, MAX_RETRIES, e)
            if attempt < MAX_RETRIES:
                time.sleep(delay)
                delay *= 2

    raise RuntimeError(f"call_llm failed after {MAX_RETRIES} attempts: {last_error}")


# =================================
# 3.3. Validate LLM output
# =================================
REQUIRED_FIELDS = {"sentence_id", "sentence", "is_fintech", "fintech_themes", "events"}


def validate_result(result: List[Dict], input_batch: List[Dict]) -> List[Dict]:
    """Kiểm tra: JSON hợp lệ, đủ số lượng, id khớp input, đủ field bắt buộc, không trùng id."""
    if not isinstance(result, list):
        raise ValueError("Kết quả trả về không phải là JSON array")

    input_ids = [str(item["id"]) for item in input_batch]
    result_ids = [str(r.get("sentence_id")) for r in result]

    if len(result) != len(input_batch):
        raise ValueError(f"Số lượng kết quả ({len(result)}) khác số câu đầu vào ({len(input_batch)})")

    if len(set(result_ids)) != len(result_ids):
        raise ValueError("Kết quả có sentence_id bị trùng lặp")

    if set(result_ids) != set(input_ids):
        raise ValueError(f"sentence_id không khớp input. Thiếu/lệch: {set(input_ids) ^ set(result_ids)}")

    for r in result:
        missing = REQUIRED_FIELDS - r.keys()
        if missing:
            raise ValueError(f"Kết quả sentence_id={r.get('sentence_id')} thiếu field: {missing}")

    return result


# =================================
# 3.4. Save the extracted events to a new JSONL file
# =================================
def save_results(results: List[Dict], output_path: Path = OUTPUT_FILE) -> None:
    """Append kết quả đã validate vào file JSONL output (mỗi dòng 1 record)."""
    with open(output_path, "a", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def update_checkpoint(new_ids: List[Any], checkpoint_path: Path = CHECKPOINT_FILE) -> None:
    """Chỉ được gọi SAU KHI save_results thành công: cập nhật danh sách id đã xử lý."""
    try:
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            processed_ids = set(json.load(f))
    except FileNotFoundError:
        processed_ids = set()

    processed_ids.update(str(i) for i in new_ids)

    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(sorted(processed_ids), f, ensure_ascii=False)


# =================================
# 4. Main function
# =================================
def main(sample_size: Optional[int] = SAMPLE_SIZE) -> None:
    """
    sample_size=50 (mặc định)  -> chế độ DEMO, test nhanh trên 50 câu.
    sample_size=None           -> chế độ PRODUCTION, chạy toàn bộ dữ liệu còn lại (không nằm trong checkpoint).
    Chỉ cần đổi tham số này (hoặc SAMPLE_SIZE ở đầu file) để chuyển demo <-> full run,
    phần logic (load, batch, call LLM, validate, save, checkpoint) giữ nguyên.
    """
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

    data = load_data(str(INPUT_FILE), str(CHECKPOINT_FILE), n=sample_size, seed=RANDOM_SEED)
    if not data:
        logger.info("Không còn dòng nào cần xử lý.")
        return

    total_batches = (len(data) + BATCH_SIZE - 1) // BATCH_SIZE
    for batch_num, batch in enumerate(make_batches(data, BATCH_SIZE), start=1):
        start_time = time.time()
        prompt = build_prompt(batch)
        try:
            raw_result = call_llm(client, prompt)
            validated = validate_result(raw_result, batch)
            save_results(validated)
            update_checkpoint([item["id"] for item in batch])
            elapsed = time.time() - start_time
            logger.info(
                "Batch %d/%d: %d câu, OK, %.1fs",
                batch_num, total_batches, len(batch), elapsed,
            )
        except Exception as e:  # noqa: BLE001
            elapsed = time.time() - start_time
            logger.error(
                "Batch %d/%d: %d câu, FAILED (%.1fs): %s",
                batch_num, total_batches, len(batch), elapsed, e,
            )
            # Batch lỗi bị bỏ qua, không cập nhật checkpoint -> sẽ được thử lại ở lần chạy sau.
            continue


if __name__ == "__main__":
    main()
