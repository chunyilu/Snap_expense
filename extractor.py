"""收據影像提取與 OpenAI GPT-4o-mini Vision API 串接模組。

負責將上傳之影像壓縮（長邊 2048px 以內）、轉為 Base64 編碼，
並透過 OpenAI Structured Outputs (`client.beta.chat.completions.parse`)
解析為型別安全的 `ReceiptSchema` 實例。同時提供離線/展示用之 Mock Data 產生器。
"""

import base64
import io
import os
from datetime import date
from typing import Optional, Tuple, Union

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image, ImageOps

from models import AnomalyReport, ExpenseCategory, ExpenseItem, ReceiptSchema

load_dotenv()

MAX_IMAGE_DIMENSION = 2048

SYSTEM_PROMPT = """你是一位專業的財務收據與統一發票 OCR 結構化提取專家。
請仔細檢視使用者提供的收據或發票影像，並精確提取以下資訊：
1. merchant_name: 店家、餐廳或開立機構名稱。若影像模糊無法辨識全名，請提取可辨識之部分或填寫「未知商家」。
2. invoice_number: 發票號碼（例如台灣統一發票格式 AB-12345678 或 AB12345678，若一般收據無發票號碼則設為 null）。
3. transaction_date: 交易日期 (YYYY-MM-DD)。若為民國年（如 115 年），請加上 1911 換算為西元年（如 2026 年）。
4. currency: 交易幣別三碼代碼（如 TWD, JPY, USD, EUR, HKD, KRW）。台灣發票預設為 TWD。
5. items: 購買品項列表，每項包含：
   - name: 商品或服務名稱
   - unit_price: 單價
   - quantity: 數量（整數，預設 1）
   - total_price: 該品項小計金額
   - category: 根據商品性質自動歸類至對應之 ExpenseCategory（餐飲食品、交通運輸、休閒娛樂、水電與通訊、生活雜貨、醫療保健、辦公文具、其他支出）
6. tax_amount: 稅額（若為內含稅或未單獨列出稅額請填 0）。
7. total_amount: 收據總計應付金額。
8. confidence_score: 你對此張收據字跡清晰度與欄位提取完整度的信心分數（0.0 至 1.0）。
"""


def preprocess_and_encode_image(
    image_source: Union[bytes, io.BytesIO, Image.Image],
    max_dim: int = MAX_IMAGE_DIMENSION,
) -> Tuple[str, Image.Image]:
    """將輸入影像修正 EXIF 方向、壓縮至長邊 2048px 以內，並轉為 JPEG Base64 字串。

    Args:
        image_source: 原始圖片位元組、BytesIO 或 PIL Image 物件。
        max_dim: 長邊最大像素限制，預設為 2048。

    Returns:
        Tuple[str, Image.Image]: (Base64 編碼字串, 處理後的 PIL Image 物件)。
    """
    if isinstance(image_source, bytes):
        img = Image.open(io.BytesIO(image_source))
    elif isinstance(image_source, io.BytesIO):
        image_source.seek(0)
        img = Image.open(image_source)
    elif isinstance(image_source, Image.Image):
        img = image_source.copy()
    else:
        raise TypeError("不支援的影像輸入格式，請提供 bytes、BytesIO 或 PIL.Image。")

    # 修正手機拍攝可能帶有的 EXIF 旋轉資訊
    img = ImageOps.exif_transpose(img)

    # 轉換為 RGB 以相容 JPEG 輸出（處理 PNG 透明背景或 RGBA/P 模式）
    if img.mode not in ("RGB", "L"):
        background = Image.new("RGB", img.size, (255, 255, 255))
        if "A" in img.mode:
            background.paste(img, mask=img.split()[-1])
        else:
            background.paste(img)
        img = background
    elif img.mode == "L":
        img = img.convert("RGB")

    # 若長邊超過 max_dim，等比例縮小
    width, height = img.size
    if max(width, height) > max_dim:
        scale = max_dim / float(max(width, height))
        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        img = img.resize(new_size, Image.Resampling.LANCZOS)

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=88, optimize=True)
    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return b64_str, img


def extract_receipt_from_image(
    image_source: Union[bytes, io.BytesIO, Image.Image],
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> ReceiptSchema:
    """呼叫 OpenAI GPT-4o-mini Vision API 並以 Structured Outputs 解析收據影像。

    Args:
        image_source: 收據影像來源 (bytes, BytesIO, PIL.Image)。
        api_key: OpenAI API Key（若未傳入則讀取環境變數 OPENAI_API_KEY）。
        model: 使用之 OpenAI 模型名稱，預設為 'gpt-4o-mini'。

    Returns:
        ReceiptSchema: 通過 Pydantic v2 驗證之收據結構化資料。

    Raises:
        ValueError: 當未設定 API Key 或模型無法解析影像內容時拋出。
        RuntimeError: 當 OpenAI API 呼叫發生網路或服務錯誤時拋出。
    """
    resolved_key = (api_key or os.getenv("OPENAI_API_KEY") or "").strip()
    if not resolved_key or resolved_key == "your_openai_api_key_here":
        raise ValueError("尚未設定有效的 OPENAI_API_KEY。請在側邊欄輸入 API Key、設定 .env 檔案，或使用「載入範例收據 (Mock Data)」功能。")

    resolved_model = (model or os.getenv("OPENAI_MODEL") or "gpt-4o-mini").strip()
    b64_image, _ = preprocess_and_encode_image(image_source)

    client = OpenAI(api_key=resolved_key)
    try:
        completion = client.beta.chat.completions.parse(
            model=resolved_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "請提取這張收據/發票的所有欄位與品項明細，並嚴格遵守結構化輸出格式。",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64_image}",
                                "detail": "high",
                            },
                        },
                    ],
                },
            ],
            response_format=ReceiptSchema,
            temperature=0.1,
        )
    except Exception as exc:
        raise RuntimeError(f"OpenAI Vision API 呼叫失敗：{exc}") from exc

    message = completion.choices[0].message
    if message.refusal:
        raise ValueError(f"模型拒絕解析此影像：{message.refusal}")
    if message.parsed is None:
        raise ValueError("模型未能回傳符合 ReceiptSchema 的結構化資料。")

    return message.parsed


def get_mock_receipt(preset: str = "standard") -> ReceiptSchema:
    """產生展示與測試用的範例收據資料，當無 API Key 時可流暢體驗完整流程。

    Args:
        preset: 預設情境名稱：
            - 'standard': 一般台灣超商/餐廳消費收據 (TWD)
            - 'high_value': 含單筆大額支出（> 3,000 TWD）與高總額之收據
            - 'foreign_jpy': 日本旅遊外幣消費收據 (JPY)，觸發自動匯率換算
            - 'math_mismatch': 品項加總與收據總額不一致之異常收據

    Returns:
        ReceiptSchema: 模擬之收據物件。
    """
    today = date.today()

    if preset == "high_value":
        return ReceiptSchema(
            merchant_name="台灣高鐵 & 商務餐敘 (範例-大額警示)",
            invoice_number="AB-88291045",
            transaction_date=today,
            currency="TWD",
            items=[
                ExpenseItem(
                    name="高鐵商務車廂來回票 (台北-左營)",
                    unit_price=2440.0,
                    quantity=2,
                    total_price=4880.0,
                    category=ExpenseCategory.TRANSPORT,
                ),
                ExpenseItem(
                    name="頂級和牛商務套餐",
                    unit_price=3680.0,
                    quantity=1,
                    total_price=3680.0,
                    category=ExpenseCategory.MEALS,
                ),
            ],
            tax_amount=0.0,
            total_amount=8560.0,
            confidence_score=0.96,
        )

    if preset == "foreign_jpy":
        return ReceiptSchema(
            merchant_name="BicCamera 有樂町店 (範例-外幣換算)",
            invoice_number="JP-20261110",
            transaction_date=today,
            currency="JPY",
            items=[
                ExpenseItem(
                    name="人體工學無線滑鼠",
                    unit_price=8800.0,
                    quantity=1,
                    total_price=8800.0,
                    category=ExpenseCategory.OFFICE,
                ),
                ExpenseItem(
                    name="藍牙降噪耳機",
                    unit_price=24500.0,
                    quantity=1,
                    total_price=24500.0,
                    category=ExpenseCategory.ENTERTAINMENT,
                ),
            ],
            tax_amount=0.0,
            total_amount=33300.0,
            confidence_score=0.94,
        )

    if preset == "math_mismatch":
        return ReceiptSchema(
            merchant_name="古早味手寫便當店 (範例-金額不平衡)",
            invoice_number="CD-55120931",
            transaction_date=today,
            currency="TWD",
            items=[
                ExpenseItem(
                    name="招牌排骨便當",
                    unit_price=110.0,
                    quantity=3,
                    total_price=330.0,
                    category=ExpenseCategory.MEALS,
                ),
                ExpenseItem(
                    name="現榨鮮果汁",
                    unit_price=65.0,
                    quantity=2,
                    total_price=130.0,
                    category=ExpenseCategory.MEALS,
                ),
            ],
            tax_amount=0.0,
            total_amount=520.0,  # 故意與 330+130=460 不一致，觸發數學一致性警示
            confidence_score=0.78,
        )

    return ReceiptSchema(
        merchant_name="春水堂人文茶館 (範例-一般收據)",
        invoice_number="AA-10948271",
        transaction_date=today,
        currency="TWD",
        items=[
            ExpenseItem(
                name="珍珠奶茶 (中)",
                unit_price=105.0,
                quantity=2,
                total_price=210.0,
                category=ExpenseCategory.MEALS,
            ),
            ExpenseItem(
                name="功夫麵",
                unit_price=115.0,
                quantity=1,
                total_price=115.0,
                category=ExpenseCategory.MEALS,
            ),
            ExpenseItem(
                name="招牌滷味拼盤",
                unit_price=185.0,
                quantity=1,
                total_price=185.0,
                category=ExpenseCategory.MEALS,
            ),
        ],
        tax_amount=0.0,
        total_amount=510.0,
        confidence_score=0.98,
    )


__all__ = [
    "ExpenseCategory",
    "ExpenseItem",
    "ReceiptSchema",
    "AnomalyReport",
    "preprocess_and_encode_image",
    "extract_receipt_from_image",
    "get_mock_receipt",
]
