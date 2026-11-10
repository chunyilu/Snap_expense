"""業務邏輯檢核與異常消費偵測引擎 (Business Rules Engine)。

負責執行：
1. 幣別匯率自動換算（以 TWD 為基準，串接公開 ExchangeRate API 並具備離線 Fallback 匯率）
2. 單筆大額品項消費警示（單一品項折合 TWD > 預設門檻，如 3,000 TWD）
3. 類別月度預算超支警示（當月累積金額 + 本次該類別金額 > 類別月度預算）
4. 收據數學一致性檢核（品項加總 + 稅額 != 收據總金額）
5. 重複發票號碼比對檢核（防止重複報帳/記帳）
6. 低辨識信心分數提醒
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import requests
from dotenv import load_dotenv

from database import DEFAULT_DB_PATH, check_duplicate_invoice, get_monthly_category_totals
from models import AnomalyReport, ExpenseCategory, ReceiptSchema

load_dotenv()

BUDGET_CONFIG_PATH = Path(__file__).parent / "category_budgets.json"

# 當無網路或匯率 API 逾時時使用的預設對 TWD 基準匯率
FALLBACK_EXCHANGE_RATES_TO_TWD: Dict[str, float] = {
    "TWD": 1.0,
    "USD": 32.15,
    "JPY": 0.215,
    "EUR": 34.80,
    "GBP": 40.90,
    "HKD": 4.12,
    "CNY": 4.45,
    "KRW": 0.0235,
    "SGD": 24.10,
    "AUD": 21.05,
    "CAD": 23.10,
    "THB": 0.94,
}

DEFAULT_CATEGORY_BUDGETS: Dict[str, float] = {
    ExpenseCategory.MEALS.value: float(os.getenv("BUDGET_MEALS", "12000.0")),
    ExpenseCategory.TRANSPORT.value: float(os.getenv("BUDGET_TRANSPORT", "4000.0")),
    ExpenseCategory.ENTERTAINMENT.value: float(os.getenv("BUDGET_ENTERTAINMENT", "5000.0")),
    ExpenseCategory.UTILITIES.value: float(os.getenv("BUDGET_UTILITIES", "3500.0")),
    ExpenseCategory.GROCERIES.value: float(os.getenv("BUDGET_GROCERIES", "6000.0")),
    ExpenseCategory.MEDICAL.value: float(os.getenv("BUDGET_MEDICAL", "3000.0")),
    ExpenseCategory.OFFICE.value: float(os.getenv("BUDGET_OFFICE", "2000.0")),
    ExpenseCategory.OTHER.value: float(os.getenv("BUDGET_OTHER", "5000.0")),
}

# 記憶體匯率快取避免重複請求
_EXCHANGE_RATE_CACHE: Dict[str, float] = {"TWD": 1.0}


def load_category_budgets(config_path: Path = BUDGET_CONFIG_PATH) -> Dict[str, float]:
    """載入各消費類別月度預算設定（優先讀取 category_budgets.json，並以環境變數與預設值補齊）。

    Args:
        config_path: 預算 JSON 設定檔路徑。

    Returns:
        Dict[str, float]: 各類別名稱對應之月度預算 (TWD)。
    """
    budgets = dict(DEFAULT_CATEGORY_BUDGETS)
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                for k, v in data.items():
                    try:
                        budgets[str(k)] = float(v)
                    except (TypeError, ValueError):
                        continue
        except (OSError, json.JSONDecodeError):
            pass
    return budgets


def save_category_budgets(
    budgets: Dict[str, float],
    config_path: Path = BUDGET_CONFIG_PATH,
) -> None:
    """將更新後的各類別月度預算寫入 category_budgets.json。

    Args:
        budgets: 類別名稱對應預算金額之字典。
        config_path: 預算 JSON 設定檔路徑。
    """
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(budgets, f, ensure_ascii=False, indent=2)


def get_exchange_rate_to_twd(currency: str, timeout: float = 3.5) -> Tuple[float, bool]:
    """取得指定幣別換算為新台幣 (TWD) 之匯率（1 單位外幣 = ? TWD）。

    優先查詢公開 ExchangeRate-API (`https://open.er-api.com/v6/latest/{currency}`)，
    若網路異常或幣別不支援則自動回退至內建 Fallback 匯率表。

    Args:
        currency: 三碼幣別代碼（如 'TWD', 'JPY', 'USD'）。
        timeout: HTTP 請求逾時秒數。

    Returns:
        Tuple[float, bool]: (對新台幣匯率, 是否來自即時 API)。
    """
    code = (currency or "TWD").strip().upper()
    if code == "TWD":
        return 1.0, True

    if code in _EXCHANGE_RATE_CACHE:
        return _EXCHANGE_RATE_CACHE[code], True

    try:
        url = f"https://open.er-api.com/v6/latest/{code}"
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            payload = response.json()
            rates = payload.get("rates", {})
            if "TWD" in rates and float(rates["TWD"]) > 0:
                rate = round(float(rates["TWD"]), 6)
                _EXCHANGE_RATE_CACHE[code] = rate
                return rate, True
    except requests.RequestException:
        pass
    except (ValueError, KeyError):
        pass

    fallback_rate = FALLBACK_EXCHANGE_RATES_TO_TWD.get(code, 1.0)
    return fallback_rate, False


def evaluate_receipt_anomalies(
    receipt: ReceiptSchema,
    category_budgets: Optional[Dict[str, float]] = None,
    single_item_threshold_twd: Optional[float] = None,
    custom_exchange_rate: Optional[float] = None,
    db_path: str = DEFAULT_DB_PATH,
) -> AnomalyReport:
    """針對收據執行完整業務規則檢核與異常消費偵測。

    檢核項目包含：
    1. 幣別匯率換算為新台幣 (TWD)
    2. 重複發票號碼檢核
    3. 收據金額與品項加總之數學一致性檢驗 (`SUM(items.total_price) + tax_amount == total_amount`)
    4. 單筆大額品項消費警示 (`item.total_price * rate > single_item_threshold_twd`)
    5. 當月類別累積支出超支警示 (`当月已累計 + 本次該類別合計 > 類別月度預算`)
    6. 模型辨識信心分數過低提醒 (`confidence_score < 0.8`)

    Args:
        receipt: 待檢核之 ReceiptSchema 物件。
        category_budgets: 各類別月度預算字典 (TWD)，未提供則自動載入設定。
        single_item_threshold_twd: 單一品項大額消費門檻 (TWD)，預設讀取環境變數或 3000.0。
        custom_exchange_rate: 使用者手動指定之匯率（若為 None 則自動查詢）。
        db_path: SQLite 資料庫檔案路徑。

    Returns:
        AnomalyReport: 包含是否異常、警示訊息清單、匯率與換算後新台幣總金額之報告。
    """
    budgets = category_budgets or load_category_budgets()
    if single_item_threshold_twd is None:
        single_item_threshold_twd = float(os.getenv("SINGLE_ITEM_ALERT_THRESHOLD", "3000.0"))

    warnings = []

    # 1. 匯率計算
    if custom_exchange_rate is not None and custom_exchange_rate > 0:
        rate = float(custom_exchange_rate)
        is_live_rate = True
    else:
        rate, is_live_rate = get_exchange_rate_to_twd(receipt.currency)

    total_twd = round(receipt.total_amount * rate, 2)

    if receipt.currency.upper() != "TWD" and not is_live_rate:
        warnings.append(
            f"ℹ️ 匯率提示：目前無法連線至即時匯率服務，已採用離線參考匯率 (1 {receipt.currency} ≈ {rate} TWD)。"
        )

    # 2. 重複發票號碼比對
    if receipt.invoice_number:
        existing = check_duplicate_invoice(receipt.invoice_number, db_path=db_path)
        if existing:
            warnings.append(
                f"🚨 重複發票警示：發票號碼 `{receipt.invoice_number}` 已於 "
                f"{existing['transaction_date']} 登錄過（商家：{existing['merchant_name']}，"
                f"金額：{existing['total_amount']} {existing['currency']}）。"
            )

    # 3. 數學一致性檢驗：SUM(items.total_price) + tax_amount == total_amount
    items_sum = round(sum(item.total_price for item in receipt.items), 2)
    expected_total = round(items_sum + receipt.tax_amount, 2)
    if abs(expected_total - receipt.total_amount) > 1.0:
        diff = round(receipt.total_amount - expected_total, 2)
        warnings.append(
            f"⚠️ 金額不平衡警示：品項小計總和 ({items_sum:,.2f}) + 稅額 ({receipt.tax_amount:,.2f}) "
            f"= {expected_total:,.2f} {receipt.currency}，與收據總金額 ({receipt.total_amount:,.2f} {receipt.currency}) "
            f"相差 {diff:+,.2f} {receipt.currency}，請確認是否有折扣、服務費或辨識誤差。"
        )

    # 4. 單筆大額品項警示
    for item in receipt.items:
        item_twd = round(item.total_price * rate, 2)
        if item_twd > single_item_threshold_twd:
            warnings.append(
                f"💸 單筆大額消費警示：品項「{item.name}」金額折合 NT$ {item_twd:,.0f} "
                f"（超過單筆警示門檻 NT$ {single_item_threshold_twd:,.0f}）。"
            )

    # 5. 當月類別預算超支檢核
    year_month = receipt.transaction_date.strftime("%Y-%m")
    current_month_totals = get_monthly_category_totals(year_month, db_path=db_path)

    receipt_category_twd: Dict[str, float] = {}
    for item in receipt.items:
        cat_name = item.category.value if isinstance(item.category, ExpenseCategory) else str(item.category)
        receipt_category_twd[cat_name] = receipt_category_twd.get(cat_name, 0.0) + round(item.total_price * rate, 2)

    for cat_name, added_twd in receipt_category_twd.items():
        budget_limit = float(budgets.get(cat_name, 0.0))
        if budget_limit <= 0:
            continue
        existing_twd = float(current_month_totals.get(cat_name, 0.0))
        projected_twd = round(existing_twd + added_twd, 2)
        if projected_twd > budget_limit:
            overrun = round(projected_twd - budget_limit, 2)
            warnings.append(
                f"📈 類別預算超支警示：「{cat_name}」於 {year_month} 已累計 NT$ {existing_twd:,.0f}，"
                f"加上本次 NT$ {added_twd:,.0f} 後達 NT$ {projected_twd:,.0f}，"
                f"超出月度預算 (NT$ {budget_limit:,.0f}) 共 NT$ {overrun:,.0f}！"
            )

    # 6. 辨識信心度檢核
    if receipt.confidence_score < 0.80:
        warnings.append(
            f"🔍 低信心度提醒：AI 對此收據的辨識信心分數為 {receipt.confidence_score:.0%}，建議仔細核對品項與金額。"
        )

    # 判斷是否屬於異常（排除單純的離線匯率提示）
    real_anomalies = [w for w in warnings if not w.startswith("ℹ️")]
    return AnomalyReport(
        is_anomaly=len(real_anomalies) > 0,
        warning_messages=warnings,
        exchange_rate_to_twd=rate,
        total_amount_twd=total_twd,
    )
