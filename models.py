"""Pydantic v2 資料驗證模型定義。

包含收據明細品項 (ExpenseItem)、收據主體結構 (ReceiptSchema)、
消費類別列舉 (ExpenseCategory) 與異常檢測報告 (AnomalyReport)。
"""

from datetime import date
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, PrivateAttr, field_validator, model_validator


class ExpenseCategory(str, Enum):
    """消費分類列舉。"""

    MEALS = "餐飲食品"
    TRANSPORT = "交通運輸"
    ENTERTAINMENT = "休閒娛樂"
    UTILITIES = "水電與通訊"
    GROCERIES = "生活雜貨"
    MEDICAL = "醫療保健"
    OFFICE = "辦公文具"
    OTHER = "其他支出"


class ExpenseItem(BaseModel):
    """單一消費品項明細模型。"""

    name: str = Field(..., description="商品或服務名稱")
    unit_price: float = Field(..., ge=0, description="單價")
    quantity: int = Field(default=1, ge=1, description="數量")
    total_price: float = Field(..., ge=0, description="該品項小計金額 (單價 * 數量)")
    category: ExpenseCategory = Field(
        default=ExpenseCategory.OTHER,
        description="自動推斷之消費分類",
    )

    @model_validator(mode="after")
    def verify_item_total(self) -> "ExpenseItem":
        """驗證單價乘以數量是否與小計一致，若誤差超過 0.5 則自動修正。"""
        calculated = round(self.unit_price * self.quantity, 2)
        if abs(calculated - self.total_price) > 0.5:
            self.total_price = calculated
        return self


class ReceiptSchema(BaseModel):
    """收據/發票完整結構化資料模型。"""

    merchant_name: str = Field(..., description="店家/開立機構名稱")
    invoice_number: Optional[str] = Field(
        default=None,
        description="發票號碼（如台灣統一發票兩碼英文+八碼數字，若無則為 None）",
    )
    transaction_date: date = Field(..., description="交易日期，格式 YYYY-MM-DD")
    currency: str = Field(default="TWD", description="幣別代碼，例如 TWD, JPY, USD")
    items: List[ExpenseItem] = Field(..., min_length=1, description="購買品項明細列表")
    tax_amount: float = Field(default=0.0, ge=0, description="稅額 (若無標示則為 0)")
    total_amount: float = Field(..., gt=0, description="收據/發票總計金額")
    confidence_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="模型對此張收據辨識完整度之自我評估信心分數 (0.0 ~ 1.0)",
    )

    _math_mismatch: bool = PrivateAttr(default=False)

    @field_validator("invoice_number")
    @classmethod
    def clean_invoice_number(cls, v: Optional[str]) -> Optional[str]:
        """清理並正規化發票號碼字串。"""
        if v:
            cleaned = v.strip().replace(" ", "").upper()
            return cleaned if cleaned else None
        return None

    @field_validator("currency")
    @classmethod
    def clean_currency(cls, v: str) -> str:
        """正規化幣別代碼為三位大寫英文字母。"""
        cleaned = v.strip().upper()
        return cleaned if cleaned else "TWD"

    @model_validator(mode="after")
    def verify_receipt_math(self) -> "ReceiptSchema":
        """檢驗品項總和加稅額是否與總金額一致，保留容錯標記供業務層觸發警示。"""
        items_sum = sum(item.total_price for item in self.items)
        expected_total = items_sum + self.tax_amount
        if abs(expected_total - self.total_amount) > 1.0:
            self._math_mismatch = True
        else:
            self._math_mismatch = False
        return self

    @property
    def has_math_mismatch(self) -> bool:
        """回傳品項小計加稅額是否與總金額不一致。"""
        items_sum = sum(item.total_price for item in self.items)
        return abs((items_sum + self.tax_amount) - self.total_amount) > 1.0


class AnomalyReport(BaseModel):
    """業務規則異常檢核報告。"""

    is_anomaly: bool = False
    warning_messages: List[str] = Field(default_factory=list)
    exchange_rate_to_twd: float = Field(default=1.0, description="對新台幣換算匯率")
    total_amount_twd: float = Field(default=0.0, description="換算後新台幣總金額")
