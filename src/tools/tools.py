from typing import Dict, Any, Union

CATALOG = {
    "iPhone":  {"price": 25_000_000, "stock": 15, "weight_kg": 0.4},
    "iPad":    {"price": 18_000_000, "stock": 8,  "weight_kg": 0.5},
    "MacBook": {"price": 35_000_000, "stock": 0,  "weight_kg": 2.0},
}

COUPONS = {
    "WINNER": {"discount_percent": 10, "valid": True},
    "SUMMER": {"discount_percent": 15, "valid": True},
    "LEGACY": {"discount_percent": 0,  "valid": False, "reason": "Expired coupon code"},
}

def check_stock(item_name: str) -> Dict[str, Any]:
    """
    Check stock quantity and price for a given product item name.
    Args:
        item_name: Product name (e.g. 'iPhone', 'iPad', 'MacBook').
    """
    if not item_name or not isinstance(item_name, str):
        return {
            "ok": False,
            "error": "invalid_argument",
            "message": "item_name must be a non-empty string."
        }
    
    # Case-insensitive match check
    matched_key = None
    for k in CATALOG:
        if k.lower() == item_name.strip().lower():
            matched_key = k
            break
            
    if not matched_key:
        return {
            "ok": False,
            "error": "item_not_found",
            "message": f"Item '{item_name}' was not found in catalog."
        }
        
    item_data = CATALOG[matched_key]
    status = "in_stock" if item_data["stock"] > 0 else "out_of_stock"
    
    return {
        "ok": True,
        "item_name": matched_key,
        "price": item_data["price"],
        "stock": item_data["stock"],
        "weight_kg": item_data["weight_kg"],
        "status": status
    }

def get_discount(coupon_code: str) -> Dict[str, Any]:
    """
    Verify discount coupon code and return discount percentage.
    Args:
        coupon_code: Discount coupon code (e.g. 'WINNER', 'LEGACY').
    """
    if not coupon_code or not isinstance(coupon_code, str):
        return {
            "ok": False,
            "error": "invalid_argument",
            "message": "coupon_code must be a non-empty string."
        }
        
    code_upper = coupon_code.strip().upper()
    if code_upper not in COUPONS:
        return {
            "ok": False,
            "error": "coupon_not_found",
            "valid": False,
            "discount_percent": 0,
            "message": f"Coupon '{coupon_code}' is invalid or does not exist."
        }
        
    coupon_data = COUPONS[code_upper]
    if not coupon_data["valid"]:
        return {
            "ok": False,
            "error": "coupon_expired",
            "valid": False,
            "discount_percent": 0,
            "message": coupon_data.get("reason", "Coupon is not valid.")
        }
        
    return {
        "ok": True,
        "coupon_code": code_upper,
        "valid": True,
        "discount_percent": coupon_data["discount_percent"]
    }

def calc_shipping(weight: Union[float, int], destination: str) -> Dict[str, Any]:
    """
    Calculate shipping cost based on weight (kg) and destination city.
    Args:
        weight: Package weight in kg.
        destination: Target city (e.g. 'Hanoi', 'Saigon').
    """
    try:
        w = float(weight)
    except (ValueError, TypeError):
        return {
            "ok": False,
            "error": "invalid_argument",
            "message": f"Invalid weight value: {weight}"
        }
        
    if w <= 0:
        return {
            "ok": False,
            "error": "invalid_argument",
            "message": "Weight must be greater than 0."
        }
        
    if not destination or not isinstance(destination, str):
        return {
            "ok": False,
            "error": "invalid_argument",
            "message": "destination must be a non-empty string."
        }
        
    dest_clean = destination.strip().lower()
    if "hanoi" in dest_clean or "ha noi" in dest_clean or "hà nội" in dest_clean:
        cost = int(30000 + w * 10000)
        est_days = 1
    elif "saigon" in dest_clean or "hcm" in dest_clean or "ho chi minh" in dest_clean or "hồ chí minh" in dest_clean:
        cost = int(40000 + w * 15000)
        est_days = 2
    else:
        cost = int(50000 + w * 20000)
        est_days = 3
        
    return {
        "ok": True,
        "shipping_cost": cost,
        "estimated_days": est_days,
        "destination": destination
    }

# Tool Specs for LLM System Prompt / Agent Registry
TOOL_REGISTRY = {
    "check_stock": check_stock,
    "get_discount": get_discount,
    "calc_shipping": calc_shipping
}

TOOL_SPECS = [
    {
        "name": "check_stock",
        "description": "Checks stock availability, unit price (VND), and weight for a product. Usage: check_stock(item_name=\"iPhone\")",
        "function": check_stock
    },
    {
        "name": "get_discount",
        "description": "Verifies a coupon code and returns the discount percentage. Usage: get_discount(coupon_code=\"WINNER\")",
        "function": get_discount
    },
    {
        "name": "calc_shipping",
        "description": "Calculates shipping fee (VND) and delivery duration based on package weight (kg) and destination city. Usage: calc_shipping(weight=0.8, destination=\"Hanoi\")",
        "function": calc_shipping
    }
]
