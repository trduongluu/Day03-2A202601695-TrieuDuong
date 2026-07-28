import pytest
from src.tools.tools import check_stock, get_discount, calc_shipping

def test_check_stock_valid():
    res = check_stock("iPhone")
    assert res["ok"] is True
    assert res["price"] == 25000000
    assert res["stock"] == 15
    assert res["status"] == "in_stock"

def test_check_stock_out_of_stock():
    res = check_stock("MacBook")
    assert res["ok"] is True
    assert res["stock"] == 0
    assert res["status"] == "out_of_stock"

def test_check_stock_not_found():
    res = check_stock("NonExistentItem")
    assert res["ok"] is False
    assert res["error"] == "item_not_found"

def test_get_discount_valid():
    res = get_discount("WINNER")
    assert res["ok"] is True
    assert res["discount_percent"] == 10
    assert res["valid"] is True

def test_get_discount_expired():
    res = get_discount("LEGACY")
    assert res["ok"] is False
    assert res["error"] == "coupon_expired"
    assert res["valid"] is False
    assert res["discount_percent"] == 0

def test_get_discount_not_found():
    res = get_discount("FAKECODE")
    assert res["ok"] is False
    assert res["error"] == "coupon_not_found"

def test_calc_shipping_valid():
    res = calc_shipping(0.8, "Hanoi")
    assert res["ok"] is True
    assert res["shipping_cost"] == 38000
    assert res["estimated_days"] == 1

def test_calc_shipping_saigon():
    res = calc_shipping(0.8, "Saigon")
    assert res["ok"] is True
    assert res["shipping_cost"] == 52000
    assert res["estimated_days"] == 2

def test_calc_shipping_invalid_args():
    res1 = calc_shipping(-5, "Hanoi")
    assert res1["ok"] is False
    assert res1["error"] == "invalid_argument"

    res2 = calc_shipping(0.8, "")
    assert res2["ok"] is False
    assert res2["error"] == "invalid_argument"

def test_tools_determinism():
    run1 = check_stock("iPhone")
    run2 = check_stock("iPhone")
    assert run1 == run2
