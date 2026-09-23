# step 1 load teh data 
# erquired data fromat : json, path , re
import json
import re
from pathlib import Path

PATH = Path('data/orders.json')

CUSTOMER_SAFE_FIELDS = [
    "order_id",
    "membership_tier",
    "placed_at",
    "status",
    "status_updated_at",
    "shipped_at",
    "delivered_at",
    "carrier",
    "tracking_number",
    "estimated_delivery",
    "customer_safe_message",
]

ITEM_SAFE_FIELDS = ["name", "quantity", "final_sale"]
    

def _load_orders():
    with open(PATH,"r",encoding="utf-8") as f:
        data = json.load(f)
        # print(type(data))
    return data

def _normalize_order_id(order_id):
    if not isinstance(order_id,str):
        return None
    
    normalized = re.sub(r"[^A-Za-z0-9]+", "-", order_id.strip()).strip("-").upper()
    if not normalized:
        return None
    normalized = re.sub(r"^([A-Z]+)(\d+)$", r"\1-\2", normalized)
    
    return normalized



def order_lookup_tool(order_id):

    normalized_id=_normalize_order_id(order_id)
    # print(normalized_id)
    if not normalized_id :
        return {"found": False, "error": "invalid order id"}

    data = _load_orders()
    for order in data["orders"]:
        if order.get("order_id") == normalized_id:
            # print(order)
            result = {}
            for field in CUSTOMER_SAFE_FIELDS :
                result[field] = order.get(field)

            items_list = []
            for item in order.get("items", []):
                item_result = {}
                for field in ITEM_SAFE_FIELDS:
                    item_result[field] = item.get(field)
                items_list.append(item_result)

            result["items"] = items_list
            result["found"] = True
            return result
                
    return {"found": False, "error": "order not found"}
            
# if __name__ == "__main__":
#     print(order_lookup_tool(order_id="ORD-1007"))
