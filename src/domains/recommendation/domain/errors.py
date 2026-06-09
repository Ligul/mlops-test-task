class ItemIdOutOfRangeError(ValueError):
    def __init__(self, item_id: int, num_items: int) -> None:
        self.item_id = item_id
        self.num_items = num_items
        message = f"item_id {item_id} is out of range [0, {num_items})"
        super().__init__(message)
