QLIB_COMMIT = "54355232463878d2eebb91fe0ee5fa7fa1f5976c"
SCHEMA_VERSION = "ptop.raw25.v1"

OHLC_FIELDS = ("open", "high", "low", "close")
TRADE_FIELDS = ("volume", "amount", "deal_number")
BOOK_PRICE_FIELDS = tuple(
    f"{side}_price{level}" for level in range(1, 4) for side in ("bid", "ask")
)
BOOK_VOLUME_FIELDS = tuple(
    f"{side}_volume{level}" for level in range(1, 4) for side in ("bid", "ask")
)
BOOK_ORDER_FIELDS = tuple(
    f"{side}_num_orders{level}" for level in range(1, 4) for side in ("bid", "ask")
)
BOOK_FIELDS = BOOK_PRICE_FIELDS + BOOK_VOLUME_FIELDS + BOOK_ORDER_FIELDS
FEATURES = OHLC_FIELDS + TRADE_FIELDS + BOOK_FIELDS
PRICE_FIELDS = OHLC_FIELDS + BOOK_PRICE_FIELDS
SCALE_FIELDS = PRICE_FIELDS + ("amount",)
BOOK12_FIELDS = tuple(
    f"{field}{level}"
    for level in range(1, 4)
    for field in ("ask_price", "ask_volume", "bid_price", "bid_volume")
)


class ContractError(ValueError):
    pass
