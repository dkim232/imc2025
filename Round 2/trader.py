"""
Island Trading Challenge – Spread‑Arb bot  ✨  (v2 – edge‑threshold logic)
-----------------------------------------------------------------------
Implements the full list of upgrades discussed:
  • trades on raw mis‑pricing (edge) with a fixed threshold
  • VWAP‑of‑top‑3 for fair mids (robust to spoofing)
  • flatten when edge crosses zero
  • guards against empty books
  • supports both picnic baskets (B1 & B2)
  • simple single‑side aggressive execution (cross basket, hit legs)
     – good enough once threshold ≥ cost of two spreads
Feel free to tweak the parameters at the top.
"""

from datamodel import OrderDepth, Order, TradingState, Symbol, Listing, Observation, Trade  # type: ignore
from typing import Dict, List, Any, Tuple
import json, jsonpickle, numpy as np

# ────────────────────────────────────────────────────────────────────────────────
# Utility logger (unchanged)
# ────────────────────────────────────────────────────────────────────────────────
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: Dict[Symbol, List[Order]], conversions: int, trader_data: str) -> None:
        from ProsperityEncoder import ProsperityEncoder  # type: ignore

        base_length = len(
            json.dumps(
                [state.timestamp, orders, conversions, "", ""],
                cls=ProsperityEncoder,
                separators=(",", ":"),
            )
        )
        max_item_length = (self.max_log_length - base_length) // 3
        print(
            json.dumps(
                [state.timestamp, orders, conversions, trader_data[:max_item_length], self.logs[:max_item_length]],
                cls=ProsperityEncoder,
                separators=(",", ":"),
            )
        )
        self.logs = ""

logger = Logger()

# ────────────────────────────────────────────────────────────────────────────────
# Product & basket metadata
# ────────────────────────────────────────────────────────────────────────────────
class Product:
    CROISSANTS      = "CROISSANTS"
    JAMS            = "JAMS"
    DJEMBES         = "DJEMBES"
    BASKET1         = "PICNIC_BASKET1"   # 6 C + 3 J + 1 D
    BASKET2         = "PICNIC_BASKET2"   # 4 C + 2 J
    SYNTHETIC       = "SYNTHETIC"        # pseudo‑symbol used internally

# Basket composition (weights per basket)
BASKET_WEIGHTS: Dict[str, Dict[str, int]] = {
    Product.BASKET1: {
        Product.CROISSANTS: 6,
        Product.JAMS:       3,
        Product.DJEMBES:    1,
    },
    Product.BASKET2: {
        Product.CROISSANTS: 4,
        Product.JAMS:       2,
    },
}

# Position limits (given by the exchange spec)
LIMITS = {
    Product.CROISSANTS: 250,
    Product.JAMS:       350,
    Product.DJEMBES:     60,
    Product.BASKET1:     60,
    Product.BASKET2:    100,
}

# Strategy parameters – tweak to taste
PARAMS = {
    Product.BASKET1: {"edge_threshold": 18, "target_pos": 30},
    Product.BASKET2: {"edge_threshold": 30, "target_pos": 40},
}

# ────────────────────────────────────────────────────────────────────────────────
# Helper functions
# ────────────────────────────────────────────────────────────────────────────────

def vwap(side: Dict[int, int], k: int = 3, reverse: bool = False) -> float:
    """Volume‑weighted average price of the top‑k levels."""
    if not side:
        return float("nan")
    levels = sorted(side.items(), reverse=reverse)[:k]
    vol = sum(abs(q) for _, q in levels)
    return sum(p * abs(q) for p, q in levels) / vol if vol else float("nan")


def safe_best(side: Dict[int, int], is_bid: bool) -> Tuple[int, int]:
    """Return (price, volume) for best bid/ask or (None,0) if empty."""
    if not side:
        return None, 0  # type: ignore
    price = max(side) if is_bid else min(side)
    return price, side[price]


def swmid(od: OrderDepth) -> float:
    bid = vwap(od.buy_orders, k=3, reverse=True)
    ask = vwap(od.sell_orders, k=3, reverse=False)
    return (bid + ask) / 2

# ────────────────────────────────────────────────────────────────────────────────
# Trader
# ────────────────────────────────────────────────────────────────────────────────
class Trader:
    def __init__(self):
        # persistent scratch‑pad between ticks
        self.state_mem: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Synthetic basket order‑book from component legs
    # ------------------------------------------------------------------
    def synthetic_depth(self, basket: str, depths: Dict[str, OrderDepth]) -> OrderDepth:
        weights = BASKET_WEIGHTS[basket]
        syn = OrderDepth()

        # best bid = sum(component bids × weight)
        bid_px = 0
        bid_qty = float("inf")
        for leg, w in weights.items():
            px, qty = safe_best(depths[leg].buy_orders, True)
            if px is None:
                return syn  # empty → return empty depth
            bid_px += px * w
            bid_qty = min(bid_qty, qty // w)
        if bid_qty > 0:
            syn.buy_orders[bid_px] = bid_qty

        # best ask = sum(component asks × weight)
        ask_px = 0
        ask_qty = float("inf")
        for leg, w in weights.items():
            px, qty = safe_best(depths[leg].sell_orders, False)
            if px is None:
                return syn
            ask_px += px * w
            ask_qty = min(ask_qty, -qty // w)
        if ask_qty > 0:
            syn.sell_orders[ask_px] = -ask_qty
        return syn

    # ------------------------------------------------------------------
    # Build and return spread orders to move basket_position → target
    # ------------------------------------------------------------------
    def build_spread_orders(self, basket: str, target: int, pos: int,
                            depths: Dict[str, OrderDepth]) -> Dict[str, List[Order]]:
        if target == pos:
            return {}

        qty = abs(target - pos)
        basket_depth = depths[basket]
        syn_depth    = self.synthetic_depth(basket, depths)
        weights      = BASKET_WEIGHTS[basket]

        orders: Dict[str, List[Order]] = {p: [] for p in weights}
        orders[basket] = []

        if target > pos:  # we need to BUY basket / SELL legs
            ask_px, ask_vol = safe_best(basket_depth.sell_orders, False)
            bid_px, bid_vol = safe_best(syn_depth.buy_orders, True)
            if ask_px is None or bid_px is None:
                return {}
            trade_vol = min(qty, -ask_vol, bid_vol)
            if trade_vol <= 0:
                return {}
            # aggressive on basket
            orders[basket].append(Order(basket, ask_px, trade_vol))
            # aggressive on each leg (sell weight × vol at bid)
            for leg, w in weights.items():
                leg_bid_px, _ = safe_best(depths[leg].buy_orders, True)
                orders[leg].append(Order(leg, leg_bid_px, -w * trade_vol))
        else:            # need to SELL basket / BUY legs
            bid_px, bid_vol = safe_best(basket_depth.buy_orders, True)
            ask_px, ask_vol = safe_best(syn_depth.sell_orders, False)
            if bid_px is None or ask_px is None:
                return {}
            trade_vol = min(qty, bid_vol, -ask_vol)
            if trade_vol <= 0:
                return {}
            orders[basket].append(Order(basket, bid_px, -trade_vol))
            for leg, w in weights.items():
                leg_ask_px, _ = safe_best(depths[leg].sell_orders, False)
                orders[leg].append(Order(leg, leg_ask_px, w * trade_vol))
        return orders

    # ------------------------------------------------------------------
    # Main per‑tick entry point
    # ------------------------------------------------------------------
    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        conversions = 0

        depths = state.order_depths
        positions = state.position

        for basket in (Product.BASKET1, Product.BASKET2):
            if basket not in depths:
                continue  # not yet listed
            pos = positions.get(basket, 0)

            # --- compute edge ---
            basket_mid = swmid(depths[basket])
            synth_mid  = swmid(self.synthetic_depth(basket, depths))
            if np.isnan(basket_mid) or np.isnan(synth_mid):
                continue  # incomplete books
            edge = basket_mid - synth_mid  # negative → basket rich

            τ     = PARAMS[basket]["edge_threshold"]
            p_max = PARAMS[basket]["target_pos"]

            if edge < -τ:
                target = -p_max
            elif edge > τ:
                target = p_max
            else:
                target = 0

            # flatten if edge crossed zero the other way
            prev_edge = self.state_mem.get(f"prev_edge_{basket}", 0.0)
            if pos != 0 and prev_edge * edge < 0:
                target = 0
            self.state_mem[f"prev_edge_{basket}"] = edge

            # build orders if needed
            spread_orders = self.build_spread_orders(basket, target, pos, depths)
            for sym, olist in spread_orders.items():
                result.setdefault(sym, []).extend(olist)

        # serialise scratch (not used for now but kept for compatibility)
        trader_data = jsonpickle.encode({})
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
