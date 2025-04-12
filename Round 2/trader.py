from datamodel import OrderDepth, UserId, TradingState, Order, Symbol, Listing, ProsperityEncoder, Observation, Trade
from typing import List, Any, Dict
import string
import jsonpickle
import json
import numpy as np
import math

def target_pos_from_edge(edge: float, tau: int, p_max: int) -> int:
    """
    Piece‑wise linear:
        |edge| ≤ τ        → 0
        τ < |edge| < 3τ   → linearly 0 … p_max
        |edge| ≥ 3τ       → ±p_max
    """
    sgn = np.sign(edge)
    mag = abs(edge)
    if mag <= tau:
        return 0
    elif mag >= 3 * tau:
        return int(sgn * p_max)
    else:
        return int(sgn * p_max * (mag - tau) / (2 * tau))
    
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()
class Product:
    GIFT_BASKET = "PICNIC_BASKET1"
    CHOCOLATE = "JAMS"
    STRAWBERRIES = "CROISSANTS"
    ROSES = "DJEMBES"
    SYNTHETIC = "SYNTHETIC"
    SPREAD = "SPREAD"


PARAMS = {
    Product.SPREAD: {
        "default_spread_mean": 15,
        "default_spread_std": 45,
        "spread_sma_window": 1500,
        "spread_std_window": 28,
        "zscore_threshold": 5,
        "target_position": 58,
    }
}

BASKET_WEIGHTS = {
    Product.CHOCOLATE: 3,
    Product.STRAWBERRIES: 6,
    Product.ROSES: 1,
}


class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params

        self.LIMIT = {
            Product.GIFT_BASKET: 60,
            Product.CHOCOLATE: 350,
            Product.STRAWBERRIES: 250,
            Product.ROSES: 60
        }
        self.meta = {           # one basket only, so a tiny dict is fine
        "entry_ts": None,   # timestamp when position opened
        "entry_edge": None  # edge at entry
        }

    # Returns buy_order_volume, sell_order_volume
    def take_best_orders(
            self,
            product: str,
            fair_value: int,
            take_width: float,
            orders: List[Order],
            order_depth: OrderDepth,
            position: int,
            buy_order_volume: int,
            sell_order_volume: int,
            prevent_adverse: bool = False,
            adverse_volume: int = 0,
    ) -> (int, int):
        position_limit = self.LIMIT[product]
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]

            if best_ask <= fair_value - take_width:
                quantity = min(
                    best_ask_amount, position_limit - position
                )  # max amt to buy
                if quantity > 0:
                    orders.append(Order(product, best_ask, quantity))
                    buy_order_volume += quantity
                    order_depth.sell_orders[best_ask] += quantity
                    if order_depth.sell_orders[best_ask] == 0:
                        del order_depth.sell_orders[best_ask]

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if best_bid >= fair_value + take_width:
                quantity = min(
                    best_bid_amount, position_limit + position
                )  # should be the max we can sell
                if quantity > 0:
                    orders.append(Order(product, best_bid, -1 * quantity))
                    sell_order_volume += quantity
                    order_depth.buy_orders[best_bid] -= quantity
                    if order_depth.buy_orders[best_bid] == 0:
                        del order_depth.buy_orders[best_bid]
        return buy_order_volume, sell_order_volume

    def take_best_orders_with_adverse(
            self,
            product: str,
            fair_value: int,
            take_width: float,
            orders: List[Order],
            order_depth: OrderDepth,
            position: int,
            buy_order_volume: int,
            sell_order_volume: int,
            adverse_volume: int,
    ) -> (int, int):

        position_limit = self.LIMIT[product]
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]
            if abs(best_ask_amount) <= adverse_volume:
                if best_ask <= fair_value - take_width:
                    quantity = min(
                        best_ask_amount, position_limit - position
                    )  # max amt to buy
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        buy_order_volume += quantity
                        order_depth.sell_orders[best_ask] += quantity
                        if order_depth.sell_orders[best_ask] == 0:
                            del order_depth.sell_orders[best_ask]

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if abs(best_bid_amount) <= adverse_volume:
                if best_bid >= fair_value + take_width:
                    quantity = min(
                        best_bid_amount, position_limit + position
                    )  # should be the max we can sell
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -1 * quantity))
                        sell_order_volume += quantity
                        order_depth.buy_orders[best_bid] -= quantity
                        if order_depth.buy_orders[best_bid] == 0:
                            del order_depth.buy_orders[best_bid]

        return buy_order_volume, sell_order_volume

    def market_make(
            self,
            product: str,
            orders: List[Order],
            bid: int,
            ask: int,
            position: int,
            buy_order_volume: int,
            sell_order_volume: int,
    ) -> (int, int):
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, round(bid), buy_quantity))  # Buy order

        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, round(ask), -sell_quantity))  # Sell order
        return buy_order_volume, sell_order_volume

    def clear_position_order(
            self,
            product: str,
            fair_value: float,
            width: int,
            orders: List[Order],
            order_depth: OrderDepth,
            position: int,
            buy_order_volume: int,
            sell_order_volume: int,
    ) -> List[Order]:
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)

        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)

        if position_after_take > 0:
            # Aggregate volume from all buy orders with price greater than fair_for_ask
            clear_quantity = sum(
                volume
                for price, volume in order_depth.buy_orders.items()
                if price >= fair_for_ask
            )
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0:
            # Aggregate volume from all sell orders with price lower than fair_for_bid
            clear_quantity = sum(
                abs(volume)
                for price, volume in order_depth.sell_orders.items()
                if price <= fair_for_bid
            )
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume


    def take_orders(
            self,
            product: str,
            order_depth: OrderDepth,
            fair_value: float,
            take_width: float,
            position: int,
            prevent_adverse: bool = False,
            adverse_volume: int = 0,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        if prevent_adverse:
            buy_order_volume, sell_order_volume = self.take_best_orders_with_adverse(
                product,
                fair_value,
                take_width,
                orders,
                order_depth,
                position,
                buy_order_volume,
                sell_order_volume,
                adverse_volume,
            )
        else:
            buy_order_volume, sell_order_volume = self.take_best_orders(
                product,
                fair_value,
                take_width,
                orders,
                order_depth,
                position,
                buy_order_volume,
                sell_order_volume,
            )
        return orders, buy_order_volume, sell_order_volume

    def clear_orders(
            self,
            product: str,
            order_depth: OrderDepth,
            fair_value: float,
            clear_width: int,
            position: int,
            buy_order_volume: int,
            sell_order_volume: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume, sell_order_volume = self.clear_position_order(
            product,
            fair_value,
            clear_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
        )
        return orders, buy_order_volume, sell_order_volume


    def get_swmid(self, order_depth) -> float:
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        best_bid_vol = abs(order_depth.buy_orders[best_bid])
        best_ask_vol = abs(order_depth.sell_orders[best_ask])
        return (best_bid * best_ask_vol + best_ask * best_bid_vol) / (best_bid_vol + best_ask_vol)

    def get_synthetic_basket_order_depth(self, order_depths: Dict[str, OrderDepth]) -> OrderDepth:
        # Constants
        CHOCOLATE_PER_BASKET = BASKET_WEIGHTS[Product.CHOCOLATE]
        STRAWBERRIES_PER_BASKET = BASKET_WEIGHTS[Product.STRAWBERRIES]
        ROSES_PER_BASKET = BASKET_WEIGHTS[Product.ROSES]

        # Initialize the synthetic basket order depth
        synthetic_order_price = OrderDepth()

        # Calculate the best bid and ask for each component
        chocolate_best_bid = max(order_depths[Product.CHOCOLATE].buy_orders.keys()) if order_depths[
            Product.CHOCOLATE].buy_orders else 0
        chocolate_best_ask = min(order_depths[Product.CHOCOLATE].sell_orders.keys()) if order_depths[
            Product.CHOCOLATE].sell_orders else float('inf')
        strawberries_best_bid = max(order_depths[Product.STRAWBERRIES].buy_orders.keys()) if order_depths[
            Product.STRAWBERRIES].buy_orders else 0
        strawberries_best_ask = min(order_depths[Product.STRAWBERRIES].sell_orders.keys()) if order_depths[
            Product.STRAWBERRIES].sell_orders else float('inf')
        roses_best_bid = max(order_depths[Product.ROSES].buy_orders.keys()) if order_depths[
            Product.ROSES].buy_orders else 0
        roses_best_ask = min(order_depths[Product.ROSES].sell_orders.keys()) if order_depths[
            Product.ROSES].sell_orders else float('inf')

        # Calculate the implied bid and ask for the synthetic basket
        implied_bid = chocolate_best_bid * CHOCOLATE_PER_BASKET + strawberries_best_bid * STRAWBERRIES_PER_BASKET + roses_best_bid * ROSES_PER_BASKET
        implied_ask = chocolate_best_ask * CHOCOLATE_PER_BASKET + strawberries_best_ask * STRAWBERRIES_PER_BASKET + roses_best_ask * ROSES_PER_BASKET

        # Calculate the maximum number of synthetic baskets available at the implied bid and ask
        if implied_bid > 0:
            chocolate_bid_volume = order_depths[Product.CHOCOLATE].buy_orders[
                                       chocolate_best_bid] // CHOCOLATE_PER_BASKET
            strawberries_bid_volume = order_depths[Product.STRAWBERRIES].buy_orders[
                                          strawberries_best_bid] // STRAWBERRIES_PER_BASKET
            roses_bid_volume = order_depths[Product.ROSES].buy_orders[roses_best_bid] // ROSES_PER_BASKET
            implied_bid_volume = min(chocolate_bid_volume, strawberries_bid_volume, roses_bid_volume)
            synthetic_order_price.buy_orders[implied_bid] = implied_bid_volume

        if implied_ask < float('inf'):
            chocolate_ask_volume = -order_depths[Product.CHOCOLATE].sell_orders[
                chocolate_best_ask] // CHOCOLATE_PER_BASKET
            strawberries_ask_volume = -order_depths[Product.STRAWBERRIES].sell_orders[
                strawberries_best_ask] // STRAWBERRIES_PER_BASKET
            roses_ask_volume = -order_depths[Product.ROSES].sell_orders[roses_best_ask] // ROSES_PER_BASKET
            implied_ask_volume = min(chocolate_ask_volume, strawberries_ask_volume, roses_ask_volume)
            synthetic_order_price.sell_orders[implied_ask] = -implied_ask_volume

        return synthetic_order_price

    def convert_synthetic_basket_orders(self,
                                        synthetic_orders: List[Order], order_depths: Dict[str, OrderDepth]
                                        ) -> Dict[str, List[Order]]:
        # Initialize the dictionary to store component orders
        component_orders = {
            Product.CHOCOLATE: [],
            Product.STRAWBERRIES: [],
            Product.ROSES: [],
        }

        # Get the best bid and ask for the synthetic basket
        synthetic_basket_order_depth = self.get_synthetic_basket_order_depth(order_depths)
        best_bid = (
            max(synthetic_basket_order_depth.buy_orders.keys())
            if synthetic_basket_order_depth.buy_orders
            else 0
        )
        best_ask = (
            min(synthetic_basket_order_depth.sell_orders.keys())
            if synthetic_basket_order_depth.sell_orders
            else float("inf")
        )

        # Iterate through each synthetic basket order
        for order in synthetic_orders:
            # Extract the price and quantity from the synthetic basket order
            price = order.price
            quantity = order.quantity

            # Check if the synthetic basket order aligns with the best bid or ask
            if quantity > 0 and price >= best_ask:
                # Buy order - trade components at their best ask prices
                chocolate_price = min(order_depths[Product.CHOCOLATE].sell_orders.keys())
                strawberries_price = min(
                    order_depths[Product.STRAWBERRIES].sell_orders.keys()
                )
                roses_price = min(order_depths[Product.ROSES].sell_orders.keys())
            elif quantity < 0 and price <= best_bid:
                # Sell order - trade components at their best bid prices
                chocolate_price = max(order_depths[Product.CHOCOLATE].buy_orders.keys())
                strawberries_price = max(
                    order_depths[Product.STRAWBERRIES].buy_orders.keys()
                )
                roses_price = max(order_depths[Product.ROSES].buy_orders.keys())
            else:
                # The synthetic basket order does not align with the best bid or ask
                continue

            # Create orders for each component
            chocolate_order = Order(
                Product.CHOCOLATE,
                chocolate_price,
                quantity * BASKET_WEIGHTS[Product.CHOCOLATE],
            )
            strawberries_order = Order(
                Product.STRAWBERRIES,
                strawberries_price,
                quantity * BASKET_WEIGHTS[Product.STRAWBERRIES],
            )
            roses_order = Order(
                Product.ROSES, roses_price, quantity * BASKET_WEIGHTS[Product.ROSES]
            )

            # Add the component orders to the respective lists
            component_orders[Product.CHOCOLATE].append(chocolate_order)
            component_orders[Product.STRAWBERRIES].append(strawberries_order)
            component_orders[Product.ROSES].append(roses_order)

        return component_orders

    def execute_spread_orders(self, target_position: int, basket_position: int,
                          order_depths: Dict[str, OrderDepth],
                          state: TradingState, spread: float):


        if target_position == basket_position:
            return None

        target_quantity = abs(target_position - basket_position)
        basket_order_depth = order_depths[Product.GIFT_BASKET]
        synthetic_order_depth = self.get_synthetic_basket_order_depth(order_depths)

        if target_position > basket_position:
            basket_ask_price = min(basket_order_depth.sell_orders.keys())
            basket_ask_volume = abs(basket_order_depth.sell_orders[basket_ask_price])

            synthetic_bid_price = max(synthetic_order_depth.buy_orders.keys())
            synthetic_bid_volume = abs(synthetic_order_depth.buy_orders[synthetic_bid_price])

            orderbook_volume = min(basket_ask_volume, synthetic_bid_volume)
            execute_volume = min(orderbook_volume, target_quantity)

            basket_orders = [Order(Product.GIFT_BASKET, basket_ask_price, execute_volume)]
            synthetic_orders = [Order(Product.SYNTHETIC, synthetic_bid_price, -execute_volume)]

            aggregate_orders = self.convert_synthetic_basket_orders(synthetic_orders, order_depths)
            aggregate_orders[Product.GIFT_BASKET] = basket_orders
            # ---- remember entry details so we can time‑stop / half‑mean exit --------
            self.meta["entry_ts"]   = state.timestamp      # pass 'state' in args, see step 4
            self.meta["entry_edge"] = spread               # idem

            return aggregate_orders

        else:
            basket_bid_price = max(basket_order_depth.buy_orders.keys())
            basket_bid_volume = abs(basket_order_depth.buy_orders[basket_bid_price])

            synthetic_ask_price = min(synthetic_order_depth.sell_orders.keys())
            synthetic_ask_volume = abs(synthetic_order_depth.sell_orders[synthetic_ask_price])

            orderbook_volume = min(basket_bid_volume, synthetic_ask_volume)
            execute_volume = min(orderbook_volume, target_quantity)

            basket_orders = [Order(Product.GIFT_BASKET, basket_bid_price, -execute_volume)]
            synthetic_orders = [Order(Product.SYNTHETIC, synthetic_ask_price, execute_volume)]

            aggregate_orders = self.convert_synthetic_basket_orders(synthetic_orders, order_depths)
            aggregate_orders[Product.GIFT_BASKET] = basket_orders
            # ---- remember entry details so we can time‑stop / half‑mean exit --------
            self.meta["entry_ts"]   = state.timestamp      # pass 'state' in args, see step 4
            self.meta["entry_edge"] = spread               # idem

            return aggregate_orders

    def spread_orders(self, order_depths: Dict[str, OrderDepth], product: Product, basket_position: int,
                      spread_data: Dict[str, Any]):
        if Product.GIFT_BASKET not in order_depths.keys():
            return None

        basket_order_depth = order_depths[Product.GIFT_BASKET]
        synthetic_order_depth = self.get_synthetic_basket_order_depth(order_depths)
        basket_swmid = self.get_swmid(basket_order_depth)
        synthetic_swmid = self.get_swmid(synthetic_order_depth)
        spread = basket_swmid - synthetic_swmid
        spread_data["spread_history"].append(spread)
        MAX_HOLD = 4_000                      # ≈ 40 seconds of sim time
        if basket_position != 0 and self.meta["entry_ts"] is not None:
            if state.timestamp - self.meta["entry_ts"] > MAX_HOLD:
                return self.execute_spread_orders(target, basket_position,
                                  order_depths, state, spread)


        #  b) half‑mean take‑profit
        if basket_position != 0 and self.meta["entry_edge"] is not None:
            if abs(spread) < 0.5 * abs(self.meta["entry_edge"]):
                return self.execute_spread_orders(target, basket_position, order_depths, state, spread)

        if len(spread_data["spread_history"]) < self.params[Product.SPREAD]["spread_std_window"]:
            return None
        else:
            spread_std = np.std(spread_data["spread_history"][-self.params[Product.SPREAD]["spread_std_window"]:])

        if len(spread_data['spread_history']) == self.params[Product.SPREAD]["spread_sma_window"]:
            spread_mean = np.mean(spread_data['spread_history'])
            spread_data['curr_mean'] = spread_mean
        elif len(spread_data['spread_history']) > self.params[Product.SPREAD]["spread_sma_window"]:
            spread_mean = spread_data['curr_mean'] + (
                        (spread - spread_data['spread_history'][0]) / self.params[Product.SPREAD]["spread_sma_window"])
            spread_data["spread_history"].pop(0)
        else:
            spread_mean = self.params[Product.SPREAD]["default_spread_mean"]

        zscore = (spread - spread_mean) / spread_std

        tau   = self.params[Product.SPREAD]["zscore_threshold"]
        pmax  = self.params[Product.SPREAD]["target_position"]

        target = target_pos_from_edge(zscore, tau, pmax)

        if target != basket_position:
            return self.execute_spread_orders(target, basket_position,
                                  order_depths, state, spread)


        # if zscore >= self.params[Product.SPREAD]["zscore_threshold"]:
        #     if basket_position != -self.params[Product.SPREAD]["target_position"]:
        #         return self.execute_spread_orders(-self.params[Product.SPREAD]["target_position"], basket_position,
        #                                           order_depths)

        # if zscore <= -self.params[Product.SPREAD]["zscore_threshold"]:
        #     if basket_position != self.params[Product.SPREAD]["target_position"]:
        #         return self.execute_spread_orders(self.params[Product.SPREAD]["target_position"], basket_position,
        #                                           order_depths)

            # if (zscore < 0 and spread_data["prev_zscore"] > 0) or (zscore > 0 and spread_data["prev_zscore"] < 0) or spread_data["clear_flag"]:
            #     if basket_position == 0:
            #         spread_data["clear_flag"] = False
            #     else:
            #         spread_data["clear_flag"] = True
            #         return self.execute_spread_orders(0, basket_position, order_depths)

        spread_data["prev_zscore"] = zscore
        logger.print(spread_data["prev_zscore"])
        logger.print(spread_mean)
        return None

    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        result = {}
        conversions = 0

        if Product.SPREAD not in traderObject:
            traderObject[Product.SPREAD] = {
                "spread_history": [],
                "prev_zscore": 0,
                "clear_flag": False,
                "curr_avg": 0,
            }

        basket_position = state.position[Product.GIFT_BASKET] if Product.GIFT_BASKET in state.position else 0
        spread_orders = self.spread_orders(state.order_depths, Product.GIFT_BASKET, basket_position,
                                           traderObject[Product.SPREAD])
        if spread_orders != None:
            result[Product.CHOCOLATE] = spread_orders[Product.CHOCOLATE]
            result[Product.STRAWBERRIES] = spread_orders[Product.STRAWBERRIES]
            result[Product.ROSES] = spread_orders[Product.ROSES]
            result[Product.GIFT_BASKET] = spread_orders[Product.GIFT_BASKET]

        traderData = jsonpickle.encode(traderObject)
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData