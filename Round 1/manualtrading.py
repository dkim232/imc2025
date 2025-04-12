# We have four currencies: "Seashells", "Snowballs", "Pizzas", "Silicon Nuggets"
# The table below is read as: rates[from_currency][to_currency] = rate

rates = {
    "Seashells": {
        "Seashells": 1.0,
        "Snowballs": 1.34,
        "Pizzas": 1.98,
        "Silicon Nuggets": 0.64
    },
    "Snowballs": {
        "Seashells": 0.72,
        "Snowballs": 1.0,
        "Pizzas": 1.45,
        "Silicon Nuggets": 0.52
    },
    "Pizzas": {
        "Seashells": 0.48,
        "Snowballs": 0.70,
        "Pizzas": 1.0,
        "Silicon Nuggets": 0.31
    },
    "Silicon Nuggets": {
        "Seashells": 1.49,
        "Snowballs": 1.95,
        "Pizzas": 3.10,
        "Silicon Nuggets": 1.0
    }
}

MAX_TRADES = 5               # including the first and final trade
START_CURRENCY = "Seashells"
START_AMOUNT = 500_000.0

best_amount = 0.0
best_path = []

def dfs(current_currency, current_amount, trades_so_far, path):
    global best_amount, best_path
    
    # If we've reached the maximum number of trades,
    # check if we are back to SeaShells
    if trades_so_far == MAX_TRADES:
        if current_currency == START_CURRENCY and current_amount > best_amount:
            best_amount = current_amount
            best_path = path[:]
        return

    # Otherwise, try every possible next currency
    for next_currency, rate in rates[current_currency].items():
        if rate <= 0:
            continue  # skip any zero or invalid rates
        next_amount = current_amount * rate
        
        # Record this trade in our path
        path.append(f"{current_currency} -> {next_currency} @ {rate:.3f} => {next_amount:,.2f}")
        
        dfs(next_currency, next_amount, trades_so_far + 1, path)
        
        # Backtrack
        path.pop()

# Kick off the search
dfs(START_CURRENCY, START_AMOUNT, 0, [])

print("BEST FINAL AMOUNT:", best_amount)
print("BEST PATH:")
for step in best_path:
    print("   ", step)