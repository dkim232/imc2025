import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results.csv", delimiter=";")

# Optional: filter for a specific product
# df = df[df['product'] == 'KELP']  # Uncomment and change if needed

# Compute cumulative profit and loss
df['cumulative_pnl'] = df['profit_and_loss'].cumsum()

# Plot
plt.figure(figsize=(12, 6))
plt.plot(df['timestamp'], df['cumulative_pnl'], color='mediumslateblue', linewidth=2)
plt.xlabel("Time / Trades")
plt.ylabel("Portfolio Value (Cumulative PnL)")
plt.title("Portfolio Value Over Time")
plt.grid(True)
plt.tight_layout()
plt.show()
