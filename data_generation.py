import csv
import random

# Output file
filename = "weather_data.csv"

# Column names
fields = ["Temperature (°C)", "Humidity (%)"]

# Generate 100 rows of random but realistic data
rows = []
for _ in range(200):
    temp = round(random.uniform(15.0, 50.0), 1)  # Temperature between 15–40 °C
    humidity = random.randint(10, 95)            # Humidity between 20–90 %
    rows.append([temp, humidity])

# Write to CSV
with open(filename, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(fields)
    writer.writerows(rows)

print(f"CSV file '{filename}' with 100 rows created.")
