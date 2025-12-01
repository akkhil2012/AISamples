import csv
import random
import uuid
from datetime import datetime, timedelta
import math

def random_ip():
    return ".".join(str(random.randint(1, 254)) for _ in range(4))

def random_date(start_year=1950, end_year=2005):
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = end - start
    return (start + timedelta(days=random.randint(0, delta.days))).strftime("%Y-%m-%d")

def random_tx_datetime(start_date="2024-01-01", end_date="2024-12-31"):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    delta = end - start
    dt = start + timedelta(seconds=random.randint(0, int(delta.total_seconds())))
    return dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M:%S")

def luhn_16_digit():
    # Simple 16-digit numeric string (not strict Luhn, but looks like a card)
    return "".join(str(random.randint(0, 9)) for _ in range(16))

def masked_card(card):
    return f"**** **** **** {card[-4:]}"

def generate_csv(path="train.csv", n_rows=1000, fraud_rate=0.03, seed=42):
    random.seed(seed)

    headers = [
        "ss","first","last","gender","street","city","state","zip","lat","long",
        "city_pop","job","dob","acct_num","trans_num","trans_date","trans_time",
        "categoty","amt","customer_device","customer_payment_method",
        "customer_card_number","customer_transaction_id","merch_name","merch_id",
        "merch_url","merch_transaction_ip","mercj_lat","merch_long","is_fraud"
    ]

    first_names = ["John","Jane","Akhil","Priya","Michael","Sara","David","Anita","Luis","Mei"]
    last_names = ["Doe","Sharma","Kumar","Patel","Smith","Brown","Garcia","Singh","Lee","Nguyen"]
    streets = ["1st Ave","2nd St","Main St","Park Ave","Oak St","Pine St","Maple Ave","Cedar Rd"]
    cities = ["New York","San Francisco","Austin","Seattle","Miami","Boston","Chicago","Dallas"]
    states = ["NY","CA","TX","WA","FL","MA","IL","GA","NJ","PA"]
    jobs = ["Engineer","Analyst","Teacher","Nurse","Manager","Developer","Cashier","Consultant","Clerk","Designer"]
    categories = ["grocery","gas","online_purchase","electronics","travel","restaurant","utilities","pharmacy"]
    devices = ["mobile","web","pos"]
    pay_methods = ["credit_card","debit_card","wallet"]
    merchants = ["Amazon","Walmart","Target","BestBuy","Uber","DoorDash","Starbucks","Shell","Delta","Costco"]

    # Random base geos for cities and merchants (very rough)
    city_geo = {c: (round(random.uniform(25.0, 48.0), 6), round(random.uniform(-124.0, -67.0), 6)) for c in cities}
    merch_geo = {m: (round(random.uniform(25.0, 48.0), 6), round(random.uniform(-124.0, -67.0), 6)) for m in merchants}

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for i in range(n_rows):
            first = random.choice(first_names)
            last = random.choice(last_names)
            gender = random.choice(["M","F"])
            city = random.choice(cities)
            state = random.choice(states)
            street = f"{random.randint(100, 9999)} {random.choice(streets)}"
            zip_code = str(random.randint(10000, 99999))
            city_lat, city_long = city_geo[city]
            lat = round(city_lat + random.uniform(-0.2, 0.2), 6)
            long = round(city_long + random.uniform(-0.2, 0.2), 6)
            city_pop = random.randint(50000, 8000000)
            job = random.choice(jobs)
            dob = random_date(1950, 2005)
            ss = "".join(str(random.randint(0, 9)) for _ in range(9))
            acct_num = "".join(str(random.randint(0, 9)) for _ in range(12))
            trans_num = str(uuid.uuid4())
            trans_date, trans_time = random_tx_datetime()
            categoty = random.choice(categories)
            base_amt = {
                "grocery": (10, 120),
                "gas": (20, 100),
                "online_purchase": (5, 400),
                "electronics": (50, 2000),
                "travel": (100, 4000),
                "restaurant": (10, 200),
                "utilities": (30, 400),
                "pharmacy": (5, 300),
            }[categoty]
            amt = round(random.uniform(*base_amt), 2)

            device = random.choice(devices)
            pay_method = random.choice(pay_methods)
            card_raw = luhn_16_digit()
            card_number = masked_card(card_raw)
            customer_tx_id = str(uuid.uuid4())

            merch_name = random.choice(merchants)
            merch_id = f"M{random.randint(10000, 99999)}"
            merch_url = f"https://www.{merch_name.lower()}.com"
            merch_ip = random_ip()
            merch_lat, merch_long = merch_geo[merch_name]
            mercj_lat = round(merch_lat + random.uniform(-0.3, 0.3), 6)
            merch_long = round(merch_long + random.uniform(-0.3, 0.3), 6)

            # Simple distance proxy
            dist = math.sqrt((lat - mercj_lat) ** 2 + (long - merch_long) ** 2)

            # Heuristic fraud assignment
            is_fraud = 0
            # Base fraud probability
            if random.random() < fraud_rate:
                is_fraud = 1
            # Boost fraud odds for unusual combos
            if categoty in ["electronics","travel"] and amt > 1000 and dist > 3:
                is_fraud = 1
            if device == "web" and pay_method == "wallet" and amt > 500 and random.random() < 0.3:
                is_fraud = 1

            writer.writerow([
                ss, first, last, gender, street, city, state, zip_code, lat, long,
                city_pop, job, dob, acct_num, trans_num, trans_date, trans_time,
                categoty, amt, device, pay_method, card_number, customer_tx_id,
                merch_name, merch_id, merch_url, merch_ip, mercj_lat, merch_long, is_fraud
            ])

    print(f"Wrote {n_rows} rows to {path}")

if __name__ == "__main__":
    # Change as needed
    generate_csv(path="train.csv", n_rows=1000, fraud_rate=0.03, seed=42)