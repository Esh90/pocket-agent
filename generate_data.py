"""
Synthetic training data generator for Pocket-Agent.
Reads teacher_examples.jsonl + hardcoded templates → train.jsonl (~320 examples).
"""

import json
import random
import re
import copy
from pathlib import Path
from datetime import date, timedelta

random.seed(42)

SYSTEM_PROMPT = (
    "You are a compact tool-calling assistant running fully offline on a mobile device. "
    "When the user's request clearly maps to one of the available tools, respond with exactly "
    "one tool call formatted as JSON inside <tool_call>...</tool_call> tags. "
    "When the request is chitchat, impossible to fulfil with the available tools, or truly "
    "ambiguous with no prior context to resolve it, respond with plain helpful text—no tool call.\n\n"
    "Available tools (schema is final):\n"
    '{"tool":"weather","args":{"location":"string","unit":"C|F"}}\n'
    '{"tool":"calendar","args":{"action":"list|create","date":"YYYY-MM-DD","title":"string?"}}\n'
    '{"tool":"convert","args":{"value":number,"from_unit":"string","to_unit":"string"}}\n'
    '{"tool":"currency","args":{"amount":number,"from":"ISO3","to":"ISO3"}}\n'
    '{"tool":"sql","args":{"query":"string"}}'
)


# ── helpers ──────────────────────────────────────────────────────────────────

def make_tool_call(tool: str, args: dict) -> str:
    return f"<tool_call>{json.dumps({'tool': tool, 'args': args}, ensure_ascii=False)}</tool_call>"


def rand_date(offset_range=(-10, 30)) -> str:
    today = date.today()
    delta = random.randint(*offset_range)
    return (today + timedelta(days=delta)).strftime("%Y-%m-%d")


def inject_typos(text: str, n: int = 1) -> str:
    """Randomly corrupt n characters in text (swap adjacent, double, or drop)."""
    words = text.split()
    if not words:
        return text
    for _ in range(n):
        idx = random.randrange(len(words))
        w = words[idx]
        if len(w) < 3:
            continue
        op = random.choice(["swap", "double", "drop"])
        ci = random.randint(1, len(w) - 2)
        if op == "swap":
            w = w[:ci] + w[ci + 1] + w[ci] + w[ci + 2:]
        elif op == "double":
            w = w[:ci] + w[ci] + w[ci:]
        else:
            w = w[:ci] + w[ci + 1:]
        words[idx] = w
    return " ".join(words)


# ── seed data loader ──────────────────────────────────────────────────────────

def load_seed(path: str = "teacher_examples.jsonl") -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    examples = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


# ── template banks ───────────────────────────────────────────────────────────

WEATHER_TEMPLATES = [
    # (user_text, location, unit)
    ("What's the weather in {loc}?", "{loc}", "C"),
    ("Is it raining in {loc} today?", "{loc}", "C"),
    ("Weather forecast for {loc} please", "{loc}", "C"),
    ("Temperature in {loc} right now", "{loc}", "C"),
    ("How hot is it in {loc}?", "{loc}", "C"),
    ("Tell me the weather in {loc} in Fahrenheit", "{loc}", "F"),
    ("Give me the weather for {loc} in °F", "{loc}", "F"),
    ("What should I wear in {loc} today?", "{loc}", "C"),
    ("Is {loc} cold right now?", "{loc}", "C"),
    ("Current conditions in {loc}", "{loc}", "C"),
    ("aaj {loc} mein mausam kaisa hai", "{loc}", "C"),          # Urdu
    ("mujhe {loc} ka mausam batao", "{loc}", "C"),               # Urdu
    ("{loc} mein aaj barish hogi kya", "{loc}", "C"),            # Urdu
    ("¿Qué tiempo hace en {loc}?", "{loc}", "C"),                # Spanish
    ("{loc} ka mosam batao Fahrenheit mein", "{loc}", "F"),      # Urdu+English
    ("weather batao {loc} ka", "{loc}", "C"),                    # Hindi/Urdu+English
    ("What is the weatehr in {loc}", "{loc}", "C"),              # typo bait
    ("wheather in {loc} please", "{loc}", "C"),                  # typo
]

LOCATIONS = [
    "Karachi", "Lahore", "Islamabad", "London", "New York",
    "Tokyo", "Dubai", "Paris", "Berlin", "Sydney",
    "Mumbai", "Delhi", "Cairo", "Istanbul", "Singapore",
    "Toronto", "Los Angeles", "Chicago", "Bangkok", "Seoul",
]

CALENDAR_LIST_TEMPLATES = [
    ("What's on my calendar for {date}?", "list", "{date}", None),
    ("Show me my schedule for {date}", "list", "{date}", None),
    ("Any meetings on {date}?", "list", "{date}", None),
    ("What do I have planned on {date}?", "list", "{date}", None),
    ("List my events for {date}", "list", "{date}", None),
    ("{date} ko kya schedule hai mera", "list", "{date}", None),  # Urdu
    ("Mere {date} ke events dikhao", "list", "{date}", None),     # Hindi
]

CALENDAR_CREATE_TEMPLATES = [
    ("Schedule '{title}' on {date}", "create", "{date}", "{title}"),
    ("Add '{title}' to my calendar on {date}", "create", "{date}", "{title}"),
    ("Create an event called '{title}' on {date}", "create", "{date}", "{title}"),
    ("Book '{title}' for {date}", "create", "{date}", "{title}"),
    ("Put '{title}' on {date} in my calendar", "create", "{date}", "{title}"),
    ("{date} ko '{title}' calendar mein add karo", "create", "{date}", "{title}"),  # Urdu
    ("Set up a meeting for '{title}' on {date}", "create", "{date}", "{title}"),
    ("Remind me about '{title}' on {date}", "create", "{date}", "{title}"),
]

CALENDAR_TITLES = [
    "Team standup", "Doctor appointment", "Dentist visit", "Project review",
    "Lunch with Alex", "Birthday party", "Flight to Dubai", "Gym session",
    "Client call", "Family dinner", "Interview", "Budget review",
    "Product demo", "Weekly sync", "Code review",
]

CONVERT_TEMPLATES = [
    ("Convert {val} {fu} to {tu}", "{val}", "{fu}", "{tu}"),
    ("How many {tu} is {val} {fu}?", "{val}", "{fu}", "{tu}"),
    ("What is {val} {fu} in {tu}?", "{val}", "{fu}", "{tu}"),
    ("{val} {fu} ko {tu} mein convert karo", "{val}", "{fu}", "{tu}"),   # Urdu
    ("{val} {fu} kitne {tu} hote hain", "{val}", "{fu}", "{tu}"),         # Hindi
    ("Change {val} {fu} to {tu}", "{val}", "{fu}", "{tu}"),
    ("I have {val} {fu}, what is that in {tu}?", "{val}", "{fu}", "{tu}"),
    ("{val} {fu} convierte a {tu}", "{val}", "{fu}", "{tu}"),              # Spanish
    ("conver {val} {fu} to {tu}", "{val}", "{fu}", "{tu}"),               # typo
]

CONVERT_PAIRS = [
    (5, "km", "miles"), (100, "meters", "feet"), (70, "kg", "pounds"),
    (32, "Celsius", "Fahrenheit"), (212, "Fahrenheit", "Celsius"),
    (1, "mile", "km"), (500, "grams", "ounces"), (2.5, "liters", "gallons"),
    (1000, "watts", "horsepower"), (180, "cm", "inches"),
    (3, "feet", "meters"), (10, "kg", "lbs"), (60, "mph", "kph"),
    (1, "acre", "square meters"), (100, "ml", "fl oz"),
]

CURRENCY_TEMPLATES = [
    ("Convert {amt} {fc} to {tc}", "{amt}", "{fc}", "{tc}"),
    ("How much is {amt} {fc} in {tc}?", "{amt}", "{fc}", "{tc}"),
    ("What is {amt} {fc} worth in {tc}?", "{amt}", "{fc}", "{tc}"),
    ("{amt} {fc} ko {tc} mein convert karo", "{amt}", "{fc}", "{tc}"),    # Urdu
    ("{amt} {fc} kitne {tc} hain", "{amt}", "{fc}", "{tc}"),              # Urdu
    ("Exchange {amt} {fc} for {tc}", "{amt}", "{fc}", "{tc}"),
    ("I want to convert {amt} {fc} to {tc}", "{amt}", "{fc}", "{tc}"),
    ("{amt} {fc} en {tc} por favor", "{amt}", "{fc}", "{tc}"),            # Spanish
    ("currnecy convert {amt} {fc} to {tc}", "{amt}", "{fc}", "{tc}"),     # typo
]

CURRENCY_PAIRS = [
    (100, "USD", "PKR"), (50, "EUR", "USD"), (200, "GBP", "EUR"),
    (1000, "PKR", "USD"), (500, "SAR", "PKR"), (75, "USD", "EUR"),
    (1000, "INR", "USD"), (300, "AED", "USD"), (1, "BTC", "USD"),
    (250, "CAD", "GBP"), (100, "JPY", "USD"), (5000, "PKR", "EUR"),
]

SQL_TEMPLATES = [
    ("Show me all users who signed up last month",
     "SELECT * FROM users WHERE created_at >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month') AND created_at < DATE_TRUNC('month', CURRENT_DATE)"),
    ("Get total sales for each product",
     "SELECT product_id, SUM(amount) AS total_sales FROM sales GROUP BY product_id"),
    ("List the top 10 customers by revenue",
     "SELECT customer_id, SUM(amount) AS revenue FROM orders GROUP BY customer_id ORDER BY revenue DESC LIMIT 10"),
    ("Find all orders placed today",
     "SELECT * FROM orders WHERE DATE(created_at) = CURRENT_DATE"),
    ("Count active users",
     "SELECT COUNT(*) FROM users WHERE status = 'active'"),
    ("Show me pending invoices",
     "SELECT * FROM invoices WHERE status = 'pending'"),
    ("Get average order value",
     "SELECT AVG(amount) FROM orders"),
    ("List products with stock below 10",
     "SELECT * FROM products WHERE stock_quantity < 10"),
    ("Show sales by region",
     "SELECT region, SUM(revenue) FROM sales GROUP BY region"),
    ("Find duplicate email addresses",
     "SELECT email, COUNT(*) FROM users GROUP BY email HAVING COUNT(*) > 1"),
    ("Get last login for each user",
     "SELECT user_id, MAX(login_time) AS last_login FROM login_logs GROUP BY user_id"),
    ("Show revenue trend by month",
     "SELECT DATE_TRUNC('month', order_date) AS month, SUM(amount) FROM orders GROUP BY 1 ORDER BY 1"),
]

SQL_PARAPHRASES = [
    ("Run a query to get all users from last month",),
    ("Execute SQL: total sales grouped by product",),
    ("Database query: top 10 customers by revenue",),
    ("Query the DB for today's orders",),
    ("SQL mein active users count karo",),    # Urdu+SQL
    ("database se pending invoices nikalo",),  # Urdu
]

REFUSAL_EXAMPLES = [
    # chitchat
    ("Hey, how are you?", "I'm doing great, thanks for asking! How can I help you today?"),
    ("Tell me a joke", "Why don't scientists trust atoms? Because they make up everything! 😄"),
    ("What's the meaning of life?", "That's a deep philosophical question! Many say it's 42, but I think it's whatever gives your life purpose."),
    ("Who are you?", "I'm Pocket-Agent, an offline AI assistant on your device. I can help with weather, calendar, unit conversion, currency exchange, and database queries."),
    ("Thanks, that's helpful!", "You're welcome! Let me know if you need anything else."),
    ("Good morning!", "Good morning! How can I assist you today?"),
    ("What day is today?", "I don't have access to the current date without a tool—try checking your device clock."),
    ("Translate this to French: hello", "I don't have a translation tool available. You could try a translation app for that."),
    # impossible tools
    ("Send an email to john@example.com", "I don't have an email tool available. Please use your email app for that."),
    ("Play some music", "I can't control media playback—I don't have a music tool. Try your music app!"),
    ("Call my mom", "I don't have a calling tool. Please use your phone's dialer for that."),
    ("Take a photo", "I can't access your camera. Please use your phone's camera app."),
    ("Set a timer for 10 minutes", "I don't have a timer tool—your phone's built-in clock can do that easily."),
    ("Navigate to the airport", "I don't have navigation or maps tools available offline."),
    ("Book a flight to London", "I can't book flights—I don't have access to booking systems."),
    # ambiguous with no history
    ("Convert that", "I'd need more context—what value and units would you like me to convert?"),
    ("What about tomorrow?", "Could you clarify what you'd like to know about tomorrow?"),
    ("Do it again", "I'm not sure what you'd like me to repeat. Could you rephrase your request?"),
    ("Show me the results", "I need a bit more context—results from what query or action?"),
    # hallucination-bait entities
    ("What's the weather on planet Xorblax?", "Planet Xorblax doesn't appear to be a real location I can retrieve weather for. Could you give me a real city?"),
    ("Convert 5 flurbles to smooches", "I don't recognize 'flurbles' or 'smooches' as real units. Could you use standard units like km, kg, or Celsius?"),
    ("Currency exchange for 100 XZY to QQQ", "XZY and QQQ don't appear to be valid ISO currency codes. Please use standard codes like USD, EUR, or PKR."),
    # Urdu refusals
    ("mujhe ek joke sunao", "Yeh lo ek mazedar joke: ek aadmi library mein gaya aur bola 'mujhe chips do' — librarian ne kaha 'yeh library hai!' — aadmi ne dhimi awaaz mein kaha 'sorry... mujhe chips chahiyein.' 😄"),
    ("aap kaun hain", "Main Pocket-Agent hoon, aapke device par ek offline AI assistant. Main mausam, calendar, unit conversion, currency, aur database queries mein madad kar sakta hoon."),
    ("shukriya", "Khushi hui! Aur koi madad chahiye?"),
]

MULTI_TURN_EXAMPLES = [
    # weather → currency
    {
        "history": [
            {"role": "user", "content": "What's the weather in Dubai?"},
            {"role": "assistant", "content": make_tool_call("weather", {"location": "Dubai", "unit": "C"})},
        ],
        "user": "Now convert 500 AED to USD",
        "assistant": make_tool_call("currency", {"amount": 500, "from": "AED", "to": "USD"}),
    },
    # convert → "convert that to euros" ambiguity resolved by history
    {
        "history": [
            {"role": "user", "content": "How much is 200 USD in GBP?"},
            {"role": "assistant", "content": make_tool_call("currency", {"amount": 200, "from": "USD", "to": "GBP"})},
        ],
        "user": "Now convert that to euros",
        "assistant": make_tool_call("currency", {"amount": 200, "from": "USD", "to": "EUR"}),
    },
    # calendar create → list same day
    {
        "history": [
            {"role": "user", "content": "Add 'Doctor appointment' to my calendar on 2025-03-15"},
            {"role": "assistant", "content": make_tool_call("calendar", {"action": "create", "date": "2025-03-15", "title": "Doctor appointment"})},
        ],
        "user": "What else is on that day?",
        "assistant": make_tool_call("calendar", {"action": "list", "date": "2025-03-15"}),
    },
    # unit convert → "now in km" referencing prior value
    {
        "history": [
            {"role": "user", "content": "Convert 26.2 miles to km"},
            {"role": "assistant", "content": make_tool_call("convert", {"value": 26.2, "from_unit": "miles", "to_unit": "km"})},
        ],
        "user": "And what about 13.1 miles in km?",
        "assistant": make_tool_call("convert", {"value": 13.1, "from_unit": "miles", "to_unit": "km"}),
    },
    # sql follow-up
    {
        "history": [
            {"role": "user", "content": "Show me all active users"},
            {"role": "assistant", "content": make_tool_call("sql", {"query": "SELECT * FROM users WHERE status = 'active'"})},
        ],
        "user": "Now count them",
        "assistant": make_tool_call("sql", {"query": "SELECT COUNT(*) FROM users WHERE status = 'active'"}),
    },
    # ambiguous multi-turn refusal
    {
        "history": [],
        "user": "convert that to euros",
        "assistant": "I'd need more context—what amount and currency would you like to convert to euros?",
    },
    # weather follow-up unit switch
    {
        "history": [
            {"role": "user", "content": "weather in Berlin"},
            {"role": "assistant", "content": make_tool_call("weather", {"location": "Berlin", "unit": "C"})},
        ],
        "user": "give me that in Fahrenheit",
        "assistant": make_tool_call("weather", {"location": "Berlin", "unit": "F"}),
    },
    # 3-turn conversation
    {
        "history": [
            {"role": "user", "content": "What's the weather in Karachi?"},
            {"role": "assistant", "content": make_tool_call("weather", {"location": "Karachi", "unit": "C"})},
            {"role": "user", "content": "Show my schedule for tomorrow"},
            {"role": "assistant", "content": make_tool_call("calendar", {"action": "list", "date": "2025-03-20"})},
        ],
        "user": "Convert 1000 PKR to USD",
        "assistant": make_tool_call("currency", {"amount": 1000, "from": "PKR", "to": "USD"}),
    },
]


# ── builders ──────────────────────────────────────────────────────────────────

def build_example(messages: list[dict]) -> dict:
    return {"messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages}


def gen_weather_examples() -> list[dict]:
    examples = []
    for loc in LOCATIONS:
        for tmpl, _, unit in random.sample(WEATHER_TEMPLATES, min(3, len(WEATHER_TEMPLATES))):
            user_text = tmpl.replace("{loc}", loc)
            answer = make_tool_call("weather", {"location": loc, "unit": unit})
            examples.append(build_example([
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": answer},
            ]))
    return examples


def gen_calendar_examples() -> list[dict]:
    examples = []
    # list examples
    for _ in range(20):
        d = rand_date()
        tmpl, action, _, _ = random.choice(CALENDAR_LIST_TEMPLATES)
        user_text = tmpl.replace("{date}", d)
        answer = make_tool_call("calendar", {"action": "list", "date": d})
        examples.append(build_example([
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": answer},
        ]))
    # create examples
    for _ in range(25):
        d = rand_date((-5, 60))
        title = random.choice(CALENDAR_TITLES)
        tmpl, action, _, _ = random.choice(CALENDAR_CREATE_TEMPLATES)
        user_text = tmpl.replace("{date}", d).replace("{title}", title)
        answer = make_tool_call("calendar", {"action": "create", "date": d, "title": title})
        examples.append(build_example([
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": answer},
        ]))
    return examples


def gen_convert_examples() -> list[dict]:
    examples = []
    for val, fu, tu in CONVERT_PAIRS:
        for tmpl, _, _, _ in random.sample(CONVERT_TEMPLATES, min(3, len(CONVERT_TEMPLATES))):
            user_text = tmpl.replace("{val}", str(val)).replace("{fu}", fu).replace("{tu}", tu)
            answer = make_tool_call("convert", {"value": val, "from_unit": fu, "to_unit": tu})
            examples.append(build_example([
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": answer},
            ]))
    return examples


def gen_currency_examples() -> list[dict]:
    examples = []
    for amt, fc, tc in CURRENCY_PAIRS:
        for tmpl, _, _, _ in random.sample(CURRENCY_TEMPLATES, min(3, len(CURRENCY_TEMPLATES))):
            user_text = tmpl.replace("{amt}", str(amt)).replace("{fc}", fc).replace("{tc}", tc)
            answer = make_tool_call("currency", {"amount": amt, "from": fc, "to": tc})
            examples.append(build_example([
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": answer},
            ]))
    return examples


def gen_sql_examples() -> list[dict]:
    examples = []
    for user_text, query in SQL_TEMPLATES:
        answer = make_tool_call("sql", {"query": query})
        examples.append(build_example([
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": answer},
        ]))
    # paraphrase extras
    for (user_text,) in SQL_PARAPHRASES:
        # map to closest SQL template
        answer = make_tool_call("sql", {"query": SQL_TEMPLATES[0][1]})
        examples.append(build_example([
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": answer},
        ]))
    return examples


def gen_refusal_examples() -> list[dict]:
    examples = []
    for user_text, response in REFUSAL_EXAMPLES:
        examples.append(build_example([
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": response},
        ]))
    return examples


def gen_multiturn_examples() -> list[dict]:
    examples = []
    for mt in MULTI_TURN_EXAMPLES:
        messages = mt["history"] + [
            {"role": "user", "content": mt["user"]},
            {"role": "assistant", "content": mt["assistant"]},
        ]
        examples.append(build_example(messages))
    return examples


def gen_adversarial_examples() -> list[dict]:
    """Extra adversarial: typo-injected versions of normal queries."""
    examples = []
    base_queries = [
        ("What's the weatehr in Karachi?",
         make_tool_call("weather", {"location": "Karachi", "unit": "C"})),
        ("convret 100 USD to EUR",
         make_tool_call("currency", {"amount": 100, "from": "USD", "to": "EUR"})),
        ("scheduel a meeting 'Budget Review' on 2025-04-10",
         make_tool_call("calendar", {"action": "create", "date": "2025-04-10", "title": "Budget Review"})),
        ("conevrt 5 km to miels",
         make_tool_call("convert", {"value": 5, "from_unit": "km", "to_unit": "miles"})),
        ("waether in Lahore in Fahrenheeit",
         make_tool_call("weather", {"location": "Lahore", "unit": "F"})),
        # unit ambiguity: temperature context
        ("It's 98.6 degrees—convert to Celsius",
         make_tool_call("convert", {"value": 98.6, "from_unit": "Fahrenheit", "to_unit": "Celsius"})),
        # hallucination bait with real fallback
        ("weather in Islamabad please asap!!",
         make_tool_call("weather", {"location": "Islamabad", "unit": "C"})),
        # mixed script
        ("Karachi mein kal ka weather Fahrenheit mein batao",
         make_tool_call("weather", {"location": "Karachi", "unit": "F"})),
        ("London ka mausam C mein chahiye",
         make_tool_call("weather", {"location": "London", "unit": "C"})),
        ("500 rupees ko dollar mein convert karo",
         make_tool_call("currency", {"amount": 500, "from": "PKR", "to": "USD"})),
        ("aaj ke liye mera schedule dikhao",
         make_tool_call("calendar", {"action": "list", "date": date.today().strftime("%Y-%m-%d")})),
        ("kitne hain 200 gram in ounces",
         make_tool_call("convert", {"value": 200, "from_unit": "grams", "to_unit": "ounces"})),
        # number as word
        ("convert fifty kilometers to miles",
         make_tool_call("convert", {"value": 50, "from_unit": "km", "to_unit": "miles"})),
        ("one hundred dollars to rupees",
         make_tool_call("currency", {"amount": 100, "from": "USD", "to": "PKR"})),
    ]
    for user_text, answer in base_queries:
        examples.append(build_example([
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": answer},
        ]))
    return examples


def incorporate_seed(seed_examples: list[dict]) -> list[dict]:
    """Wrap teacher examples in the standard schema if they aren't already."""
    results = []
    for ex in seed_examples:
        # Expect format: {"messages": [...]} or {"prompt": ..., "completion": ...}
        if "messages" in ex:
            msgs = ex["messages"]
            # Ensure system prompt is ours
            if msgs and msgs[0].get("role") == "system":
                msgs[0]["content"] = SYSTEM_PROMPT
            else:
                msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + msgs
            results.append({"messages": msgs})
        elif "prompt" in ex and "completion" in ex:
            results.append(build_example([
                {"role": "user", "content": ex["prompt"]},
                {"role": "assistant", "content": ex["completion"]},
            ]))
    return results


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    seed = load_seed("teacher_examples.jsonl")
    print(f"Loaded {len(seed)} seed examples")

    all_examples: list[dict] = []
    all_examples += incorporate_seed(seed)
    all_examples += gen_weather_examples()
    all_examples += gen_calendar_examples()
    all_examples += gen_convert_examples()
    all_examples += gen_currency_examples()
    all_examples += gen_sql_examples()
    all_examples += gen_refusal_examples()
    all_examples += gen_multiturn_examples()
    all_examples += gen_adversarial_examples()

    # Deduplicate by first user message
    seen: set[str] = set()
    deduped: list[dict] = []
    for ex in all_examples:
        user_msgs = [m["content"] for m in ex["messages"] if m["role"] == "user"]
        key = user_msgs[0] if user_msgs else ""
        if key not in seen:
            seen.add(key)
            deduped.append(ex)

    random.shuffle(deduped)

    out_path = Path("train.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in deduped:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"✅  Wrote {len(deduped)} examples to {out_path}")

    # breakdown
    tool_counts: dict[str, int] = {t: 0 for t in ["weather", "calendar", "convert", "currency", "sql", "refusal"]}
    for ex in deduped:
        last_asst = next((m["content"] for m in reversed(ex["messages"]) if m["role"] == "assistant"), "")
        if "<tool_call>" in last_asst:
            try:
                tool = json.loads(last_asst.replace("<tool_call>", "").replace("</tool_call>", ""))["tool"]
                tool_counts[tool] = tool_counts.get(tool, 0) + 1
            except Exception:
                pass
        else:
            tool_counts["refusal"] += 1
    print("Distribution:", tool_counts)


if __name__ == "__main__":
    main()