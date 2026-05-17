# ── 11. LLM-based data validation (GenAI Certification Engine) 
# IBM: automated data validation using LLM-based reasoning
import ollama, json
 
def validate_record(record: dict, rules: list[str]) -> dict:
    rules_str = "\n".join(f"- {r}" for r in rules)
    prompt = f"""You are a data quality validator.
Check this record against the rules and return JSON:
{{"valid": true/false, "violations": ["...", ...]}}
 
Record: {json.dumps(record)}
Rules:
{rules_str}
Return ONLY valid JSON."""
    response = ollama.chat(
        model="llama3.2", format="json",
        messages=[{"role": "user", "content": prompt}],
    )
    return json.loads(response["message"]["content"])
 
record = {"card_number": "4111111111111111", "amount": -50, "currency": "USD"}
rules  = ["amount must be positive", "currency must be 3-letter ISO code"]
print(validate_record(record, rules))