from openai import OpenAI
import os

# Set your API key 
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_explanation(data, prediction, reasons):
    try:
        prompt = f"""
        Prediction: {prediction}
        Key Factors: {", ".join(reasons)}

        Customer Data: {data}

        Explain briefly why the customer will churn or not.
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a business analyst."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100
        )

        return response.choices[0].message.content

    except Exception as e:
        return "LLM unavailable (quota or API issue)"