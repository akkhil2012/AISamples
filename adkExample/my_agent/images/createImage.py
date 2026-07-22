import argparse
import base64
import json
import os
import re
from typing import List, Dict

from openai import OpenAI
from PIL import Image


class ImageIdentifier:
    def __init__(self, model_name: str = "openai/gpt-4o-mini"):
        self.model_name = model_name
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
        print(f"Using OpenRouter model: {self.model_name}")

    def _encode_image(self, image_path: str) -> str:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode("utf-8")

        image = Image.open(image_path)
        mime_type = image.get_format_mimetype() or "image/jpeg"
        return f"data:{mime_type};base64,{encoded}"

    def _parse_results(self, text: str, top_k: int) -> List[Dict[str, float]]:
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [
                    {
                        "label": str(item.get("label", "unknown")),
                        "score": float(item.get("score", 0.0)),
                    }
                    for item in parsed[:top_k]
                ]
        except Exception:
            pass

        try:
            match = re.search(r"\[(.*?)\]", text, re.S)
            if match:
                parsed = json.loads(f"[{match.group(1)}]")
                return [
                    {
                        "label": str(item.get("label", "unknown")),
                        "score": float(item.get("score", 0.0)),
                    }
                    for item in parsed[:top_k]
                ]
        except Exception:
            pass

        return [{"label": "unknown", "score": 0.0}]

    def identify(self, image_path: str, top_k: int = 5):
        image_data_url = self._encode_image(image_path)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an image classifier. Analyze the supplied image "
                        "and return a JSON array of up to {top_k} objects. "
                        "Each object must have 'label' and 'score' fields."
                    ).format(top_k=top_k),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"Classify this image into at most {top_k} categories. "
                                "Return only valid JSON."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data_url},
                        },
                    ],
                },
            ],
            temperature=0,
        )

        content = response.choices[0].message.content or ""
        results = self._parse_results(content, top_k)

        print("\nPrediction Results")
        print("-" * 40)
        for idx, result in enumerate(results, start=1):
            print(f"{idx}. {result['label']} (confidence: {result['score']:.4f})")

        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to image")
    parser.add_argument("--top_k", type=int, default=5, help="Number of predictions")

    args = parser.parse_args()

    identifier = ImageIdentifier()
    identifier.identify(args.image, args.top_k)


if __name__ == "__main__":
    main()