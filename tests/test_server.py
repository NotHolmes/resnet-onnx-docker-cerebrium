import argparse
import base64
from pathlib import Path

import requests

DEFAULT_URL = "https://api.cortex.cerebrium.ai/v4/p-fecb2032/mtailor-assessment/v1/predict"
SAMPLE_IMAGE = Path(__file__).parent.parent / "images" / "n01667114_mud_turtle.JPEG"
REQUEST_TIMEOUT = 10


def predict(api_key: str, url: str, image_path: str) -> None:
    """Send a prediction request to the specified API endpoint with the given image."""
    # Read and encode the image as base64
    with open(image_path, "rb") as img_file:
        b64_image = base64.b64encode(img_file.read()).decode("utf-8")

    payload = {"image": b64_image}
    headers = {"Authorization": f"Bearer {api_key}"}

    print(f"Sending request to {url} ...")
    response = requests.post(url, json=payload, headers=headers, timeout=10)
    print("Status code:", response.status_code)
    try:
        print("Response:", response.json())
    except Exception:  # noqa
        print("Raw response:", response.text)


def run_custom_tests(api_key: str, url: str) -> None:
    """Run a set of preset test cases against the remote endpoint."""
    import json

    test_images = [
        ("n01440764_tench.jpeg", "tench"),
        ("n01667114_mud_turtle.JPEG", "mud turtle"),
    ]
    images_dir = Path(__file__).parent.parent / "images"
    for fname, expected_label in test_images:
        image_path = images_dir / fname
        print(f"\nTesting {fname} (expecting: {expected_label})...")

        with open(image_path, "rb") as img_file:
            b64_image = base64.b64encode(img_file.read()).decode("utf-8")

        payload = {"image": b64_image}
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.post(url, json=payload, headers=headers, timeout=REQUEST_TIMEOUT)

        print("Status code:", response.status_code)

        try:
            resp_json = response.json()
            print("Response:", json.dumps(resp_json, indent=2))
            # Optionally, check if expected_label is in the response
            if expected_label.lower() in json.dumps(resp_json).lower():
                print("PASS: Expected label found in response.")
            else:
                print("FAIL: Expected label NOT found in response.")
        except Exception:  # noqa
            print("Raw response:", response.text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test deployed API.")
    parser.add_argument("--api_key", required=True, help="API key")
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"API endpoint URL (default: {DEFAULT_URL})",
    )
    parser.add_argument(
        "--image_path",
        default=str(SAMPLE_IMAGE),
        help=f"Path to image file to send (default: {SAMPLE_IMAGE})",
    )
    parser.add_argument(
        "--run_tests",
        action="store_true",
        help="Run preset custom tests against the endpoint instead of a single prediction.",
    )
    args = parser.parse_args()

    if args.run_tests:
        run_custom_tests(api_key=args.api_key, url=args.url)
    else:
        predict(api_key=args.api_key, url=args.url, image_path=args.image_path)
