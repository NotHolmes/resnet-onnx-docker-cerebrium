import base64
import requests

def main():
    image_path = "images/n01667114_mud_turtle.JPEG"

    # Read and encode the image as base64
    with open(image_path, "rb") as img_file:
        b64_image = base64.b64encode(img_file.read()).decode('utf-8')

    # Prepare the payload
    payload = {
        "image": b64_image
    }

    # Send POST request to the prediction endpoint
    response = requests.post("http://localhost:8000/v1/predict", json=payload)

    # Print the response
    print("Status code:", response.status_code)
    print("Response:", response.json())

if __name__ == "__main__":
    main()