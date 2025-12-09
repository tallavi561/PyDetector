# PyDetector

Simple Flask service using Poetry.

## Run Server

```bash
poetry install
poetry run start

POST: 
http://localhost:5000/api/v1/detectBase64Image
send the "example-message.json" as the body.