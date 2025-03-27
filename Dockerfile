# Dockerfile
FROM python:3.10-slim

# Install dependencies
RUN pip install --upgrade pip
RUN pip install transformers optimum-intel openvino

# Copy model and application code
COPY . /app

# Run the application
CMD ["python", "app.py"]
