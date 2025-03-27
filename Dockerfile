# Use the official Python 3.12 slim image as the base image
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Install necessary system dependencies (git and wget)
RUN apt-get update && apt-get install -y git wget

# Create the model directory and download the model artifacts
RUN mkdir -p model && \
    wget https://huggingface.co/OpenVINO/Phi-3.5-vision-instruct-int8-ov/resolve/main/openvino_language_model.xml -O model/openvino_language_model.xml && \
    wget https://huggingface.co/OpenVINO/Phi-3.5-vision-instruct-int8-ov/resolve/main/openvino_language_model.bin -O model/openvino_language_model.bin

# Copy the application code and configuration files
COPY app.py config.py requirements.txt /app/

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Expose the port the app will run on
EXPOSE 8080

# Override the default entrypoint
ENTRYPOINT []

# Command to run your application
CMD ["python", "app.py"]
