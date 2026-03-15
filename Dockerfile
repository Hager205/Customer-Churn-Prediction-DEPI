# Use official Python image
FROM python:3.10.14-slim-bullseye

# Set work directory
WORKDIR /app

# Upgrade system packages to reduce vulnerabilities
RUN apt-get update && apt-get upgrade -y

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy FastAPI app, model, and web assets
COPY fastapi_app.py ./
COPY model ./model
COPY app ./app

# Expose port 9000
EXPOSE 9000

# Run the FastAPI app with uvicorn
CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "9000"] 