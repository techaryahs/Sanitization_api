# Use official Python image
FROM python:3.10

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy all your code
COPY . .

# Expose port (important for Hugging Face Spaces)
EXPOSE 7860

# Tell Flask to run
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=7860

# Run Flask
CMD ["flask", "run"]
