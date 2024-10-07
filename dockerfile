# Use an official Python image as the base image
FROM python:3.8.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files into the container
COPY . .

# Expose the Flask port (5000)
EXPOSE 5000

# Command to run the Flask app using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "APIClassification:app"]
