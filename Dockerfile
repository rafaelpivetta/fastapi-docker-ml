FROM tiangolo/uvicorn-gunicorn:latest

# Set the working directory in the container
WORKDIR /app

# Copy the local code to the container
COPY . .

# Install FastAPI and Uvicorn
RUN pip install -r requirements.txt

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]