# Use the official Python image as the base image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI app and Streamlit dashboard files to the container
COPY api.py .
COPY dashboard.py .
COPY models/logreg_model.joblib ./models/logreg_model.joblib

# Install the 'dill' package
RUN pip install dill

# Expose the necessary ports
EXPOSE 8000 8501

# Set the command to run the FastAPI app
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]


#

