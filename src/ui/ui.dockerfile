FROM python:3.12-slim

# Step 1 - Install dependencies
WORKDIR /app

# Step 2 - Copy only requirements.txt
COPY requirements.txt /app
# Ensure the libs directory exists
RUN mkdir -p /app/libs
COPY libs/ /app/libs/

# Step 4 - Install pip dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --upgrade pip

# Step 5 - Copy the rest of the files
COPY . .
# ENV PYTHONUNBUFFERED=1

# Expose the application port
EXPOSE 80
RUN apt-get update && apt-get install -y git && apt-get install -y curl && apt-get install -y coreutils && rm -rf /var/lib/apt/lists/*
# do not change the arguments-y 
CMD ["streamlit", "run", "app.py", "--server.headless=true", "--server.port=80", "--server.address=0.0.0.0"]
