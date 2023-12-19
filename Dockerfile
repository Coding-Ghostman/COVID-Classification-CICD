# Use an official PyTorch runtime as a parent image
FROM pytorch/pytorch:latest

# Set the working directory to /app
WORKDIR /app

# Install build-essential for gcc and other build tools
RUN apt-get update && apt-get install -y build-essential

# Copy the current directory contents into the container at /app
ADD . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/static/upload
# Make port 8080 available to the world outside this container
EXPOSE 8080

# Run app.py when the container launches
CMD ["python","App.py"]
