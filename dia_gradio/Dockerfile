# Use a base image with PyTorch and CUDA support
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# Set the working directory
WORKDIR /app

# Install necessary Python packages
RUN pip install --no-cache-dir gradio transformers scipy

# Copy the application script into the container
COPY app.py .

# Expose the port Gradio will run on
EXPOSE 7860

# Command to run the application
CMD ["python", "app.py"]
