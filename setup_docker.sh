#!/bin/bash

# Define the project directories
BACKEND_DIR="backend"
APP_DIR="rsg_hub_app"

# Build the Docker image for the backend
echo "Building the backend Docker image..."
docker build -t rsg_hub_backend $BACKEND_DIR

# Run the backend container
echo "Running the backend container..."
docker run -d -p 5000:5000 --name rsg_hub_backend_container rsg_hub_backend

# Build the Docker image for the React Native app
echo "Building the React Native Docker image..."
docker build -t rsg_hub_react_native $APP_DIR

# Run the React Native container
echo "Running the React Native container..."
docker run -d -p 8081:8081 --name rsg_hub_react_native_container rsg_hub_react_native

echo "Setup complete. Backend is running on port 5000 and React Native Metro Bundler is running on port 8081."
