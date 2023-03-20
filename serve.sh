#!/usr/bin/env sh

# Set environment variable for the tracking URL where the Model Registry resides
export MLFLOW_TRACKING_URI=http://localhost:9487

# Serve the production model from the model registry
mlflow models serve -m "models:/spell-correction/1" -h 0.0.0.0 -p 9488