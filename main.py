from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import os

# Keeping CORS middleware for potential cross-origin requests
app = FastAPI()

# Allowing CORS for specified origins
origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Path to the saved model directory
saved_model_dir = '/home/zakidev/Desktop/FastAPI/Models/8'

# Load the model using TFSMLayer
model_layer = tf.keras.layers.TFSMLayer(saved_model_dir, call_endpoint='serving_default')

# Define class names
CLASS_NAMES = ['Cassava Mosaic Disease (CMD)','Cassava___bacterial_blight','Cassava___healthy']


@app.get("/ping")
async def ping():
    return "Hello, I am alive"


def read_file_as_image(data) -> np.ndarray:
    """
    Read the uploaded file as an image.
    """
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict(
        file: UploadFile = File(...)
):
    try:
        # Read the image file
        image = read_file_as_image(await file.read())

        # Prepare the image for prediction
        img_batch = np.expand_dims(image, 0)  # Assuming batch size of 1

        # Make prediction using the model layer
        predictions = model_layer(img_batch)

        # Handle different prediction formats
        if isinstance(predictions, dict):
            predicted_class_index = np.argmax(predictions['dense_1'].numpy())
            confidence = float(predictions['dense_1'][0][predicted_class_index]) * 100
        else:
            predicted_class_index = np.argmax(predictions['dense_1'].numpy())
            confidence = float(predictions['dense_1'][0][predicted_class_index]) * 100

        # Get the predicted class label
        predicted_class = CLASS_NAMES[predicted_class_index]

        # Format confidence to display as a percentage with two decimal places
        confidence_formatted = "{:.2f}%".format(confidence)

        # Create the response dictionary
        response = {
            "Predicted": predicted_class,
            "Confidence": confidence_formatted
        }

        # Return the response
        return response

    except Exception as e:
        # Handle errors
        error_message = f"Error occurred: {str(e)}"
        return {"error": error_message}


if __name__ == "__main__":
    # Get the port number from the environment variable or use a default value
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host='localhost', port=port)
