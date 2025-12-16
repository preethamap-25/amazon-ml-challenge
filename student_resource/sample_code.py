import os
import pandas as pd
import numpy as np
from keras.models import load_model
from PIL import Image
import requests
from io import BytesIO

# Load your .h5 model once when the script starts
MODEL_PATH = 'student_resource/src/best_model_fold1.h5'  # Replace with your actual model path
model = load_model(MODEL_PATH)

# Define expected input size for the image (change as per your model's requirement)
IMAGE_SIZE = (224, 224)

def preprocess_text(catalog_content):
    '''
    Implement actual text preprocessing/tokenization used in your model training here.
    For example, convert text to sequences or embeddings as needed by the model.
    '''
    # Example placeholder: raise error to remind to implement proper preprocessing
    raise NotImplementedError("Implement text preprocessing according to your model.")

def preprocess_image(image_link):
    '''
    Download image from URL, resize and normalize it.
    Return as a numpy array formatted for model input.
    '''
    response = requests.get(image_link)
    img = Image.open(BytesIO(response.content)).convert('RGB')
    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img) / 255.0  # Normalize pixel values to [0,1]
    return np.expand_dims(img_array, axis=0)  # Add batch dimension

def predictor(sample_id, catalog_content, image_link):
    '''
    Calls model to predict price based on catalog text and image.
    '''
    # Preprocess inputs
    # Uncomment and implement text preprocessing if model requires text input
    # text_features = preprocess_text(catalog_content)

    # Preprocess image
    image_array = preprocess_image(image_link)

    # Example for model that takes only image input:
    prediction = model.predict(image_array)

    # If your model takes multiple inputs, pass them as a list, e.g.:
    # prediction = model.predict([image_array, text_features])

    price = float(prediction[0][0])  # Adjust indexing based on model output shape

    return round(price, 2)


if __name__ == "__main__":
    DATASET_FOLDER = 'dataset/'
    
    # Read test data
    test = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))
    
    # Apply predictor function to each row
    test['price'] = test.apply(
        lambda row: predictor(row['sample_id'], row['catalog_content'], row['image_link']),
        axis=1
    )
    
    # Select only required columns for output
    output_df = test[['sample_id', 'price']]
    
    # Save predictions
    output_filename = os.path.join(DATASET_FOLDER, 'test_out.csv')
    output_df.to_csv(output_filename, index=False)
    
    print(f"Predictions saved to {output_filename}")
    print(f"Total predictions: {len(output_df)}")
    print(f"Sample predictions:\n{output_df.head()}")
