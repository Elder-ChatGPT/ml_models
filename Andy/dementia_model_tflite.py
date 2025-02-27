import os
import math
import librosa
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
import logging

logger = logging.getLogger(__name__)

# Constants
DATASET_PATH = "media/dementia_audio"


# Function to extract MFCCs
def extract_mfcc(dataset_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=10):
    """Extracts MFCCs from music dataset and returns them along with genre labels.

    :param dataset_path (str): Path to dataset
    :param num_mfcc (int): Number of coefficients to extract
    :param n_fft (int): Interval we consider to apply FFT. Measured in # of samples
    :param hop_length (int): Sliding window for FFT. Measured in # of samples
    :param: num_segments (int): Number of segments we want to divide sample tracks into
    :return: data (dict): Dictionary containing mapping, labels, and MFCCs
    """
    SAMPLE_RATE = 22050
    TRACK_DURATION = 30  # measured in seconds
    SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION

    # Dictionary to store mapping, labels, and MFCCs
    data = {
        "mapping": [],
        "labels": [],
        "mfcc": []
    }

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    # Loop through all genre sub-folder
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        
        # Ensure we're processing a genre sub-folder level
        if dirpath is not dataset_path:

            # Save genre label (i.e., sub-folder name) in the mapping
            semantic_label = dirpath.split("/")[-1]
            data["mapping"].append(semantic_label)
            print("\nProcessing: {}".format(semantic_label))

            # Process all audio files in genre sub-dir
            for f in filenames:
                if f.endswith('.mp3'): 
                    # Load audio file
                    try:
                        file_path = os.path.join(dirpath, f)
                        signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

                        # Process all segments of audio file
                        for d in range(num_segments):
                            try:
                                # Calculate start and finish sample for current segment
                                start = samples_per_segment * d
                                finish = start + samples_per_segment

                                # Extract MFCC
                                mfcc = librosa.feature.mfcc(y=signal[start:finish], sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
                                mfcc = mfcc.T

                                # Store only MFCC feature with expected number of vectors
                                if len(mfcc) == num_mfcc_vectors_per_segment:
                                    data["mfcc"].append(mfcc.tolist())
                                    data["labels"].append(i-1)
                                    print("{}, segment:{}".format(file_path, d+1))
                                
                            except Exception as e:
                                print(f"Error processing segment {d+1}: {e}")
                                continue
                    except Exception as e:
                        print(f"Error processing file {f}: {e}")
                        continue

    return data

def predict(interpreter, X, y, input_details, output_details):

    X = X[np.newaxis, ...].astype(np.float32)

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], X)

    # Run the interpreter
    interpreter.invoke()

    # Get the output tensor
    prediction = interpreter.get_tensor(output_details[0]['index'])

    # Extract index with max value
    predicted_index = np.argmax(prediction, axis=1)
    print("Expected index: {}, Predicted index: {}".format(y, predicted_index))

    return predicted_index

def predictions(interpreter, data, input_details, output_details):
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])
    test_amount = len(X)
    correct_predictions = 0
    test_count = 0
    
    for i in range(test_amount): 

        # Make a prediction on a sample
        print("Sample number: " + str(test_count))
        X_sample = X[test_count]
        y_sample = y[test_count]
        predicted_index = predict(interpreter, X_sample, y_sample, input_details, output_details)
        if y_sample == predicted_index:
            correct_predictions += 1
        test_count += 1
        percentage_correct = correct_predictions / test_count
        print("Percent accuracy so far: " + str(percentage_correct * 100) + "%")

    percentage_correct = correct_predictions / test_amount
    print("Test accuracy was: " + str(percentage_correct * 100) + "%")

    return float(percentage_correct * 100)

def draw_vertical_line(image_path, output_path, percent):
    # Open the image
    img = Image.open(image_path)
    
    # Get the width and height of the image
    width, height = img.size
    
    # Calculate the x-coordinate for the vertical line
    x = int(width * (percent / 100))
    
    # Create a drawing context
    draw = ImageDraw.Draw(img)

    # Specify the path to a font file
    font_path = "memorytest/static/DejaVuSans.ttf"  # Change this to the path of the font file on your system
    
    # Load the font
    font = ImageFont.truetype(font_path, 20)  # Change size as needed
    
    # Draw the percentage text
    text = f"{percent}%"
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_y = 5  # Adjust this value to raise or lower the text
    draw.text((x - text_width - 5, text_y), text, fill="black", font=font)
    
    # Draw the vertical line
    draw.line((x, 0, x, height), fill="red", width=2)
    
    # Save the modified image
    img.save(output_path)

def main():
    # Extract MFCC data
    data = extract_mfcc(DATASET_PATH, num_segments=10)

    if not data["mfcc"]:
            logger.error("No MFCC data extracted")
            return 0.0

    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path='memorytest/model.tflite')
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Make predictions
    percent = predictions(interpreter, data, input_details, output_details)

    if percent is None:
            logger.error("Prediction returned None")
            return 0.0

    # Draw vertical line on image
    image_path = "memorytest/static/memorytest/images/overlapping_bell_curve.png"
    output_path = "memorytest/static/memorytest/images/overlapping_bell_curve_with_line.png"
    draw_vertical_line(image_path, output_path, percent)

    return percent

if __name__ == "__main__":
    main()
