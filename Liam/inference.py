from joblib import load # type: ignore
import os

# set model path to current directory
ML_MODELS_PATH = os.path.join(os.path.dirname(__file__), 'ml_models')

model_path = os.path.join(ML_MODELS_PATH, 'tfidf_logistic.joblib')
vectorizer_path = os.path.join(ML_MODELS_PATH, 'tfidf_vectorizer.joblib')

#raise error if models or vectoriser do not exist
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

model = load(model_path)
vectorizer = load(vectorizer_path)

def classify_text(input_text, model, vectorizer):

    # uncased
    cleaned_input = input_text.lower()

    # only words or characters 
    cleaned_input = ''.join([char for char in cleaned_input if char.isalpha() or char.isspace()])

    # convert to tfidf tokens
    input_vectorized = vectorizer.transform([cleaned_input])
    
    # run the model to predict 0 for control or 1 for dementia
    prediction = model.predict(input_vectorized)

    predict_proba = model.predict_proba(input_vectorized)
    
    # Format probability as a percentage of the predicted class
    probability = round(predict_proba[0][prediction[0]] * 100, 2)

    return prediction[0], probability