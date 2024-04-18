import gradio as gr
import numpy as np
import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertModel
import pandas as pd

# Load your data from new_bangla.csv
df = pd.read_csv("new_bangla.csv")  # Update the filename here

# Extract text and label columns
sentences = df["text"]
labels = df["label"]

# Load tokenizer and model
distil_bert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
distil_bert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')

# Define custom objects for model loading
custom_objects = {'TFDistilBertModel': TFDistilBertModel}

# Load the saved model with custom objects
model = tf.keras.models.load_model("bangla_fake.h5", custom_objects=custom_objects)

# Define Gradio interface
def classify_bangla_fake_news(description):
    input_ids = distil_bert_tokenizer.encode(description, add_special_tokens=True, max_length=40, truncation=True, padding='max_length')
    input_mask = [1] * len(input_ids)
    input_ids = np.asarray(input_ids).reshape(1, -1)
    input_mask = np.asarray(input_mask).reshape(1, -1)
    prediction = model.predict([input_ids, input_mask])[0]
    predicted_class = np.argmax(prediction)
    return "Fake" if predicted_class == 0 else "Real"

iface = gr.Interface(
    fn=classify_bangla_fake_news,
    inputs="text",
    outputs="label",
    title="Bangla Fake News Detection",
    description="Enter a Bangla news article and get prediction whether it's real or fake."
)

# Launch the Gradio interface
iface.launch(inline=False)
