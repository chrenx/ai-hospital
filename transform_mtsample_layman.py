import pandas as pd
import requests
import ollama
from tqdm import tqdm


# Function to generate layman description using Ollama
def generate_layman_description(description, transcription):
    client = ollama.Client()
    model = "deepseek-v2:16b"
    prompt = f"""
        You are a patient who wants to explain your symptoms in your own words. Please rewrite the following medical description and transcription in plain, conversational language like a real patient would say. Avoid medical jargon and just talk about what you're feeling.

        DESCRIPTION:
        {description}

        TRANSCRIPTION:
        {transcription}

        Rewrite it as a short paragraph in layman terms.
        """
    
    response = client.generate(model=model, prompt=prompt, stream=False).response.strip()
    return response


# Prepare new dataset list
new_dataset = []
# Load the original dataset
data = pd.read_csv('datasets/mtsamples_internal_medicine.csv')

# Iterate through dataset and create layman entries
for idx, row in tqdm(data.iterrows(), total=len(data)):
    # Get layman description
    layman_description = generate_layman_description(row['description'], row['transcription'])

    # Prepare new row
    new_row = {
        'row_id': row['row_id'],
        'description': layman_description,
        'medical_specialty': row['medical_specialty'],
        'sample_name': row['sample_name'],  # original diagnosis
        'keywords': row['keywords']
    }
    new_dataset.append(new_row)

# Convert to DataFrame
layman_df = pd.DataFrame(new_dataset)

# Save to a new CSV
layman_df.to_csv('datasets/mtsamples_layman.csv', index=False)

print("✅ Dataset transformed and saved as 'mtsample_layman.csv'!")
