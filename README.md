# LLM-NewsClassifier

LLM-NewsClassifier is a text classification project that uses **DistilBERT**, a lightweight transformer model, to categorize news articles into different categories. The project leverages **PyTorch, Transformers, and Pandas** for preprocessing and model training.

## Features
- Loads a dataset of news articles from an **S3 bucket**.
- Preprocesses and tokenizes the data using **DistilBERTTokenizer**.
- Trains a deep learning model to classify news into categories (e.g., Business, Health, Science, Entertainment).
- Evaluates model performance and makes predictions on new data.

## Installation
To run this project locally, install the required dependencies:

```bash
pip install torch transformers pandas numpy tqdm jupyter argparse
```

or 

```bash
pip install -r requirements.txt
```

## Dataset
The dataset is loaded from an **S3 bucket**:

```python
s3_path = 'https://your-s3-bucket.s3.us-east-1.amazonaws.com/newsCorpora.csv'
df = pd.read_csv(s3_path, sep='\t', names=['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP'])
df = df[['TITLE', 'CATEGORY']]
```

## Model Training
The project uses **DistilBERT** for text classification:

```python
from transformers import DistilBertTokenizer, DistilBertModel

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")
```

## Running the Notebook
Start a Jupyter Notebook and execute the provided Python scripts:

```bash
jupyter notebook
```

## Future Improvements
- Fine-tune the DistilBERT model for better accuracy.
- Experiment with larger datasets.
- Deploy the model using **AWS SageMaker** or **Flask API**.

## License
This project is open-source under the **MIT License**.

## Contributing
Feel free to submit issues or pull requests to improve the project!

