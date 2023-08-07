# Required Libraries
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer

import string
import nltk
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


class TextSummarizer:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.trainer = None

    def preprocess_text(self, text):
        # Convert to lower case
        text = text.lower()

        # Remove HTML tags
        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text(separator=" ")

        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(text)
        filtered_text = [word for word in word_tokens if word not in stop_words]

        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        lemmatized = [lemmatizer.lemmatize(word) for word in filtered_text]

        return ' '.join(lemmatized)
    
    def preprocess_function(self, examples):
        inputs = [self.preprocess_text(doc) for doc in examples['article']]
        model_inputs = self.tokenizer(inputs, max_length=512, truncation=True)

        # Setup the tokenizer for targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(examples['highlights'], max_length=150, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def load_and_preprocess(self, train_path, val_path):
        dataset = load_dataset('csv', data_files={'train': train_path, 'validation': val_path})
        dataset = dataset.map(self.preprocess_function, batched=True)
        return dataset

    def train(self, dataset, output_dir='./results', num_train_epochs=3, per_device_train_batch_size=4, 
              per_device_eval_batch_size=4, warmup_steps=500, weight_decay=0.01, logging_dir='./logs'):
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            logging_dir=logging_dir,
        )

        self.trainer = Trainer(
            model=self.model,                         
            args=training_args,                  
            train_dataset=dataset['train'],         
            eval_dataset=dataset['validation']     
        )

        self.trainer.train()

    def summarize(self, text, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True):
        inputs = self.tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
        outputs = self.model.generate(inputs, max_length=max_length, min_length=min_length, length_penalty=length_penalty, 
                                      num_beams=num_beams, early_stopping=early_stopping)
        return self.tokenizer.decode(outputs[0])

# Example Usage
if __name__ == "__main__":
    summarizer = TextSummarizer('t5-base')
    dataset = summarizer.load_and_preprocess('archive/cnn_dailymail/train.csv', 'archive/cnn_dailymail/validation.csv')
    summarizer.train(dataset)
    text = """
Artificial intelligence (AI) is intelligence demonstrated by machines, unlike the natural intelligence displayed by humans and animals, which involves consciousness and emotionality. The distinction between the former and the latter categories is often revealed by the acronym chosen. 'Strong' AI is usually labelled as AGI (Artificial General Intelligence) while attempts to emulate 'natural' intelligence have been called ABI (Artificial Biological Intelligence). Leading AI textbooks define the field as the study of 'intelligent agents': any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals. Colloquially, the term 'artificial intelligence' is often used to describe machines (or computers) that mimic 'cognitive' functions that humans associate with the human mind, such as 'learning' and 'problem solving'.
"""
    summary = summarizer.summarize(text)
    print(summary)
