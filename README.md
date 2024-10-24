TTS_FINETUNE_TELUGU

Overview
This project implements a Text-to-Speech (TTS) system using the SpeechT5 model fine-tuned on themozilla-foundation common_voice_17_0 dataset for telugu language. The goal is to generate natural-sounding speech from textual input, leveraging state-of-the-art machine learning techniques.

Introduction
Text-to-Speech (TTS) technology converts written text into spoken words, enabling computers to generate human-like speech. This technology has gained significant traction due to its diverse applications across various fields, including:

Accessibility: TTS assists visually impaired individuals by reading text aloud, enhancing their ability to interact with digital content.
Education: It facilitates language learning and literacy development by providing auditory feedback and pronunciation guidance.
Entertainment: TTS is utilized in audiobooks, video games, and virtual assistants, enhancing user experience through engaging and immersive interactions.
Customer Service: Automated voice response systems use TTS to provide information and support, improving efficiency and accessibility.
Importance of Fine-Tuning
While pre-trained models like SpeechT5 offer a robust starting point for TTS applications, fine-tuning on specific datasets is crucial for several reasons:

Personalization: Fine-tuning allows the model to adapt to specific voices, accents, and speaking styles, resulting in more natural and relatable speech output.

Domain Adaptation: Different applications may require distinct speech characteristics. Fine-tuning enables the model to better handle domain-specific terminology and contexts.

Quality Improvement: By training on curated datasets like LJ Speech, the model learns to generate clearer, more expressive, and emotionally nuanced speech, enhancing overall user satisfaction.

Performance Optimization: Fine-tuning can lead to better alignment of text and audio, reducing issues like mispronunciations or unnatural intonations, thus improving the model’s reliability in real-world applications.

In this project, we leverage the LJ Speech dataset to fine-tune the SpeechT5 model, aiming to create a TTS system that excels in delivering high-quality, lifelike speech tailored to specific user needs.

Methodology
This section outlines the detailed steps taken for model selection, dataset preparation, and fine-tuning of the Text-to-Speech (TTS) system using the SpeechT5 model.

1. Model Selection
Model Choice: SpeechT5

SpeechT5 is a transformer-based model designed for TTS tasks. It provides a balance of performance and efficiency, making it suitable for generating high-quality speech.
We selected SpeechT5 due to its ability to handle various speech generation tasks, including text-to-speech synthesis, making it versatile for our project.
2. Dataset Preparation
Dataset: mozilla-foundation common_voice_17_0
The Mozilla Common Voice dataset is a large, open-source collection of voice recordings designed to help develop speech recognition and synthesis systems. The Telugu segment of the dataset specifically focuses on capturing the nuances of the Telugu language, making it an invaluable resource for researchers and developers in the field of speech technology.

Key Features
Language: Telugu
Size: Contains thousands of audio clips and corresponding transcripts.
Audio Quality: Recordings are made in a controlled environment, ensuring high clarity and consistency.
Diversity: Includes a wide range of speakers, providing variation in accents and pronunciations.
Transcripts: Accompanying text that reflects the spoken content, allowing for effective training of ASR and TTS models

Data Download: Steps to Download the Telugu Dataset
Visit the Common Voice Website: Go to the Mozilla Common Voice page.

Select the Dataset Version: Find the latest version of the dataset (e.g., version 17.0) and select it.

Choose Language: Scroll to the language section and select Telugu from the list of available languages.

Download the Files:

The dataset is typically available in ZIP or TAR format.
Click on the download link to obtain the compressed file.
Extract the Files: Once downloaded, extract the contents to your desired directory using an archive manager (e.g., WinRAR, 7-Zip, or the command line).

Preprocessing:

Audio Processing:
Convert all audio files to a consistent format (16000 Hz sample rate, mono channel).
Normalize audio levels to ensure consistent volume across samples.
Text Cleaning:
Remove any extraneous characters or formatting from the transcripts to ensure clean input for the model.
Optionally, apply phonetic transcriptions for improved pronunciation.
Alignment:

Generate a mapping between audio files and their corresponding text transcripts, ensuring that each audio clip can be paired with its correct spoken text.
Train-Validation Split:

Split the dataset into training and validation sets (e.g., 90% training, 10% validation) to evaluate model performance during and after fine-tuning.
3. Fine-Tuning
Fine-Tuning Process:

Environment Setup:

Ensure that all necessary libraries and dependencies (e.g., PyTorch, Transformers, NumPy) are installed as specified in the requirements.txt file.
Fine-Tuning Script:

Model loading: Load the pre-trained SpeechT5 model.
Data loading: Use a data loader to read audio and text pairs for training.
Training loop: Implement a loop that iterates over the training dataset for a specified number of epochs, updating model weights based on the loss calculated from predictions.
Hyperparameter Configuration:

Set hyperparameters such as learning rate, batch size, and the number of epochs. Commonly used values might include:
Learning Rate: 5e-5
Batch Size: 16
Epochs: 2000
Monitoring:

Monitor training loss and validation metrics during the fine-tuning process to prevent overfitting. Use techniques like early stopping if validation loss starts to increase.
Model Saving:

After fine-tuning, save the trained model and any associated artifacts (e.g., tokenizer) for later use in generating speech.
4. Evaluation
Post-Fine-Tuning Evaluation:

After fine-tuning, evaluate the model using the validation dataset. Metrics for evaluation may include:
Mean Opinion Score (MOS): A subjective score based on human evaluations of audio quality.
Alignment and accuracy: Check how well the generated speech aligns with the input text.
By following this methodology, we can ensure that the SpeechT5 model is effectively fine-tuned on the LJ Speech dataset, resulting in a robust TTS system capable of producing high-quality speech outputs.

RESULTS:

Results In this section, we present the results of our Text-to-Speech (TTS) system, including both objective and subjective evaluations. We tested the model's performance on two types of speech: English technical speecH

Objective Evaluations Metrics Used: Mean Opinion Score (MOS): A numerical measure of perceived audio quality, usually rated on a scale from 1 (poor) to 5 (excellent). English Technical Speech Subjective evaluations were conducted through listener studies, where participants rated the audio samples generated by the model. The evaluations focused on clarity, naturalness, and overall satisfaction.
English Technical Speech Feedback Clarity: Most listeners appreciated the clarity of the speech, especially in technical contexts. Naturalness: While the speech was generally natural, some participants felt that the intonation could be improved to sound more conversational. Comments: “Very clear for technical topics, but sometimes feels robotic.”

THANK YOU
