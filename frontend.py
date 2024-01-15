import os
import tkinter as tk
from tkinter import ttk
from model import predict_sentiment


def save_and_analyze_sentiment():
    text_to_save_analyze = entry.get()
    save_to_txt(text_to_save_analyze)
    predicted_sentiment = predict_sentiment(text_to_save_analyze)
    status_label.config(text=f"Predicted Sentiment: {predicted_sentiment}")


def save_to_txt(text):
    file_path = os.path.join(os.path.dirname(__file__), "statement.txt")

    with open(file_path, 'w') as file:
        file.write(text)


# Create the main window
root = tk.Tk()
root.title("Text Saver and Sentiment Analyzer")

# Create GUI components
label = tk.Label(root, text="Enter a review:")
label.pack(pady=10)

entry = tk.Entry(root, width=40)
entry.pack(pady=10)

save_analyze_button = tk.Button(
    root, text="Analyze Sentiment", command=save_and_analyze_sentiment)
save_analyze_button.pack(pady=10)

status_label = tk.Label(root, text="")
status_label.pack(pady=10)

# Start the GUI application
root.mainloop()
