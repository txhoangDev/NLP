import json 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    
incorrect_records = []
    
with open("eval_output/eval_predictions.jsonl", 'r') as file:
    data = [json.loads(line) for line in file]
    
    for record in data:
        if record.get("predicted_label") != record.get("label"):
            print(record)
            incorrect_records.append(record)
            
with open("incorrect_predictions.jsonl", 'w') as file:
    for record in incorrect_records:
        json.dump(record, file)
        file.write('\n')
        
true_labels = [record.get("label") for record in incorrect_records]
predicted_labels = [record.get("predicted_label") for record in incorrect_records]
classes = [0, 1, 2]
conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=classes)

disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=classes)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()
