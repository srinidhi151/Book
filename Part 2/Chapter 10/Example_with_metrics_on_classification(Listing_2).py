from sklearn.metrics import confusion_matrix
y_true = ["tiger", "elephant", "tiger", "tiger", "elephant", "peacock"]
y_pred = ["elephant", "elephant", "tiger", "tiger", "elephant", "tiger"]

print(confusion_matrix(y_true, y_pred, labels=["elephant", "peacock", "tiger"]))
