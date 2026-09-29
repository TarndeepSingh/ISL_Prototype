import json
import numpy as np

# Load LiteRT or TFLite
try:
    from ai_edge_litert.interpreter import Interpreter
    interpreter = Interpreter(model_path="isl_production_model.tflite")
except Exception:
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_path="isl_production_model.tflite")

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("=" * 50)
print("TFLITE MODEL DIAGNOSTIC")
print("=" * 50)
print("Expected Input Tensor Shape :", input_details[0]['shape'])
print("Expected Input Data Type    :", input_details[0]['dtype'])
print("Expected Output Shape       :", output_details[0]['shape'])

with open("labels.json", "r") as f:
    labels = json.load(f)

print(f"Total Loaded Classes in JSON: {len(labels)}")
print("Sample First 5 Classes      :", [labels.get(str(i)) for i in range(min(5, len(labels)))])
print("=" * 50)