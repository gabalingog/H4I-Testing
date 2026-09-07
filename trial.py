# This is a basic object detection that runs in the terminal
# Use this to check if the API is working on your end to not rack up API credits

from google.cloud import vision

client = vision.ImageAnnotatorClient()

with open("photos/Public Garden, spring, flowering trees and swanboats.jpg", "rb") as f:
    content = f.read()

image = vision.Image(content=content)
response = client.label_detection(image=image)

for label in response.label_annotations:
    print(label.description, label.score)