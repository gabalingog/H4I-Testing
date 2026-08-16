from google.cloud import vision

client = vision.ImageAnnotatorClient()

with open("test-photo-2.jpg", "rb") as f:
    content = f.read()

image = vision.Image(content=content)
response = client.label_detection(image=image)

for label in response.label_annotations:
    print(label.description, label.score)