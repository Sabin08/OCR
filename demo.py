import os, io
from google.cloud import vision
from google.cloud.vision_v1 import types
import pandas as pd

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'quiet-result-312506-7c8c1d594da2.json'

client = vision.ImageAnnotatorClient()


def detectText(img):
    with io.open(img, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations

    df = pd.DataFrame(columns=['locale', 'description'])
    for text in texts:
        df = df.append(
            dict(
                locale=text.locale,
                description=text.description
            ),
            ignore_index=True
        )
    return df         
FILE_NAME = 'nks.jpg'
FILE_PATH = r'C:\Users\Airos\Desktop\harrashment'
print(detectText(os.path.join(FILE_PATH, FILE_NAME)))
#print(df['description'][5])

