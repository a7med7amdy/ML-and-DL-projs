import os,io
from google.cloud import vision
from google.cloud.vision import types
import pandas as pd

def makeConnection(credentialPath):
    # link the path of the credential here
    os.environ['GOOGLE_APPLICATION_CREDENTIALS']= credentialPath
    client = vision.ImageAnnotatorClient()
    return client

def readImage(imagePath):
    with io.open(imagePath,'rb') as image_file: 
        content = image_file.read()
    image = vision.types.Image(content= content)
    return image

def makeTextDetection(image, client):
    response = client.text_detection(image= image,  image_context={"language_hints": [
        "af","sq","am","ar","hy","az","eu","be","bn","bs","bg","ca","ceb","ny","zh-cn","zh-tw","co","hr","cs","da",
        "nl","en","eo","et","tl","fi","fr","fy","gl","ka","de","el","gu","ht","ha","haw","iw","hi","hmn","hu","is",
        "ig","id","ga","it","ja","jw","kn","kk","km","ko","ku","ky","lo","la","lv","lt","lb","mk","mg","ms","ml",
        "mt","mi","mr","mn","my","ne","no","ps","fa","pl","pt","pa","ro","ru","sm","gd","sr","st","sn","sd","si",
        "sk","sl","so","es","su","sw","sv","tg","ta","te","th","tr","uk","ur","uz","vi","cy","xh","yi","yo","zu",
        "fil","he"]})
    textDic={}
    texts = response.text_annotations
    for i,text in enumerate(texts):
        textDic.setdefault(i, []).append(text.description)
        textDic.setdefault(i, []).append(text.bounding_poly)
    return textDic

def makeLogoDetection(image, client):
    response = client.logo_detection(image= image)
    logoDic={}
    logos = response.logo_annotations
    for i,logo in enumerate(logos):
        logoDic.setdefault(i, []).append(logo.description)
        logoDic.setdefault(i, []).append(logo.bounding_poly)
    return logoDic

def main(imagePath , credintialPath ):
    client = makeConnection(credintialPath)
    image = readImage(imagePath)
    textDetectionResponse = makeTextDetection(image, client)
    logoRecognitionResponse = makeLogoDetection(image, client)
    return bool(textDetectionResponse), bool(logoRecognitionResponse), textDetectionResponse, logoRecognitionResponse

# pass the image path and cred. path to main func
credentialPath=r'/home/ahmed-hamdy/intern work/google api/text and logo detection and recognition/avian-tract-283207-bc0721fc8622.json'
imagePath = "/home/ahmed-hamdy/intern work/google api/text and logo detection and recognition/mac.jpeg"
textFlag, logoFlag, textResponse, logoResponse= main(imagePath,credentialPath)

#textResponse and logoResponse are both dictionaries
#keys are numbers from 0 to len(dic)-1
#values are text/logo itself in index=0 and bounding box in index=1
if(textFlag): #means if it's not empty
    print(textResponse)
if(logoFlag): #means if it's not empty
    print(logoResponse)

# printing values of dict

#to print word number 1
print(textResponse[1][0])
#to print bounding box of word number 1
print(textResponse[1][1])

#to print logo number 0 (as if it'll be more than one logo in image)
print(textResponse[0][0])
#to print bounding box of word number 0
print(textResponse[0][1])


# note that the first text at textResponse contains all texts and its bounding box 
#is a large one surrounding them all