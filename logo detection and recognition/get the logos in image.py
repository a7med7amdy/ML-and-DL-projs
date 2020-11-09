import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models as models
torch.set_printoptions(linewidth=120)
import matplotlib.pyplot as plt
from torch.autograd import Variable
import cv2 as cv2
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import random
import torchvision.transforms as transforms
from pathlib import Path
import sys
import PIL.ImageOps 

import os,io
from google.cloud import vision_v1
from google.cloud import vision
from google.cloud.vision_v1 import types
import pandas as pd
import skimage.io as ios
import matplotlib.patches as patches
from PIL import Image

def makeConnection(credentialPath):
    # link the path of the credential here
    os.environ['GOOGLE_APPLICATION_CREDENTIALS']= credentialPath
    client = vision.ImageAnnotatorClient()
    return client

def readImage(imagePath):
    with io.open(imagePath,'rb') as image_file: 
        content = image_file.read()
    image = vision_v1.types.Image(content= content)
    return image

def makeLogoDetection(image, client):
    response = client.logo_detection(image= image)
    logoDic={}
    logos = response.logo_annotations
    for i,logo in enumerate(logos):
        logoDic.setdefault(i, []).append(logo.description)
        logoDic.setdefault(i, []).append(logo.bounding_poly)
    return logoDic

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

def recognize(client,imagePath):
    image = readImage(imagePath)
    textDetectionResponse = makeTextDetection(image, client)
    logoRecognitionResponse = makeLogoDetection(image, client)
    return bool(logoRecognitionResponse),logoRecognitionResponse,bool(textDetectionResponse),textDetectionResponse

# imagePath = "/home/ahmed-hamdy/intern work/google api/text and logo detection and recognition/mac.jpeg"

def getImageLogo(client,imagePath,device,modelWeightPath="model_47.pth"):
    
    log=['AM-007', 'AM-028', 'AM-030', 'AM-032', 'AM-033', 'AM-034', 'AM-035', 'AM-036', 'AM-037', 'AM-038', 'AM-039',
     'AM-040', 'AM-041', 'AM-042', 'AM-043', 'AM-044', 'AM-045', 'AM-046', 'AM-047', 'AM-048', 'AM-049', 'AM-051',
     'AM-052', 'AM-054', 'AM-055', 'AM-056', 'AM-057', 'AM-058', 'AM-059', 'AM-060', 'AM-061', 'AM-062', 'AM-063',
     'AM-064', 'AN-002', 'AT-001', 'AT-003', 'AT-004', 'AT-005', 'AT-008', 'AT-010', 'AT-011', 'AT-012', 'AT-013',
     'AT-014', 'AT-016', 'AT-017', 'AT-018', 'AT-019', 'AT-021', 'AT-022', 'AT-023', 'AT-024', 'AT-025', 'AV-005',
     'BA-015', 'BA-020', 'BA-025', 'BA-029', 'BA-032', 'BA-033', 'BA-034', 'BA-035', 'BA-036', 'BA-037', 'BA-038',
     'BA-039', 'BA-040', 'BA-041', 'BA-042', 'BC-001', 'BC-008', 'BC-009', 'BC-011', 'BC-017', 'BC-018', 'BC-019',
     'BC-020', 'BC-021', 'BC-022', 'BC-023', 'BT-018', 'BT-020', 'BT-021', 'BT-022', 'BT-023', 'BT-024', 'BT-026',
     'BT-027', 'BT-028', 'BT-029', 'BT-030', 'BT-033', 'BT-035', 'BT-037', 'BV-009', 'BV-015', 'BV-016', 'BV-017',
     'BV-018', 'BV-019', 'BV-020', 'BV-021', 'BV-022', 'BV-023', 'BV-024', 'BV-025', 'BV-026', 'BV-027', 'BV-028',
     'BV-029', 'BV-030', 'BV-032', 'CE-020', 'CE-021', 'CE-022', 'CE-023', 'CE-024', 'CE-026', 'CH-005', 'CH-006',
     'CH-007', 'CH-008', 'CH-010', 'CH-011', 'CH-012', 'CO-012', 'CO-014', 'CO-015', 'CO-017', 'DA-017', 'DA-018',
     'DA-019', 'DA-020', 'DA-021', 'DC-002', 'DC-003', 'DC-004', 'DC-005', 'DC-006', 'DC-008', 'EC-028', 'EC-030',
     'EC-031', 'EC-032', 'EC-035', 'ED-013', 'ED-017', 'ED-018', 'ED-021', 'ED-023', 'ED-024', 'ED-025', 'ED-026',
     'ED-027', 'ED-028', 'ED-029', 'ED-030', 'ED-032', 'ED-033', 'ED-034', 'ED-035', 'ED-036', 'ED-037', 'ED-038',
     'ED-039', 'ED-040', 'EV-002', 'EV-004', 'EV-005', 'EV-015', 'EV-021', 'EV-023', 'EV-031', 'EV-034', 'EV-035',
     'EV-036', 'EV-043', 'EV-045', 'EV-046', 'EV-047', 'EV-049', 'EV-051', 'EV-053', 'EV-054', 'EV-059', 'EV-060',
     'EV-062', 'EV-063', 'EV-064', 'EV-065', 'EV-066', 'EV-067', 'EV-068', 'EV-069', 'EV-070', 'EV-071', 'EV-072',
     'EV-073', 'EV-076', 'EV-077', 'EV-078', 'EV-079', 'EV-080', 'EV-084', 'EV-086', 'EV-087', 'EV-088', 'EV-089',
     'EV-090', 'EV-091', 'EV-093', 'FA-038', 'FA-053', 'FA-058', 'FA-059', 'FA-061', 'FA-062', 'FA-063', 'FA-064',
     'FA-065', 'FA-067', 'FA-068', 'FA-069', 'FA-070', 'FA-071', 'FA-072', 'FA-073', 'FA-074', 'FA-075', 'FA-076', 
     'FA-077', 'FA-078', 'FA-079', 'FA-080', 'FA-081', 'FA-082', 'FA-083', 'FA-084', 'FA-085', 'FA-086', 'FA-087',
     'FA-088', 'FH-004', 'FH-009', 'FH-022', 'FH-025', 'FH-026', 'FH-027', 'FH-028', 'FH-029', 'FH-030', 'FH-031',
     'FH-032', 'FH-033', 'FH-034', 'FH-035', 'FH-036', 'FH-037', 'FH-039', 'FS-027', 'FS-028', 'FS-029', 'FS-030',
     'FS-031', 'FS-032', 'FS-033', 'HA-009', 'HA-025', 'HA-027', 'HA-028', 'HA-029', 'HC-021', 'HC-022', 'HC-024',
     'HC-025', 'HC-026', 'HC-027', 'HC-028', 'HC-030', 'HC-031', 'HC-032', 'HC-033', 'HC-035', 'HC-036', 'HC-037',
     'HC-038', 'HC-039', 'HC-040', 'HC-041', 'HC-042', 'HC-043', 'HC-044', 'HC-045', 'HC-046', 'HC-047', 'HC-048',
     'HC-049', 'HC-050', 'HC-051', 'HC-052', 'HC-053', 'HC-054', 'HC-055', 'HC-056', 'HC-057', 'HC-058', 'HC-059',
     'HC-060', 'HC-061', 'HC-062', 'HC-064', 'HT-004', 'HT-007', 'HT-008', 'HT-009', 'HT-014', 'HT-015', 'HT-016',
     'HT-017', 'HT-018', 'HT-019', 'HT-020', 'HT-021', 'IN-004', 'OG-007', 'OG-009', 'OG-010', 'OS-007', 'OS-009', 
     'OS-011', 'OS-012', 'OS-013', 'OS-014', 'OS-016', 'OS-017', 'OS-018', 'OS-019', 'OS-020', 'OS-021', 'OS-022', 
     'OS-023', 'OS-024', 'OS-025', 'OS-026', 'OS-027', 'OS-028', 'OS-029', 'OS-030', 'OS-031', 'OS-032', 'OS-033', 
     'RE-022', 'RE-039', 'RE-043', 'RE-049', 'RE-059', 'RE-062', 'RE-063', 'RE-080', 'RE-082', 'RE-092', 'RE-095',
     'RE-102', 'RE-105', 'RE-106', 'RE-107', 'RE-108', 'RE-115', 'RE-117', 'RE-118', 'RE-119', 'RE-121', 'RE-123',
     'RE-128', 'RE-132', 'RE-134', 'RE-135', 'RE-136', 'RE-140', 'RE-141', 'RE-142', 'RE-145', 'RE-146', 'RE-147',
     'RE-148', 'RE-149', 'RE-151', 'RE-152', 'RE-153', 'RE-154', 'RE-155', 'RE-156', 'RE-157', 'RE-158', 'RE-160', 
     'RE-161', 'RE-162', 'RE-163', 'RE-164', 'RE-165', 'RE-166', 'RE-167', 'RE-168', 'RE-169', 'RE-170', 'RE-173',
     'RE-174', 'RE-175', 'RE-176', 'RE-178', 'RE-179', 'RE-180', 'RE-181', 'RE-182', 'RE-183', 'RE-184', 'RE-185',
     'RE-186', 'RE-187', 'RE-188', 'RE-189', 'RE-190', 'RE-191', 'RE-192', 'RE-193', 'RE-194', 'RE-195', 'RE-196',
     'RE-197', 'RE-198', 'RE-199', 'RE-200', 'RE-202', 'RE-203', 'RE-205', 'RF-040', 'RF-041', 'RF-046', 'RF-047',
     'RF-048', 'RF-049', 'RF-050', 'RF-051', 'RF-052', 'RF-053', 'RF-054', 'RF-055', 'RF-056', 'RF-057', 'Re-116', 
     'SC-011', 'SC-012', 'SC-013', 'SC-014', 'SC-015', 'SC-017', 'SC-018', 'SC-019', 'SC-021', 'SC-022', 'SF-018', 
     'SF-019', 'SF-023', 'SF-026', 'SF-030', 'SF-031', 'SF-032', 'SF-033', 'SF-036', 'SF-037', 'SF-038', 'SF-039',
     'SF-040', 'SF-041', 'SF-042', 'SF-043', 'SF-044', 'SF-045', 'SF-046', 'SF-047', 'SF-048', 'SF-049', 'SF-050',
     'SF-052', 'SF-053', 'SF-054', 'SF-055', 'SF-056', 'SF-057', 'SF-058', 'SF-059', 'SF-060', 'SF-061', 'TP-006',
     'TP-007', 'TP-008', 'TP-009', 'background']
    
    resnet34 = models.resnet34(pretrained=True)
    resnet34 = resnet34.to(device)
    resnet34.load_state_dict(torch.load(modelWeightPath))
    resnet34.eval()
    ans=[]
    image=imagePath
    trans=transforms.Compose([transforms.ToTensor(),transforms.Normalize(0.4986, 0.4474, 0.4107)])
    boolLogo,retLogo,boolText,retText=recognize(client,image)
    im=Image.open(image)
    if boolLogo ==True:
        for i in range(len(retLogo)):
            x1,y1= retLogo[i][1].vertices[0].x,retLogo[i][1].vertices[0].y
            x2,y2= retLogo[i][1].vertices[1].x,retLogo[i][1].vertices[1].y
            x3,y3= retLogo[i][1].vertices[2].x,retLogo[i][1].vertices[2].y
            x4,y4= retLogo[i][1].vertices[3].x,retLogo[i][1].vertices[3].y
            top_left_x  = min([x1,x2,x3,x4])
            top_left_y  = min([y1,y2,y3,y4])
            bot_right_x = max([x1,x2,x3,x4])
            bot_right_y = max([y1,y2,y3,y4])
            im1 = im.crop((top_left_x-30, top_left_y-30, bot_right_x+35, bot_right_y+35)).convert('RGB').resize((160,160))
            im1=trans(im1)
            im1=im1.to(device).unsqueeze(0)
            output=resnet34(im1)
            if log[output.argmax(dim=1).item()]!="background":
                landmarks= (top_left_x-30, top_left_y-30, bot_right_x+35, bot_right_y+35)
                ans.append([log[output.argmax(dim=1).item(),landmarks]])
    if boolText==True:
        for i in range(1,len(retText)):
            x1,y1= retText[i][1].vertices[0].x,retText[i][1].vertices[0].y
            x2,y2= retText[i][1].vertices[1].x,retText[i][1].vertices[1].y
            x3,y3= retText[i][1].vertices[2].x,retText[i][1].vertices[2].y
            x4,y4= retText[i][1].vertices[3].x,retText[i][1].vertices[3].y
            top_left_x  = min([x1,x2,x3,x4])
            top_left_y  = min([y1,y2,y3,y4])
            bot_right_x = max([x1,x2,x3,x4])
            bot_right_y = max([y1,y2,y3,y4])
            im1 = im.crop((top_left_x-30, top_left_y-30, bot_right_x+35, bot_right_y+35)).convert('RGB').resize((160,160))
            im1=trans(im1)
            im1=im1.to(device).unsqueeze(0)
            output=resnet34(im1)
            if log[output.argmax(dim=1).item()]!="background":
                landmarks= (top_left_x-30, top_left_y-30, bot_right_x+35, bot_right_y+35)
                ans.append([log[output.argmax(dim=1).item(),landmarks]])
    im=trans(im)
    im=im.to(device).unsqueeze(0)
    output=resnet34(im)
    if log[output.argmax(dim=1).item()]!="background":
        landmarks= (0,0,160,160)
        ans.append([log[output.argmax(dim=1).item(),landmarks]])
    return ans, retLogo