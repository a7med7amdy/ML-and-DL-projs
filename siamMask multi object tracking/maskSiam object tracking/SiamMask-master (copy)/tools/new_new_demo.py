import glob
import sys
#sys.path.insert(1, '/home/ahmed/intern work/maskSiam object tracking/SiamMask-master (copy)/tools')
sys.path.insert(1, '../tools')
from test import *
import time
import numpy as np
import torch.multiprocessing as mp
import queue as Queue
from multiprocessing import Pool
import itertools
from multiprocessing.dummy import Pool as ThreadPool 
import os
import pandas as pd

execution_path = os.getcwd()


parser = argparse.ArgumentParser(description='PyTorch Tracking Demo')

parser.add_argument('--resume', default='', type=str, required=True,
                    metavar='PATH',help='path to latest checkpoint (default: none)')
parser.add_argument('--config', dest='config', default='config_davis.json',
                    help='hyper-parameter of SiamMask in json format')
parser.add_argument('--base_path', default='../../data/tennis', help='datasets')
parser.add_argument('--cpu', action='store_true', help='cpu mode')
args = parser.parse_args()




if __name__ == '__main__':
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    # Setup Model
    cfg = load_config(args)
    #sys.path.insert(1, '/home/ahmed/intern work/maskSiam object tracking/SiamMask-master (copy)/experiments/siammask_sharp')
    sys.path.insert(1, '../../experiments/siammask_sharp')
    from custom import Custom
    siammask = Custom(anchors=cfg['anchors'])
    if args.resume:
        assert isfile(args.resume), 'Please download {} first.'.format(args.resume)
        siammask = load_pretrain(siammask, args.resume)

    siammask.eval().to(device)

    # Parse Image fil

    csvfilepath= input("please enter a path to the csv file used \n")
    df= pd.read_csv(str(csvfilepath))
    ROIs = [[]] * (int(df["name"][len(df)-1].split(".")[0].split("_")[1].strip("0"))+1)
    ROIs = [[] for _ in ROIs]
    for index,j in df.iterrows():
        if j["name"].split("_")[1].strip("0").split(".")[0]=="":
            x1=int(j["x1"])
            y1=int(j["y1"])
            w=int(j["x2"])-x1
            h=int(j["y2"])-y1
            ROIs[0].append([x1,y1,w,h])
        else:
            x1=int(j["x1"])
            y1=int(j["y1"])
            w=int(j["x2"])-x1
            h=int(j["y2"])-y1
            ROIs[int(j["name"].split("_")[1].strip("0").split(".")[0])].append([x1,y1,w,h])
    

    cap = cv2.VideoCapture(args.base_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter('/home/ahmed/intern work/maskSiam object tracking/SiamMask-master (copy)/experiments/siammask_sharp/output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

    if(cap.isOpened() == False):
        print("Unable to open")
        exit()
    ret, frame = cap.read()
    
    # Select ROI
    #cv2.namedWindow("SiamMask", cv2.WND_PROP_FULLSCREEN)
    #try:
     #   ROIs = cv2.selectROIs('SiamMask', frame, False, False)
    #except:
     #   print("exit")
      #  exit()
    targets = []
    ##for i in  ROIs:
      ##  x,y,w,h = i
    f = 0
    toc = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        print(int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if ret == True:
            tic = cv2.getTickCount()
            count =0
            ff=False
            currentframe=int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if currentframe<=2:
            	currentframe=0
            for j in range(len(ROIs[currentframe])):
                ff=True
                x,y,w,h= ROIs[currentframe][j]
                target_pos = np.array([x  + w / 2, y + h / 2])
                target_sz = np.array([w, h])
                s ={"target_pos":target_pos,"target_sz":target_sz,"x":x,"y":y,"w":w,"h":h}
                targets.append(s)
            if f==0:
            	state = siamese_init(frame, siammask, cfg['hp'], device=device,targets=targets)  # init tracker
            if ff==True:
            	ff=False
            	state = siamese_init(frame, siammask, cfg['hp'], device=device,targets=targets)  # init tracker



            state = siamese_track(state, frame)

            for i,target in enumerate(state["targets"]):

                location = target['ploygon'].flatten()
                mask = target['mask'] > state['p'].seg_thr
                masks = (mask > 0) * 255     
                masks = masks.astype(np.uint8)
                frame[:, :, 2] = (mask > 0) * 255 + (mask == 0) * frame[:, :, 2]
            # frame[:, :, 2] = (mask1 > 0) * 255 + (mask1 == 0) * frame[:, :, 2]
                cv2.polylines(frame, [np.int0(location).reshape((-1, 1, 2))], True, (0, 255, 0), 3)

            cv2.imshow('SiamMask', frame)

            key = cv2.waitKey(1)
                # print(states)
            if key > 0:
                break
            f= f+1        
            toc += cv2.getTickCount() - tic
            toc /= cv2.getTickFrequency()
            fps = f / toc


            # Break the loop
        else:
            break

            #   out.release()
            # When everything done, release the video capture object
            cap.release()

            # Closes all the frames
            cv2.destroyAllWindows()
