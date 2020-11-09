# -*- coding: utf-8 -*-

import numpy as np
from flask import Flask, redirect, url_for, render_template, request, session,jsonify
import json
import sys
import os


app = Flask(__name__)


@app.route('/calc_freq',methods=['POST','GET'])
def calc_freq():
    if request.method == 'POST':
        content=request.get_json()
        paramX1=content["paramX1"]
        paramY1=content["paramY1"]
        unique_vals=np.unique(paramX1)
        freq=[]
        sum=0
        if(paramY1!=[]):  #y is x/y not the category
            argument1=np.array(paramX1)
            argument2=np.array(paramY1)
            sum=np.sum(argument2)
            for i in unique_vals :
                freq.append(np.sum(argument2[np.where(argument1==i)]))
        else:
            argument1=np.array(paramX1)
            sum=len(argument1)
            for i in unique_vals :  #each category repeated how much
                freq.append(len((np.where(argument1==i))))
        freq=np.array(freq)
        return json.dumps({"unique":unique_vals.tolist(),"freq":freq.tolist(),"sum":int(sum)})
    return render_template('index1.html')





#if I want to filter by date and time



def date_time_match(commonIndices, paramIndices):
   lst= np.intersect1d(commonIndices,paramIndices)
   return lst

def date_time_common_indices(dateNeededIndices, timeNeededIndices ,timeNeeded, dateNeeded):
    out=[]
    for i in range(len(dateNeededIndices)):
        if(dateNeededIndices[i]==dateNeeded):
            if(timeNeededIndices[i]==timeNeeded):
                out.insert(len(out),i)
    return out

@app.route('/calc_freq_date_time',methods=['POST','GET'])

def calc_freq_date_time():
    if request.method == 'POST':
        content=request.get_json()
        paramX1=content["paramX1"]
        paramY1=content["paramY1"]
        date=content["date"]
        time=content["time"]
        dateNeeded=content["dateNeeded"]
        timeNeeded=content["timeNeeded"]
        unique_vals=np.unique(paramX1)
        freq=[]
        sum=0
        commonIndices= date_time_common_indices(date,time,timeNeeded=timeNeeded,dateNeeded=dateNeeded)
        if(paramY1!=[]):  #y is x/y not the category
            argument1=np.array(paramX1)
            argument2=np.array(paramY1)
            sum=np.sum(argument2[commonIndices])   #need to get common and sum them or sum them with the under me now
            for i in unique_vals :
                dataTimeMatchIndices= date_time_match(commonIndices, np.where(argument1==i))
                if(len(dataTimeMatchIndices)>0):
                    freq.append(np.sum(argument2[dataTimeMatchIndices]))
                else :
                    freq.append(0)
        else:
            argument1=np.array(paramX1)
            sum=len(date_time_common_indices(date,time, timeNeeded=timeNeeded, dateNeeded=dateNeeded))
            for i in unique_vals :  #each category repeated how many
                dataTimeMatchIndices= date_time_match(commonIndices, np.where(argument1==i))
                if(len(dataTimeMatchIndices)>0):
                    freq.append(len(dataTimeMatchIndices))
                else:
                    freq.append(0)
        freq=np.array(freq)
        return json.dumps({"unique":unique_vals.tolist(),"freq":freq.tolist(),"sum":int(sum)})
    return render_template('index1.html')




@app.route('/calc_freq_for_line',methods=['POST','GET'])

def calc_freq_for_line ():
     if request.method == 'POST':
        content=request.get_json()

        paramX1=content["paramX1"]
        paramY1=content["paramY1"]
       
        unique_vals=np.unique(paramX1)
        freq=[]
        argument1=np.array(paramX1)
        argument2=np.array(paramY1)
        for i in unique_vals :
            freq.append(argument2[np.where(argument1==i)].tolist())
        freq=np.array(freq)
        return json.dumps({"unique":unique_vals.tolist(),"freq":freq.tolist()})
     else:
        return render_template('index1.html')






# ======== Main ============================================================== #
if __name__ == "__main__":
    app.secret_key = os.urandom(12)  # Generic key for dev purposes only
    app.run(host='0.0.0.0', port=8800)
