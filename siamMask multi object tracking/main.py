import os


from tkinter import *
def alert_popup(title, message):
    """Generate a pop-up window for special messages."""
    root = Tk()
    root.title(title)
    w = 900     # popup window width
    h = 300     # popup window height
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    x = (sw - w)/2
    y = (sh - h)/2
    root.geometry('%dx%d+%d+%d' % (w, h, x, y))
    m = message
    m += '\n'
    w = Label(root, text=m, width=120, height=10)
    w.pack()
    b = Button(root, text="OK", command=root.destroy, width=10)
    b.pack()
    mainloop()
# Examples



print("hi, this is multiple object tracking siamMask")
path = os.getcwd()
vid=str(input("please enter the path to the video you wanna track objects in'\n"))
print("---------------------------------------------------------------------")
print("")

choose= int(input("please enter a number (1) if you want the interactive app or (2) if you want the automatic app \n"))

if choose==1:
	mess="""1- if you want to make the video looks better and to choose the bbox perfectly, enlarge the image please.
		2- use your mouse to draw bbox around object to be tracked.
		3- press enter/space button if you want to add another bbox.
		4- press esc button if you want to start (tracking).
		5- at any time you wanna get out of the video, press any key"""

	alert_popup("important notes", mess)

	new_path="/maskSiam object tracking/SiamMask-master (copy)/experiments/siammask_sharp"
	os.chdir(path+new_path)
	os.system("python ../../tools/new-demo.py --resume SiamMask_DAVIS.pth --config config_davis.json --base_path '" +vid+"'")
else:
	new_path="/maskSiam object tracking/SiamMask-master (copy)/experiments/siammask_sharp"
	os.chdir(path+new_path)
	os.system("python ../../tools/new_new_demo.py --resume SiamMask_DAVIS.pth --config config_davis.json --base_path '" +vid+"'")