# Importing Libraries

import numpy as np
from customtkinter import *
import cv2
import os, sys
import time
import operator

from string import ascii_uppercase

# import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk

# from hunspell import Hunspell
import enchant

from keras.models import model_from_json

os.environ["THEANO_FLAGS"] = "device=cuda, assert_no_cpu_op=True"

#Application :

class Application:

    def __init__(self,root):

        # self.hs = Hunspell('en_US')
        self.root  = root
        self.d = enchant.Dict("en_US")
        self.vs = cv2.VideoCapture(0)
        self.current_image = None
        self.current_image2 = None
        self.json_file = open("Models\model_new.json", "r")
        self.model_json = self.json_file.read()
        self.json_file.close()

        self.loaded_model = model_from_json(self.model_json)
        self.loaded_model.load_weights("Models\model_new.h5")

        self.json_file_dru = open("Models\model-bw_dru.json" , "r")
        self.model_json_dru = self.json_file_dru.read()
        self.json_file_dru.close()

        self.loaded_model_dru = model_from_json(self.model_json_dru)
        self.loaded_model_dru.load_weights("Models\model-bw_dru.h5")
        self.json_file_tkdi = open("Models\model-bw_tkdi.json" , "r")
        self.model_json_tkdi = self.json_file_tkdi.read()
        self.json_file_tkdi.close()

        self.loaded_model_tkdi = model_from_json(self.model_json_tkdi)
        self.loaded_model_tkdi.load_weights("Models\model-bw_tkdi.h5")
        self.json_file_smn = open("Models\model-bw_smn.json" , "r")
        self.model_json_smn = self.json_file_smn.read()
        self.json_file_smn.close()

        self.loaded_model_smn = model_from_json(self.model_json_smn)
        self.loaded_model_smn.load_weights("Models\model-bw_smn.h5")

        self.ct = {}
        self.ct['blank'] = 0
        self.blank_flag = 0

        for i in ascii_uppercase:
          self.ct[i] = 0
        
        print("Loaded model from disk")

        # self.root = Tk()
        # self.root.title("Sign Language To Text Conversion")
        # self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        # self.root.geometry("900x900")

        self.panel = Label(self.root)
        self.panel.place(x = 520, y = 10, width = 580, height = 580)
        # self.panel.pack(side=TOP )
        
        self.panel2 = Label(self.root) # initialize image panel
        self.panel2.place(x = 820, y = 65, width = 275, height = 275)

        self.T = CTkLabel(self.root,text="Sign Language to Text Conversion" , font=("Inter",30 ,"bold"), bg_color='#9098A3')
        # self.T.place(x = 60, y = 5)
        self.T.pack(side=TOP , fill=X)
        # self.T.config(text = "Sign Language To Text Conversion", font = ("Inter", 30, "bold") , background="white")

        self.panel3 = Label(self.root , font=("Inter" ,30 ,'bold') , background="white") # Current Symbol
        self.panel3.place(x = 310, y = 540)

        self.T1 = Label(self.root ,text="Character :" ,font=("Inter" , 30,"bold") , background='white')
        # self.T1.place(x = 10, y = 540)
        self.T1.place(x=10 , y=540)
        # self.T1.config(text = "Character :", font = ("Inter", 30, "bold"))

        self.panel4 = Label(self.root , bg='white') # Word
        self.panel4.place(x = 220, y = 595)

        self.T2 = Label(self.root)
        self.T2.place(x = 10,y = 595)
        self.T2.config(text = "Word :", font = ("Inter", 30, "bold") , background='white')

        self.panel5 = Label(self.root,background="white") # Sentence
        self.panel5.place(x = 350, y = 645)

        self.T3 = Label(self.root)
        self.T3.place(x = 10, y = 645)
        self.T3.config(text = "Sentence :",font = ("Inter", 30, "bold") ,background='white')

        self.T4 = Label(self.root)
        self.T4.place(x = 10, y = 695)
        self.T4.config(text = "Suggestions :", fg = "red", font = ("Inter", 30, "bold") , background='white')

        self.bt1 = Button(self.root, command = self.action1, height = 0, width = 0)
        self.bt1.place(x = 460, y = 710)

        self.bt2 = Button(self.root, command = self.action2, height = 0, width = 0)
        self.bt2.place(x = 1325, y = 710)

        self.bt3 = Button(self.root, command = self.action3, height = 0, width = 0)
        self.bt3.place(x = 825, y = 710)


        self.str = ""
        self.word = " "
        self.current_symbol = "Empty"
        self.photo = "Empty"
        self.video_loop()


    def video_loop(self):
        ok, frame = self.vs.read()

        if ok:
            cv2image = cv2.flip(frame, 1)

            x1 = int(0.5 * frame.shape[1])
            y1 = 10
            x2 = frame.shape[1] - 10
            y2 = int(0.5 * frame.shape[1])

            cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 0, 0) ,1)
            cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGBA)

            self.current_image = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image = self.current_image)

            self.panel.imgtk = imgtk
            self.panel.config(image = imgtk)

            cv2image = cv2image[y1 : y2, x1 : x2]

            gray = cv2.cvtColor(cv2image, cv2.COLOR_BGR2GRAY)

            blur = cv2.GaussianBlur(gray, (5, 5), 2)

            th3 = cv2.adaptiveThreshold(blur, 255 ,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

            ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            self.predict(res)

            self.current_image2 = Image.fromarray(res)

            imgtk = ImageTk.PhotoImage(image = self.current_image2)

            self.panel2.imgtk = imgtk
            self.panel2.config(image = imgtk)

            self.panel3.config(text = self.current_symbol, font = ("Inter", 30))

            self.panel4.config(text = self.word, font = ("Inter", 30))

            self.panel5.config(text = self.str,font = ("Inter", 30))

            # predicts = self.hs.suggest(self.word)
            # predicts = self.d.suggest(self.word)
            
            # if(len(predicts) > 1):

            #     self.bt1.config(text = predicts[0], font = ("Courier", 20))

            # else:

            #     self.bt1.config(text = "")

            # if(len(predicts) > 2):

            #     self.bt2.config(text = predicts[1], font = ("Courier", 20))

            # else:

            #     self.bt2.config(text = "")

            # if(len(predicts) > 3):

            #     self.bt3.config(text = predicts[2], font = ("Courier", 20))

            # else:

            #     self.bt3.config(text = "")

            if self.word:  # Check if self.word is not an empty string
                predicts = self.d.suggest(self.word)
            else:
                predicts = []  # Set predicts to an empty list if self.word is empty

            if len(predicts) > 1:
                self.bt1.config(text=predicts[0], font=("Courier", 20))
            else:
                self.bt1.config(text="")

            if len(predicts) > 2:
                self.bt2.config(text=predicts[1], font=("Courier", 20))
            else:
                self.bt2.config(text="")

            if len(predicts) > 3:
                self.bt3.config(text=predicts[2], font=("Courier", 20))
            else:
                self.bt3.config(text="")



        self.root.after(5, self.video_loop)

    def predict(self, test_image):

        test_image = cv2.resize(test_image, (128, 128))

        result = self.loaded_model.predict(test_image.reshape(1, 128, 128, 1))


        result_dru = self.loaded_model_dru.predict(test_image.reshape(1 , 128 , 128 , 1))

        result_tkdi = self.loaded_model_tkdi.predict(test_image.reshape(1 , 128 , 128 , 1))

        result_smn = self.loaded_model_smn.predict(test_image.reshape(1 , 128 , 128 , 1))

        prediction = {}

        prediction['blank'] = result[0][0]

        inde = 1

        for i in ascii_uppercase:

            prediction[i] = result[0][inde]

            inde += 1

        #LAYER 1

        prediction = sorted(prediction.items(), key = operator.itemgetter(1), reverse = True)

        self.current_symbol = prediction[0][0]


        #LAYER 2

        if(self.current_symbol == 'D' or self.current_symbol == 'R' or self.current_symbol == 'U'):

            prediction = {}
            prediction['D'] = result_dru[0][0]
            prediction['R'] = result_dru[0][1]
            prediction['U'] = result_dru[0][2]
            prediction = sorted(prediction.items(), key = operator.itemgetter(1), reverse = True)

            self.current_symbol = prediction[0][0]

        if(self.current_symbol == 'D' or self.current_symbol == 'I' or self.current_symbol == 'K' or self.current_symbol == 'T'):
            prediction = {}
            prediction['D'] = result_tkdi[0][0]
            prediction['I'] = result_tkdi[0][1]
            prediction['K'] = result_tkdi[0][2]
            prediction['T'] = result_tkdi[0][3]

            prediction = sorted(prediction.items(), key = operator.itemgetter(1), reverse = True)

            self.current_symbol = prediction[0][0]

        if(self.current_symbol == 'M' or self.current_symbol == 'N' or self.current_symbol == 'S'):
            prediction1 = {}
            prediction1['M'] = result_smn[0][0]
            prediction1['N'] = result_smn[0][1]
            prediction1['S'] = result_smn[0][2]

            prediction1 = sorted(prediction1.items(), key = operator.itemgetter(1), reverse = True)
            if(prediction1[0][0] == 'S'):
                self.current_symbol = prediction1[0][0]
            else:
                self.current_symbol = prediction[0][0]

            # prediction1[0][0] == 'S' ? self.current_symbol = prediction1[0][0] : self.current_symbol = prediction[0][0]
            # self.current_symbol = prediction1[0][0] if (prediction1[0][0] == 'S') else self.current_symbol = prediction[0][0]
            
            
            
        
        if(self.current_symbol == 'blank'):

            for i in ascii_uppercase:
                self.ct[i] = 0

        self.ct[self.current_symbol] += 1

        if(self.ct[self.current_symbol] > 60):

            for i in ascii_uppercase:
                if i == self.current_symbol:
                    continue

                tmp = self.ct[self.current_symbol] - self.ct[i]

                if tmp < 0:
                    tmp *= -1

                if tmp <= 20:
                    self.ct['blank'] = 0

                    for i in ascii_uppercase:
                        self.ct[i] = 0
                    return

            self.ct['blank'] = 0

            for i in ascii_uppercase:
                self.ct[i] = 0

            if self.current_symbol == 'blank':

                if self.blank_flag == 0:
                    self.blank_flag = 1

                    if len(self.str) > 0:
                        self.str += " "

                    self.str += self.word

                    self.word = ""

            else:

                if(len(self.str) > 16):
                    self.str = ""

                self.blank_flag = 0

                self.word += self.current_symbol

    def action1(self):
        # predicts = self.hs.suggest(self.word)
        # predicts = self.d.suggest(self.word)

        # if(len(predicts) > 0):
        #     self.word = ""
        #     self.str += " "
        #     self.str += predicts[0]

        if self.word:
            predicts = self.d.suggest(self.word)
            if len(predicts) > 0:
                self.word = ""
                self.str += " "
                self.str += predicts[0]

    def action2(self):
        # predicts = self.hs.suggest(self.word)
        # predicts = self.d.suggest(self.word)
        # if(len(predicts) > 1):
        #     self.word = ""
        #     self.str += " "
        #     self.str += predicts[17 ]
        if self.word:
            predicts = self.d.suggest(self.word)
            if len(predicts) > 1:
                self.word = ""
                self.str += " "
                self.str += predicts[1]

    def action3(self):
        # predicts = self.hs.suggest(self.word)
        # predicts = self.d.suggest(self.word)
        # if(len(predicts) > 2):
        #     self.word = ""
        #     self.str += " "
        #     self.str += predicts[2]
        if self.word:
            predicts = self.d.suggest(self.word)
            if len(predicts) > 2:
                self.word = ""
                self.str += " "
                self.str += predicts[2]

    def action4(self):
        # predicts = self.hs.suggest(self.word)
        # predicts = self.d.suggest(self.word)
        # if(len(predicts) > 3):
        #     self.word = ""
        #     self.str += " "
        #     self.str += predicts[3]
        if self.word:
            predicts = self.d.suggest(self.word)
            if len(predicts) > 3:
                self.word = ""
                self.str += " "
                self.str += predicts[3]

    def action5(self):
        # predicts = self.hs.suggest(self.word)
        # predicts = self.d.suggest(self.word)
        # if(len(predicts) > 4):
        #     self.word = ""
        #     self.str += " "
        #     self.str += predicts[4]
        if self.word:
            predicts = self.d.suggest(self.word)
            if len(predicts) > 4:
                self.word = ""
                self.str += " "
                self.str += predicts[4]
            
    def destructor(self):

        print("Closing Application...")

        self.root.destroy()
        self.vs.release()
        cv2.destroyAllWindows()
    
print("Starting Application...")

# if __name__ == "__main__":
#    Application().root.mainloop()


# kamlesh code

class DashBoard:
     def __init__(self , root):
          self.root  = root  

          lbl_title = CTkLabel(self.root, text="DashBoard",  font=("Inter", 30, "bold") ,bg_color='#9098A3')
          lbl_title.pack(side=TOP, fill=X)


          # description = Text(self.root , wrap=WORD , font=("Inter",12 ) , fg='black')
          # description.insert(
          #     END , 
          #     "Sign languages (also known as signed languages) are languages that use the visual-manual modality to convey meaning, instead of spoken words. Sign languages are expressed through manual articulation in combination with non-manual markers. Sign languages are full-fledged natural languages with their own grammar and lexicon.[1] Sign languages are not universal and are usually not mutually intelligible,[2] although there are also similarities among different sign languages."
          # )
          # description.config(state='disabled')
          # description.pack(padx=10 ,pady=10 , fill=X)

          description = Label(self.root, text="Sign languages (also known as signed languages) are languages that use the visual-manual modality to convey meaning, instead of spoken words. Sign languages are expressed through manual articulation in combination with non-manual markers. Sign languages are full-fledged natural languages with their own grammar and lexicon. Sign languages are not universal and are usually not mutually intelligible,although there are also similarities among different sign languages.", font=("Inter", 15, "bold"),background='white' , wraplength=1600 ,anchor='w', justify='left')
          description.place(x=23,y=50)



          history_text = Label(self.root,text="History", anchor='w',justify='left' ,font=("Inter",30,'bold'),background='white')
          history_text.place(x=23,y=180)


          description = Label(self.root, text="Groups of deaf people have used sign languages throughout history. One of the earliest written records of a sign language is from the fifth century B.C., in Plato's Cratylus, where Socrates says: 'If we hadn't a voice or a tongue, and wanted to express things to one another, wouldn't we try to make signs by moving our hands, head, and the rest of our body, just as dumb people do at present' Until the 19th century, most of what is known about historical sign languages is limited to the manual alphabets (fingerspelling systems) that were invented to facilitate the transfer of words from a spoken language to a sign language, rather than documentation of the language itself. Debate around the monastic sign-language developed in the Middle Ages has come to regard it as a gestural system rather than a true sign language", font=("Inter", 15, "bold"),background='white' , wraplength=1600 ,anchor='w', justify='left')
          description.place(x=23,y=250)


          btn_dash_learn = Button(self.root,text="Learn 🎯", font=("Inter" ,20 ,"bold") ,command=self.ASLalphabetviewer ,compound=LEFT , padx=5 , pady=4 ,anchor="w", bg='white' , cursor='hand2',borderwidth=0 ,background='#CCDFDE').place(x=23 , y=480)

          btn_dash_Practice = Button(self.root , text="Practice 🚀", font=("Inter" ,20 ,"bold") ,command=self.PracticeSection ,compound=LEFT , padx=5 , pady=4 ,anchor="w", bg='white' , cursor='hand2' ,borderwidth=0, background='#CCDFDE').place(x=190,y=480)

     def ASLalphabetviewer(self) :
         frames.tkraise()

    # def videos_section(self):
    #      frames_videos.tkraise()

     def dashboard_section(self):
         frames_dashboard.tkraise()   

     def PracticeSection(self):
         frames_practice.tkraise() 





          

class ASLAlphabetViewer:
    def __init__(self, master, alphabet_images):
        self.master = master
        self.alphabet_images = alphabet_images
        self.current_alphabet_index = 0

        self.header_label = CTkLabel(master, text="Learning Section", font=("Inter", 25 ,'bold') ,corner_radius=200 , width=90 , height=20 )
        self.header_label.pack(pady=9 ,padx=9 )

        
        self.canvas = Canvas(master, width=500, height=500 , background='white')
        self.canvas.pack(side="top" , pady=30)
        

        self.alphabet_label = Label(master, text="Alphabet: ", font=("Inter", 25,"bold") , background="white")
        self.alphabet_label.pack()
        
        self.show_alphabet_image()

        self.prev_button = CTkButton(master, text="Previous", command=self.show_prev_alphabet, width=60, height=30)
        self.prev_button.pack(side=LEFT, padx=(400, 20))  # Added padx to provide left and right padding

        self.next_button = CTkButton(master, text="Next", command=self.show_next_alphabet, width=60, height=30)
        self.next_button.pack(side=RIGHT, padx=(20, 400))  # Added padx to provide left and right padding
        
        # self.master.bind("<Left>",self.show_prev_alphabet)
        # self.master.bind("<Right>",self.show_next_alphabet)

    def show_alphabet_image(self):
        alphabet_image_path = self.alphabet_images[self.current_alphabet_index]
        alphabet_image = Image.open(alphabet_image_path)
        alphabet_image = alphabet_image.resize((400, 400))
        alphabet_photo = ImageTk.PhotoImage(alphabet_image)
        self.canvas.create_image(50, 50, anchor=NW, image=alphabet_photo)
        self.canvas.image = alphabet_photo
        alphabet_name = alphabet_image_path.split('\\')[-2]
        self.alphabet_label.config(text=f"Alphabet: {alphabet_name.upper()}")

    def show_prev_alphabet(self):
        self.current_alphabet_index = (self.current_alphabet_index - 1) % len(self.alphabet_images)
        self.show_alphabet_image()

    def show_next_alphabet(self):
        self.current_alphabet_index = (self.current_alphabet_index + 1) % len(self.alphabet_images)
        self.show_alphabet_image()



class SignLanguageClass: 
    def __init__(self , root):
          self.root = root 
          self.root.geometry("1520x790+0+0")
          self.root.title('Sign Language Learning & Detection Desktop Application')
        #   self.root.config(bg='white')
          back = set_appearance_mode('light')

        # Title =====
          self.icon_title=Image.open('./images/logo.png')
          self.icon_title=self.icon_title.resize((50 , 50),Image.LANCZOS)
          self.icon_title=ImageTk.PhotoImage(self.icon_title)

          title = CTkLabel(self.root,text="Sign Language Learning Companion",image=self.icon_title , compound=RIGHT ,font=('Inter' ,40 ,'bold') ,height=70  ,padx =20 , pady = 27).place(x=0,y=0)


          #    left menu
          self.menuLogo=Image.open('./images/signlogo.jpg')
          self.menuLogo=self.menuLogo.resize((200, 200),Image.LANCZOS)
          self.menuLogo=ImageTk.PhotoImage(self.menuLogo)
 
          LeftMenu = CTkFrame(self.root , width=209 , height=680 , border_color='#E4E9F1' , border_width=2)
          LeftMenu.place(x=14 , y=90) 

          label_menulogo=Label(LeftMenu , image=self.menuLogo)
          label_menulogo.pack(side=TOP , fill=X)



        #   right arrow image
          self.right_arrow = PhotoImage(file='./images/rightarrow.png')

          lbl_menu=Label(LeftMenu , text='Menu' , font=("Inter" , 20 ,"bold"),bg="#009688" ,borderwidth=1).pack(side=TOP , fill=X)


          btn_dashboard = Button(LeftMenu , text="Dashboard", font=("Inter" ,20 ,"bold") ,command=self.dashboard_section,   compound=LEFT , padx=5 , pady=4 ,anchor="w", bg='white' , cursor='hand2',borderwidth=0).pack(side=TOP , fill=X)

          btn_learn = Button(LeftMenu , text="Learn 🎯", font=("Inter" ,20 ,"bold") , command=self.ASLalphabetviewer ,compound=LEFT , padx=5 , pady=4 ,anchor="w", bg='white' , cursor='hand2',borderwidth=0 ).pack(side=TOP , fill=X)

          btn_Practice = Button(LeftMenu , text="Practice 🚀", font=("Inter" ,20 ,"bold") ,command=self.PracticeSection ,compound=LEFT , padx=5 , pady=4 ,anchor="w", bg='white' , cursor='hand2' ,borderwidth=0).pack(side=TOP , fill=X)

          # btn_Videos = Button(LeftMenu , text="Videos 📹", font=("Inter" ,20 ,"bold") , command=self.videos_section, compound=LEFT , padx=5 , pady=4 ,anchor="w", bg='white' , cursor='hand2' ,borderwidth=0).pack(side=TOP , fill=X)
    

          frames_dashboard.tkraise()

          


          
    def ASLalphabetviewer(self) :
         frames.tkraise()

    # def videos_section(self):
    #      frames_videos.tkraise()

    def dashboard_section(self):
         frames_dashboard.tkraise()   

    def PracticeSection(self):
         frames_practice.tkraise()   


if __name__ == '__main__':
      

      # List of image paths for ASL alphabets
      alphabet_images = [
           r"C:\Users\Hanesh\Downloads\Sign-Language-To-Text-Conversion-main\dataSet\testingData\A\3.jpg",
           r"C:\Users\Hanesh\Downloads\Sign-Language-To-Text-Conversion-main\dataSet\testingData\B\3.jpg",
           r"C:\Users\Hanesh\Downloads\Sign-Language-To-Text-Conversion-main\dataSet\testingData\C\3.jpg",
           r"C:\Users\Hanesh\Downloads\Sign-Language-To-Text-Conversion-main\dataSet\testingData\D\3.jpg",
           r"C:\Users\Hanesh\Downloads\Sign-Language-To-Text-Conversion-main\dataSet\testingData\E\3.jpg",
           r"C:\Users\Hanesh\Downloads\Sign-Language-To-Text-Conversion-main\dataSet\testingData\F\3.jpg",
           r"C:\Users\Hanesh\Downloads\Sign-Language-To-Text-Conversion-main\dataSet\testingData\G\3.jpg",
           r"C:\Users\Hanesh\Downloads\Sign-Language-To-Text-Conversion-main\dataSet\testingData\H\3.jpg",
           r"C:\Users\Hanesh\Downloads\Sign-Language-To-Text-Conversion-main\dataSet\testingData\I\3.jpg",
           r"C:\Users\Hanesh\Downloads\Sign-Language-To-Text-Conversion-main\dataSet\testingData\J\3.jpg",
           r"C:\Users\Hanesh\Downloads\Sign-Language-To-Text-Conversion-main\dataSet\testingData\K\3.jpg",
           r"C:\Users\Hanesh\Downloads\Sign-Language-To-Text-Conversion-main\dataSet\testingData\L\3.jpg",
           r"C:\Users\Hanesh\Downloads\Sign-Language-To-Text-Conversion-main\dataSet\testingData\M\3.jpg",
           r"C:\Users\Hanesh\Downloads\Sign-Language-To-Text-Conversion-main\dataSet\testingData\N\3.jpg",
           r"C:\Users\Hanesh\Downloads\Sign-Language-To-Text-Conversion-main\dataSet\testingData\O\3.jpg",
           r"C:\Users\Hanesh\Downloads\Sign-Language-To-Text-Conversion-main\dataSet\testingData\P\3.jpg",
           r"C:\Users\Hanesh\Downloads\Sign-Language-To-Text-Conversion-main\dataSet\testingData\Q\3.jpg",
           r"C:\Users\Hanesh\Downloads\Sign-Language-To-Text-Conversion-main\dataSet\testingData\R\3.jpg",
           r"C:\Users\Hanesh\Downloads\Sign-Language-To-Text-Conversion-main\dataSet\testingData\S\3.jpg",
           r"C:\Users\Hanesh\Downloads\Sign-Language-To-Text-Conversion-main\dataSet\testingData\T\3.jpg",
           r"C:\Users\Hanesh\Downloads\Sign-Language-To-Text-Conversion-main\dataSet\testingData\U\3.jpg",
           r"C:\Users\Hanesh\Downloads\Sign-Language-To-Text-Conversion-main\dataSet\testingData\V\3.jpg",
           r"C:\Users\Hanesh\Downloads\Sign-Language-To-Text-Conversion-main\dataSet\testingData\W\3.jpg",
           r"C:\Users\Hanesh\Downloads\Sign-Language-To-Text-Conversion-main\dataSet\testingData\X\3.jpg",
           r"C:\Users\Hanesh\Downloads\Sign-Language-To-Text-Conversion-main\dataSet\testingData\Y\3.jpg",
           r"C:\Users\Hanesh\Downloads\Sign-Language-To-Text-Conversion-main\dataSet\testingData\Z\3.jpg",
      ]
      root = CTk()

      frames = Frame(root,relief=RIDGE, bg="White")
      frames.place(x=250, y=118, width=1630, height=810)
      learn_section = ASLAlphabetViewer(frames, alphabet_images)
    

      frames_dashboard = Frame(root , relief=RIDGE , bg='white')
      frames_dashboard.place(x=250 , y=118 , width=1630 , height=810) 
      dashboard = DashBoard(frames_dashboard)


      video_urls = [
       "https://www.youtube.com/embed/ScMzIvxBSi4",
    "https://www.youtube.com/embed/kJQP7kiw5Fk",
    "https://www.youtube.com/embed/8Pb97pK1HFA",
    ]

    #   frames_videos = Frame(root, relief=RIDGE , bg='white')
    #   frames_videos.place(x=250 , y=118 , width=1630 ,  height=810)
    #   video = YouTube(frames_videos  , video_urls)

      frames_practice = Frame(root, relief=RIDGE , bg='white')
      frames_practice.place(x=250 , y=118 , width=1630 ,  height=810)
      practice = Application(frames_practice)
      
      obj = SignLanguageClass(root)
      root.mainloop()