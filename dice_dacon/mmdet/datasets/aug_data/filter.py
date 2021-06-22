import cv2
from .filters import resize_filters,apply_fog,WarmingFilter,CoolingFilter,apply_night,increase_brightness,apply_green
from random import randint
#bright 조절
import os
class Appear_aug:
    def __init__(self):
        self.wF=WarmingFilter()
        self.cF=CoolingFilter()

    def aug_data(self,img):
        img=self.set_tone(img)
        img=self.set_fog(img)
        img=self.set_blur(img)
        img=self.set_bright(img)
        return img
    def set_tone(self,img):
        i =randint(1,100)
        if i <=65:
            return img
        elif i <=80 and i>65:
            return self.wF.render(img)
        elif i <=95 and i>80:
            return self.cF.render(img)
        else:
            return apply_green(img)
    def set_fog(self,img):
        i = randint(1,25)
        if i>6:
            return img
        else:
            fog_name= 'fog/'+str(i)+'.png'
            fog=cv2.imread(fog_name)
            img=apply_fog(img,fog)
            return img
    def set_blur(self,img):
        i = randint(0,20)
        if i>6:
            return img
        else:
            return cv2.GaussianBlur(img,(5,5),i)
    def set_bright(self,img):
        i = randint(-30,100)
        if i>50:
            return img
        else:
            return increase_brightness(img,i)
