import xml.etree.ElementTree as ET 
import os
from os import listdir
from os.path import isfile, join

def parseXML(width, height, xmlfile): 
  
    # create element tree object 
    tree = ET.parse(xmlfile) 
    root = tree.getroot() 
    newsitems = [] 
  
    for item in root.findall('size'): 
        width.append(int(item.find('width').text))
        height.append(int(item.find('height').text))

path = './BV_devkit/data/Annotations/'
files = [f for f in listdir(path) if isfile(join(path, f))]
width = list()
height = list()
for file in files:
    newsitems = parseXML(width, height, path+file)
width.sort()
height.sort()
print('width min:{}, width max:{}'.format(width[0], width[-1]))
print('height min:{}, height max:{}'.format(height[0], height[-1]))
