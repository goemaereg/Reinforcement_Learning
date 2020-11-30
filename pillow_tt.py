from PIL import Image
import numpy as np

background = Image.open("img/parking_overlay.png")
width, height = background.size
true_width, true_height = 23,29
pw = width/true_width # pixel width
ph = height/true_height # pixel height
print(pw, ph)

tractor = Image.open("img/tractor_sprite.png")
haystack = Image.open("img/haystack.png")

s = (15, 15)
pose = 2
angle_per_pose = {0:0, 1:180+90, 2:90, 3:180}
tractor = tractor.rotate(angle_per_pose[pose])
print(angle_per_pose[pose])
tractor.thumbnail((pw*2,ph*2), Image.ANTIALIAS)
tractor_position = (int(15*pw), int(15*ph))

haystack.thumbnail((pw*2,ph*2), Image.ANTIALIAS)
haystack_position = (int(19*pw), int(3*ph))

print(tractor_position, haystack_position)

background.paste(tractor, tractor_position , tractor)
background.paste(haystack, haystack_position, haystack)
filename = "img/overlay_all.png"
background.save(filename,"PNG")
print("Saved file as {}".format(filename))
