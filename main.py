# -*- coding: utf-8 -*-
"""
Created on Fri Aug 08 16:11:33 2021

@author: rizvee
"""
from patterns import pattern_detect
from collections import defaultdict
from PIL import Image, ImageDraw
from math import sqrt, pi, cos, sin
# ----------------------------------------------Input Image--------------------------------------------------#
input_image = Image.open("sphere.jpg")
output_image = Image.new("RGB", input_image.size)
output_image.paste(input_image)
draw_result = ImageDraw.Draw(output_image)
# ----------------------------------------------Define Range--------------------------------------------------#
steps = 100
rmin = 40
rmax = 80
threshold = 0.4
points = []
for r in range(rmin, rmax + 1):
    for t in range(steps):
        points.append((r, int(r * cos(2 * pi * t / steps)), int(r * sin(2 * pi * t / steps))))
acc = defaultdict(int)
for x, y in pattern_detect(input_image):
    for r, dx, dy in points:
        a = x - dx
        b = y - dy
        acc[(a, b, r)] += 1
circles_count = []
for k, v in sorted(acc.items(), key=lambda i: -i[1]):
    x, y, r = k
    if v / steps >= threshold and all((x - xc) ** 2 + (y - yc) ** 2 > rc ** 2 for xc, yc, rc in circles_count):
        print(v / steps, x, y, r)
        circles_count.append((x, y, r))
for x, y, r in circles_count:
    draw_result.ellipse((x-r, y-r, x+r, y+r), outline=(255,0,0,0))

output_image.save("Final_Final_Final_Result.png")