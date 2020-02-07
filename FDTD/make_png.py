from PIL import Image, ImageDraw
import numpy as np

# 1 pixel = 0.25 mm

img = Image.new('L', (20000, 6000))
draw = ImageDraw.Draw(img)
draw.line((6000, 0, 6000, 2640), fill=255, width=40)
draw.line((10000, 6000, 10000, 3359), fill=255, width=40)
draw.line((14000, 0, 14000, 2640), fill=255, width=40)

img.save('18000_6000_0.25mm.png')

img = Image.new('L', (2000, 600))
draw = ImageDraw.Draw(img)
draw.line((600, 0, 600, 264), fill=255, width=4)
draw.line((1000, 600, 1000, 336), fill=255, width=4)
draw.line((1400, 0, 1400, 264), fill=255, width=4)

img.save('2000_600_600.png')

img = Image.new('L', (2000, 600))
draw = ImageDraw.Draw(img)
# draw.line((600, 0, 600, 264), fill=255, width=4)
draw.line((1000, 600, 1000, 336), fill=255, width=4)
draw.line((1400, 0, 1400, 264), fill=255, width=4)

img.save('2000_600_1000.png')

img = Image.new('L', (2000, 600))
draw = ImageDraw.Draw(img)
#draw.line((600, 0, 600, 264), fill=255, width=4)
#draw.line((1000, 600, 1000, 336), fill=255, width=4)
draw.line((1400, 0, 1400, 264), fill=255, width=4)

img.save('2000_600_1400.png')

img = Image.new('L', (2000, 600))
draw = ImageDraw.Draw(img)
#draw.line((600, 0, 600, 264), fill=255, width=4)
#draw.line((1000, 600, 1000, 336), fill=255, width=4)
#draw.line((1400, 0, 1400, 264), fill=255, width=4)

img.save('2000_600_nowall.png')

img = Image.new('L', (2000, 600))
draw = ImageDraw.Draw(img)
draw.line((450, 0, 450, 600), fill=255, width=4)
#draw.line((1000, 600, 1000, 336), fill=255, width=4)
#draw.line((1400, 0, 1400, 264), fill=255, width=4)

img.save('2000_600_450.png')

img = Image.new('L', (2000, 600))
draw = ImageDraw.Draw(img)
draw.line((450, 400, 450, 600), fill=255, width=4)
#draw.line((1000, 600, 1000, 336), fill=255, width=4)
#draw.line((1400, 0, 1400, 264), fill=255, width=4)

img.save('2000_600_diffraction.png')

img = Image.new('L', (600, 600))
draw = ImageDraw.Draw(img)
#draw.line((450, 400, 450, 600), fill=255, width=4)
#draw.line((1000, 600, 1000, 336), fill=255, width=4)
#draw.line((1400, 0, 1400, 264), fill=255, width=4)

img.save('600_600.png')

img = Image.new('L', (2000, 1200))
draw = ImageDraw.Draw(img)
draw.line((1000, 80, 1000, 480), fill=255, width=4)
draw.line((1000, 720, 1000, 1120), fill=255, width=4)
draw.line((1240, 400, 1240, 800), fill=255, width=4)

img.save('case1.png')

img = Image.new('L', (2000, 1200))
draw = ImageDraw.Draw(img)
draw.line((1300, 80, 1000, 480), fill=255, width=4)
draw.line((1000, 720, 1300, 1120), fill=255, width=4)

img.save('case2.png')

img = Image.new('L', (2000, 1200))
draw = ImageDraw.Draw(img)
draw.line((700, 80, 1000, 480), fill=255, width=4)
draw.line((1000, 720, 700, 1120), fill=255, width=4)

img.save('case3.png')

img = Image.new('L', (2000, 1200))
draw = ImageDraw.Draw(img)
draw.line((1000, 80, 1000, 480), fill=255, width=4)
draw.line((1000, 720, 1000, 1120), fill=255, width=4)

img.save('case4.png')

img = Image.new('L', (2000, 1200))
draw = ImageDraw.Draw(img)
# 2本線
draw.line((800, 440, 1280, 440), fill=255, width=4)
draw.line((800, 760, 1280, 760), fill=255, width=4)
# 
draw.line((1280, 320, 1280, 440), fill=255, width=4)
draw.line((1280, 760, 1280, 1160), fill=255, width=4)
#
draw.line((1600, 320, 1600, 1160), fill=255, width=4)
#
draw.line((800, 600, 1440, 600), fill=255, width=4)
#
draw.line((1440, 600, 1600, 440), fill=255, width=4)
draw.line((1440, 600, 1600, 760), fill=255, width=4)
draw.line((1280, 1160, 1600, 1160), fill=255, width=4)
draw.line((1280, 320, 1600, 0), fill=255, width=4)
draw.line((1600, 0, 2000, 0), fill=255, width=4)

img.save('project.png')
# 1 pixel = 0.4 mm

# img = Image.new('L', (11250, 3750))
# draw = ImageDraw.Draw(img)
# draw.line((2500, 0, 2500, 1650), fill=255, width=25)
# draw.line((5000, 3750, 5000, 2100), fill=255, width=25)
# draw.line((7500, 0, 7500, 1650), fill=255, width=25)

# img.save('18000_6000_0.4mm.png')

# img = Image.new('L', (1125, 375))
# draw = ImageDraw.Draw(img)
# draw.line((250, 0, 250, 165), fill=255, width=2)
# draw.line((500, 375, 500, 210), fill=255, width=2)
# draw.line((750, 0, 750, 165), fill=255, width=2)

# img.save('1800_600_0.4mm.png')