import visdom
import numpy as np
vis=visdom.Visdom()
for i in range(100):
    vis.text('hello world '+str(i),win='text1')
    vis.image(np.random.random((3,512,512)),win='image1')