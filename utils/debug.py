import os




a = {'im1':[1,23,45], 'imd1':[2,22,11], 'iminf1':[3,89,81], 'im2':[1,23,45], 'imd2':[2,22,11], 'iminf2':[3,89,81]}


labels, shapes, segments = zip(*a.values())


print('labels', labels)
# print('shapes', shapes)
# print('segments', segments)
# print(labels[0])
# print(list(a.keys()))
# x, y, z =labels
# print(x)