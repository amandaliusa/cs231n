import os

# assign directory
directory = 'demo/video'
 
# iterate over files in
# that directory
paths = []
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        paths.append(filename)

for i, video in enumerate(paths):
  print(i, video)
  #if (os.path.exists('demo/output/{}'.format(video[:-4]))): 
  #  print("Skipping as output already exists") 
  #else:
  
  print(video[:-4])
  if video[:-4] == 'IMG_0884': 

    os.system("python demo/vis_poseformer.py --video {}".format(video))
