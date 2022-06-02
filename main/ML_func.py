from utils_asl import *
import pandas as pd
from keras.models import load_model
 
vid = None
tensor = video2tensor(vid)

mpPose = mp.solutions.pose

pose = mpPose.Pose(static_image_mode=True,
model_complexity=2,
enable_segmentation=True,
min_detection_confidence=0.5) ### create a new MediaPipe Pose instance


# im
### SE PROCESAN 6 FRAMES POR SEGUNDO EN LUGAR DE LOS 25-30 ORIGINALES DEL VIDEO
S = []
cont = 0
T = tensor[::6] ### se va a tomar 1 de cada 6 frames del video
for t in T:
    _, series = frame_df(t, mpPose, pose)
S.append(series)
cont+=1
ccc = int_str(cont)
# print(ccc, end="|")
# if cont%10==0:
# print()
ds = pd.concat(S, axis=0).reset_index().drop(columns=['index']) ### dataframe | (20, 4)
print(ds.shape)

idx = ds.shape[0]
ids = np.arange(0, idx, 2)

LX = []
for i in ids:
    i2 = i+1
    d1 = ds.iloc[i:i+1]
    d2 = ds.iloc[i2:i2+1]
    ll = pd.concat([d1, d2], axis=0).mean(axis=0)
    ll2 = ll.tolist()
    LX += ll2
    

vector = np.array(LX)[:40].reshape(-1, 1).T

model = load_model("model.h5")
preds = model.predict(vector)
ind_max = preds.argmax()

frases = ['bd', 'bt', 'bn']
print("La frase que ha dicho es:", frases[ind_max])

