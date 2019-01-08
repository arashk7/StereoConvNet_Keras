from StereoConvNet_AK import StereoConvNet as scn
from PIL import Image
import numpy as np
import os


def load_image(infilename: object) -> object:
    img = Image.open(infilename).convert('L')
    img.load()

    data = np.asarray(img, dtype="int32")
    return data


def load_input_dataset(path) -> object:
    files = os.listdir(path)
    x = np.zeros((len(files), int(100), int(300)))
    for i in range(len(files)):
        # img = load_image(path + "/" + files[i])
        img = Image.open(path + "/" + files[i]).convert('L')
        img.load()
        img = img.resize((x.shape[2], x.shape[1]), Image.ANTIALIAS)
        data = np.asarray(img, dtype="int32")
        x[i, :, :] = data

    return x


def load_output_dataset(path) -> object:
    files = os.listdir(path)
    x = np.zeros((len(files), int(100), int(150)))
    for i in range(len(files)):

        img = Image.open(path + "/" + files[i]).convert('L')
        img.load()
        img = img.resize((x.shape[2], x.shape[1]), Image.ANTIALIAS)
        data = np.asarray(img, dtype="int32")
        x[i, :, :] = data

    return x


path = "../../Dataset/DepthMap_dataset-master/Stereo"
x = load_input_dataset(path)
X = np.zeros((x.shape[0], x.shape[1], int(x.shape[2] / 2), 2))
X[:, :, :, 0] = x[:, :x.shape[1], :int(x.shape[2] / 2)]
X[:, :, :, 1] = x[:, :x.shape[1], int(x.shape[2] / 2):]

path = "../../Dataset/DepthMap_dataset-master/Depth_map"
y = load_output_dataset(path)
Y = np.zeros((y.shape[0], y.shape[1], y.shape[2], 1))
Y[:, :, :, 0] = y[:, :, :]


model = scn.get_model()
model.compile(optimizer=scn.get_optimizer(), loss=scn.get_loss_function(), metrics=['accuracy'])
model.fit(X, Y, batch_size=scn.get_batch_size(), epochs=scn.get_num_epoch())

# serialize model to JSON
model_json = model.to_json()
with open("model_StereoConvNet_1.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_StereoConvNet_1.h5")
print("Saved model to disk")
