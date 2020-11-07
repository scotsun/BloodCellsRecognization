from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from target_preparation import generator
from yolo3 import body

num_anchors = 3
num_classes = 5  # three types of cell

model_input = Input(shape=(416, 416, 3))
model_output = body(model_input, num_anchors, num_classes)
model = Model(model_input, model_output)

# model.summary()
# print('the num of layers ' + str(len(model.layers)))

batch_size = 100
num_classes = 3
annotation_folder = "C:/Users/sunm1/git/BloodCellsRecognization/BCCD-RBC-WBC-differentiation/Annotations"

model.compile(optimizer=Adam(lr=1e-3), loss="MeanSquaredError")
model.fit_generator(generator(batch_size, num_classes, annotation_folder))