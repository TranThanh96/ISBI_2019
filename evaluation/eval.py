import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())


    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="model")

    return graph


graph = load_graph('./eval/frozen_model.pb')

    # We can verify that we can access the list of operations in the graph
# for op in graph.get_operations():
#     print(op.name)

input_ph = graph.get_tensor_by_name('model/Placeholder:0')

istraining = graph.get_tensor_by_name('model/PlaceholderWithDefault:0')
predict = graph.get_tensor_by_name('model/resnet50/probs:0')


config = tf.ConfigProto(allow_soft_placement = True)
config.gpu_options.allow_growth = True

writer_train = tf.summary.FileWriter('./log')
with tf.Session(config=config,graph=graph) as sess: # another session
    writer_train.add_graph(sess.graph)
    print(sess)
    # output = sess.run(decode, feed_dict={Z: np.random.normal(size=(10,100)), istraining: False, img_1: })

    
    
    
    # plt.show()
    # variables_names = [v.name for v in tf.trainable_variables()]
    # values = sess.run(variables_names)
    # for k, v in zip(variables_names, values):
    #     print("Variable: ", k)
    #     print("Shape: ", v.shape)
    #     print(v)

    
#     print(sess.run(mul, feed_dict={x1: 5}))