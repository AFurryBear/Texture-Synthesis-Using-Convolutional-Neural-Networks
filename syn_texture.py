from tensorflow_vgg import vgg16_avg_pool
import matplotlib.pyplot as plt
import numpy as np
import helper
import sys
import os
import tf_helper
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
filename = sys.argv[1]
def loss_function(m, texture_op, noise_layers):
    loss = tf.constant(0, dtype=tf.float32, name="Loss")

    for i in range(len(m)):
        texture_filters = np.squeeze(texture_op[m[i][0]], 0)
        texture_filters = np.reshape(texture_filters, newshape=(texture_filters.shape[0] * texture_filters.shape[1], texture_filters.shape[2]))
        gram_matrix_texture = np.matmul(texture_filters.T, texture_filters)

        noise_filters = tf.squeeze(noise_layers[m[i][0]], 0)
        noise_filters = tf.reshape(noise_filters, shape=(noise_filters.shape[0] * noise_filters.shape[1], noise_filters.shape[2]))
        gram_matrix_noise = tf.matmul(tf.transpose(noise_filters), noise_filters)

        denominator = (4 * tf.convert_to_tensor(texture_filters.shape[1], dtype=tf.float32) * tf.convert_to_tensor(texture_filters.shape[0], dtype=tf.float32))

        loss += m[i][1] * (tf.reduce_sum(tf.square(tf.subtract(gram_matrix_texture, gram_matrix_noise))) / tf.cast(denominator, tf.float32))
    
    return loss

def apply_mask(image_array):
    # Ensure the image array is square
    height, width, _ = image_array.shape
    if height != width:
        raise ValueError("Image array must be square (n x n pixels).")

    # Create a circular mask
    radius = width // 2
    Y, X = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((X - width / 2) ** 2 + (Y - height / 2) ** 2)
    mask = dist_from_center <= radius

    # Initialize an image array with an alpha channel
    masked_img = np.zeros((height, width, 4), dtype=np.float32)  # 4 channels: R, G, B, and Alpha

    # Copy the RGB channels
    masked_img[:, :, :3] = image_array

    # Apply the mask to the alpha channel
    masked_img[:, :, 3] = mask

    return masked_img

def intensity_match(texture_array, output_array):
    
    new_output = np.zeros(output_array.shape)
    mask = np.where(texture_array[:,:,3]>0,True,False)
    texture_data = texture_array[mask].flatten()
    output_data = output_array[mask].flatten()
    
    sorted_output = np.argsort(output_data)
    sorted_texture = np.argsort(texture_data)
    
    new_output_data = np.zeros(output_data.shape)
    
    new_output_data[sorted_output] = texture_data[sorted_texture]
    
    new_output[mask] = new_output_data.reshape(output_array[mask])

    return new_output
    

def post_process(texture_array, output_array):
    
    ## first apply a mask to both texture and output 
    masked_texture = apply_mask(texture_array)
    masked_output = apply_mask(output_array)
    ## use masked output to match intensity in masked texture
    new_output = intensity_match(masked_texture,masked_output)
    
    return new_output
    

def run_texture_synthesis(input_filename, processed_path, processed_filename, m, eps, op_dir, initial_filename, final_filename,i_w=256,i_h = 256):

    
    texture_array = helper.resize_and_rescale_img(input_filename, i_w, i_h, processed_path, processed_filename)
    texture_outputs = tf_helper.compute_tf_output(texture_array)
    
    tf.reset_default_graph()
    vgg = vgg16_avg_pool.Vgg16()

    random_ = tf.random_uniform(shape=texture_array.shape, minval=0, maxval=0.2)
    input_noise = tf.Variable(initial_value=random_, name='input_noise', dtype=tf.float32)

    vgg.build(input_noise)

    noise_layers_list = dict({0: vgg.conv1_1, 1: vgg.conv1_2, 2: vgg.pool1, 3: vgg.conv2_1, 4: vgg.conv2_2, 5: vgg.pool2, 6: vgg.conv3_1, 7: vgg.conv3_2, 
                   8: vgg.conv3_3, 9: vgg.pool3, 10: vgg.conv4_1, 11: vgg.conv4_2, 12: vgg.conv4_3, 13: vgg.pool4, 14: vgg.conv5_1, 15: vgg.conv5_2, 
                   16: vgg.conv5_3, 17: vgg.pool5 })

    loss = loss_function(m, texture_outputs, noise_layers_list)
    optimizer = tf.train.AdamOptimizer().minimize(loss)


    epochs = eps
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(tf.trainable_variables())
        init_noise = sess.run(input_noise)
        for i in range(epochs):
            _, s_loss = sess.run([optimizer, loss])
            if (i+1) % 1000 == 0:
                print("Epoch: {}/{}".format(i+1, epochs), " Loss: ", s_loss)
        final_noise = sess.run(input_noise)
    
    final_output = post_process(texture_array, final_noise)
    
    initial_noise = helper.post_process_and_display(init_noise, op_dir, initial_filename, save_file=False)
    final_output_ = helper.post_process_and_display(final_output, op_dir, final_filename)


# if __name__ == "__main__":
m = [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (9, 1), (10, 1), (11, 1), (12, 1), (13, 1)]
print("Configuration : 5 - Upto Pooling Layer 4")
root_path = "/gpfs/laur/data/xiongy/visualencoding/visual_stimuli/TextureSynthesis/"
os.chdir(root_path)
ip_f = "./image_resources/original/"+filename
processed = "./image_resources/processed/"
processed_path = os.path.splitext(filename)[0]+"_processed.jpg"
eps = 50000
output_dir = "./image_resources/outputs/"
noise_fn = os.path.splitext(filename)[0]+'_'+sys.argv[2]+"_C5_noise.jpg"
final_fn = os.path.splitext(filename)[0]+'_'+sys.argv[2]+"_C5_final.jpg"
run_texture_synthesis(ip_f, processed, processed_path, m, eps, output_dir, noise_fn, final_fn)