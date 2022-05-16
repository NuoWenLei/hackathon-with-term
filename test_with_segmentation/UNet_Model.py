import tensorflow as tf, numpy as np

def process_to_tensor(im, im_x, im_y):
	im_arr = im.astype("float32") # np.array(im).astype("float32")
	if len(im_arr.shape) == 2:
		im_arr = im_arr[..., np.newaxis].astype("float32")
	return tf.image.resize_with_pad(im_arr, im_x, im_y)[tf.newaxis, ...]

def u_net_section(x, num_filters, section_num, num_convs = 2):
	for i in range(num_convs):
		x = tf.keras.layers.Conv2D(num_filters, kernel_size = (3,3), activation = "relu", padding = "same", name = f"section_{section_num}_conv_{i}")(x)
	return x

def u_net(num_convs_per_section, image_dims, batch_size, base_filters = 32, filter_multiply_rate = 2, num_sections = 3):

	input_layer = tf.keras.layers.Input(shape = image_dims, batch_size = batch_size)
	x = input_layer
	section_tracker = []
	for section in range(num_sections):
		if section == 0:
			filter = 16
		else:
			filter = base_filters
		x = u_net_section(x, filter, section, num_convs_per_section)
		section_tracker.append(x)
		x = tf.keras.layers.MaxPool2D(name = f"maxpool_downsize_{section}")(x)

	x = u_net_section(x, base_filters, num_sections, num_convs_per_section)

	for section in range(num_sections):
		if section == num_sections - 1:
			filter = 16
		else:
			filter = base_filters
		x = tf.keras.layers.UpSampling2D(name = f"upsample_upsize_{section}")(x)
		prev_section = section_tracker.pop()
		x = tf.concat([prev_section, x], axis = -1)
		x = u_net_section(x, filter, num_sections + 1 + section, num_convs_per_section)

	result = tf.keras.layers.Conv2D(1, (3,3), padding = "same", activation = "sigmoid")(x)

	return tf.keras.models.Model(inputs = [input_layer], outputs = [result])



