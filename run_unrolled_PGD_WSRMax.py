# Import libraries
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
from utility_functions import compute_channel, compute_WSR_nn, PGD_step
from utility_functions import zero_forcing, regularized_zero_forcing, compute_weighted_sum_rate
from wmmse_algorithm import run_WMMSE


# Set variables
nr_of_users = 4
nr_of_BS_antennas = 4
scheduled_users = [0, 1, 2, 3]  # array of scheduled users
epsilon = 0.0001  # the stopping criterion of the WMMSE iterations
power_tolerance = 0.0001  # the stopping criterion of the bisection search at each iteration of WMMSE
total_power = 10  # maximum transmit power at the BS
noise_power = 1
path_loss_option = False  # used to add a random path loss (drawn from a uniform distribution) to the channel of each user
path_loss_range = [-5, 5]  # uniform distribution of the random path loss (in dB)
nr_of_batches_training = 10000  # used for training
nr_of_batches_test = 100  # used for testing
nr_of_samples_per_batch = 64
nr_of_iterations = 4  # for the WMMSE algorithm
nr_of_iterations_nn = 4  # for the deep unfolded WMMSE


# User rate weights
user_weights = np.reshape(np.ones(nr_of_users*nr_of_samples_per_batch), (nr_of_samples_per_batch, nr_of_users, 1))  # for NN loss function
user_weights_for_regular_WMMSE = np.ones(nr_of_users)

channel_realization_nn,init_transmitter_precoder,channel_realization_regular,regularization_parameter_for_RZF_solution = compute_channel(nr_of_BS_antennas, nr_of_users, total_power, path_loss_option, path_loss_range)
_,_,_,WSR_WMMSE_one_sample = run_WMMSE(epsilon, power_tolerance, channel_realization_regular, scheduled_users, total_power, noise_power, user_weights_for_regular_WMMSE, nr_of_iterations)


# Tensorflow computation graph to run the deep unfolded WMMSE algorithm
tf.reset_default_graph()

channel_input = tf.placeholder(tf.float64, shape=None, name='channel_input')
initial_tp = tf.placeholder(tf.float64, shape=None, name='initial_transmitter_precoder')
initial_transmitter_precoder = initial_tp

# Arrays that contain the initialization values of the step size
mse_weights1_init = np.expand_dims(np.reshape([1.0 for i in range(nr_of_users)],(nr_of_users,1)), 0)
mse_weights2_init = np.expand_dims(np.reshape([1.0 for i in range(nr_of_users)],(nr_of_users,1)), 0)
mse_weights3_init = np.expand_dims(np.reshape([1.0 for i in range(nr_of_users)],(nr_of_users,1)), 0)
mse_weights4_init = np.expand_dims(np.reshape([1.0 for i in range(nr_of_users)],(nr_of_users,1)), 0)
result_col_1 = np.reshape([0.7 for i in range(nr_of_users)],(nr_of_users,1))
result_col_2 = np.reshape([0.7 for i in range(nr_of_users)],(nr_of_users,1))
receiver_precoder1_init = np.expand_dims(np.hstack((result_col_1,result_col_2)), (0,-1))
receiver_precoder2_init = np.expand_dims(np.hstack((result_col_1,result_col_2)), (0,-1))
receiver_precoder3_init = np.expand_dims(np.hstack((result_col_1,result_col_2)), (0,-1))
receiver_precoder4_init = np.expand_dims(np.hstack((result_col_1,result_col_2)), (0,-1))
# mse_weights1_init = np.reshape([1.0 for i in range(nr_of_users)],(nr_of_users,1))[None, :, :]
# receiver_precoder1_init = np.hstack((result_col_1,result_col_2))[None, :, :, None]


profit = []  # stores the WSR obtained at each iteration

# Layer 1
transmitter_precoder1, receiver_precoder1, mse_weights1 = PGD_step(receiver_precoder1_init, mse_weights1_init, 'PGD_step1', user_weights, channel_input, initial_transmitter_precoder, total_power, nr_of_BS_antennas, nr_of_users, nr_of_samples_per_batch)
profit.append(compute_WSR_nn(user_weights, channel_input, transmitter_precoder1, noise_power, nr_of_users, nr_of_samples_per_batch))

# Layer 2
transmitter_precoder2, receiver_precoder2, mse_weights2 = PGD_step(receiver_precoder2_init, mse_weights2_init, 'PGD_step2', user_weights, channel_input, transmitter_precoder1, total_power, nr_of_BS_antennas, nr_of_users, nr_of_samples_per_batch)
profit.append(compute_WSR_nn(user_weights, channel_input, transmitter_precoder2, noise_power, nr_of_users, nr_of_samples_per_batch))

# Layer 3
transmitter_precoder3, receiver_precoder3, mse_weights3 = PGD_step(receiver_precoder3_init, mse_weights3_init, 'PGD_step3', user_weights, channel_input, transmitter_precoder2, total_power, nr_of_BS_antennas, nr_of_users, nr_of_samples_per_batch)
profit.append(compute_WSR_nn(user_weights, channel_input, transmitter_precoder3, noise_power, nr_of_users, nr_of_samples_per_batch))

# Layer 4
transmitter_precoder4, receiver_precoder4, mse_weights4 = PGD_step(receiver_precoder4_init, mse_weights4_init, 'PGD_step4', user_weights, channel_input, transmitter_precoder3, total_power, nr_of_BS_antennas, nr_of_users, nr_of_samples_per_batch)
profit.append(compute_WSR_nn(user_weights, channel_input, transmitter_precoder4, noise_power, nr_of_users, nr_of_samples_per_batch))

all_mse_weights = tf.stack([mse_weights1[0,:,0], mse_weights2[0,:,0], mse_weights3[0,:,0], mse_weights4[0,:,0]])
all_receiver_precoder = tf.stack([receiver_precoder1[0,:,:,0], receiver_precoder2[0,:,:,0], receiver_precoder3[0,:,:,0], receiver_precoder4[0,:,:,0]])

# Output
final_precoder = transmitter_precoder4
WSR_final = compute_WSR_nn(user_weights, channel_input, final_precoder, noise_power, nr_of_users, nr_of_samples_per_batch)/nr_of_samples_per_batch

WSR = tf.reduce_sum(tf.stack(profit))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(-WSR)


# running the deep unfolded WMMSE
WSR_WMMSE = []
WSR_ZF = []
WSR_RZF = []
WSR_nn = []
training_loss = []

with tf.Session() as sess:
    print("start of session")
    start_of_time = time.time()
    sess.run(tf.global_variables_initializer())

    for i in range(nr_of_batches_training):
        batch_for_training = []
        initial_transmitter_precoder_batch = []

        # Buliding a batch for training
        for ii in range(nr_of_samples_per_batch):
            channel_realization_nn,init_transmitter_precoder,_,_ = compute_channel(nr_of_BS_antennas, nr_of_users, total_power, path_loss_option, path_loss_range)
            batch_for_training.append(channel_realization_nn)
            initial_transmitter_precoder_batch.append(init_transmitter_precoder)
        # Training
        sess.run(optimizer, feed_dict={channel_input:batch_for_training, initial_tp:initial_transmitter_precoder_batch})
        training_loss.append(-1*(sess.run(WSR,feed_dict={channel_input:batch_for_training, initial_tp:initial_transmitter_precoder_batch})))

    print("mse_weights:\n", sess.run(all_mse_weights))
    print("receiver_precoder:\n", sess.run(all_receiver_precoder))

    print("Training took:", time.time()-start_of_time)
    
    # For repeatability
    np.random.seed(1234)
    
    for i in range(nr_of_batches_test):
        batch_for_tesing = []
        initial_transmitter_precoder_batch = []
        WSR_WMMSE_batch = 0.0
        WSR_ZF_batch = 0.0
        WSR_RZF_batch = 0.0
        
        for ii in range(nr_of_samples_per_batch):
            channel_realization_nn,init_transmitter_precoder,channel_realization_regular,regularization_parameter_for_RZF_solution = compute_channel(nr_of_BS_antennas, nr_of_users, total_power, path_loss_option, path_loss_range)
            
            # run ZF precoding
            precoder_ZF = zero_forcing(channel_realization_regular, total_power)
            WSR_ZF_batch = WSR_ZF_batch + compute_weighted_sum_rate(user_weights_for_regular_WMMSE, channel_realization_regular, precoder_ZF, noise_power, scheduled_users)

            # run RZF precoding
            precoder_RZF = regularized_zero_forcing(channel_realization_regular, total_power, regularization_parameter_for_RZF_solution, path_loss_option)
            WSR_RZF_batch = WSR_RZF_batch + compute_weighted_sum_rate(user_weights_for_regular_WMMSE, channel_realization_regular, precoder_RZF, noise_power, scheduled_users)

            # run WMMSE precoding
            _,_,_,WSR_WMMSE_one_sample = run_WMMSE(epsilon, power_tolerance, channel_realization_regular, scheduled_users, total_power, noise_power, user_weights_for_regular_WMMSE, nr_of_iterations)
            WSR_WMMSE_batch = WSR_WMMSE_batch + WSR_WMMSE_one_sample
            
            batch_for_tesing.append(channel_realization_nn)
            initial_transmitter_precoder_batch.append(init_transmitter_precoder)
        
        # Testing
        WSR_nn.append(sess.run(WSR_final, feed_dict={channel_input:batch_for_tesing, initial_tp:initial_transmitter_precoder_batch}))
        WSR_WMMSE.append(WSR_WMMSE_batch/nr_of_samples_per_batch)
        WSR_ZF.append(WSR_ZF_batch/nr_of_samples_per_batch)
        WSR_RZF.append(WSR_RZF_batch/nr_of_samples_per_batch)

print("Training and testing took:", time.time()-start_of_time)
print("WSR achieved by NN:", np.mean(WSR_nn))
print("WSR achieved by WMMSE:", np.mean(WSR_WMMSE))
print("WSR achieved by ZF:", np.mean(WSR_ZF))
print("WSR achieved by RZF:", np.mean(WSR_RZF))

plt.figure()
plt.plot(training_loss)
plt.ylabel("Training loss")
plt.xlabel("Sample index")

