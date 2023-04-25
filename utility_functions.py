import numpy as np
import math
import tensorflow as tf


def compute_sinr(channel, precoder, noise_power, user_id, selected_users):
    nr_of_users = channel.shape[0]
    numerator = np.absolute(np.inner(np.conj(channel[user_id,:]), precoder[user_id,:]))**2  # |(h_i^H)(v_i)|^2, inner product

    inter_user_interference = 0
    for user_index in range(nr_of_users):
        if user_index != user_id and user_index in selected_users:
            inter_user_interference = inter_user_interference + np.absolute(np.inner(np.conj(channel[user_id,:]), precoder[user_index,:]))**2
    denominator = inter_user_interference + noise_power  # \sum_{j\uneq i}|(h_i^H)(v_j)|^2 + pn

    result = numerator/denominator
    return result


def compute_weighted_sum_rate(user_weights, channel, precoder, noise_power, selected_users):
    nr_of_users = np.size(channel, 0)

    result = 0
    for user_index in range(nr_of_users):
        if user_index in selected_users:
            user_sinr = compute_sinr(channel, precoder, noise_power, user_index, selected_users)
            result = result + user_weights[user_index]*np.log(1 + user_sinr)
    return result


def zero_forcing(channel_realization, total_power):
    """Compute the zero-frocing precoder"""
    # H^T*(conj(H)(H^T))^-1 = (conj(H)^H)(conj(H)(conj(H)^H))^-1    
    ZF_solution = np.dot(channel_realization.T, np.linalg.inv(np.inner(channel_realization.conjugate(), channel_realization)))   
    ZF_solution = ZF_solution*np.sqrt(total_power)/np.linalg.norm(ZF_solution, ord='fro')  # scaled according to the power constraint

    return ZF_solution.T


def regularized_zero_forcing(channel_realization, total_power, regularization_parameter=0, path_loss_option=False):
    """Compute regularized ZF precoder for multiuser MISO with non-homogeneous user SNR conditions"""
    nr_of_users = np.size(channel_realization, 0)
    if not path_loss_option:
        RZF_solution = np.dot(channel_realization.T, np.linalg.inv(np.inner(np.conj(channel_realization), channel_realization) + nr_of_users/total_power*np.eye(nr_of_users)))
    else:
        RZF_solution = np.dot(channel_realization.T, np.linalg.inv(np.inner(np.conj(channel_realization), channel_realization) + regularization_parameter*np.eye(nr_of_users)))
    RZF_solution = RZF_solution*np.sqrt(total_power)/np.linalg.norm(RZF_solution, ord='fro')  # scaled according to the power constraint

    return RZF_solution.T


def compute_channel(nr_of_BS_antennas, nr_of_users, total_power, path_loss_option=False, path_loss_range=[-5, 5]):
    """Generate a channel realization as well as the corresponding MRT precoder (used for initialization of the transmitter precoder, which serves as input in the computation graph of the deep unfolded network)"""
    channel_nn = []
    initial_transmitter_precoder = []
    channel_WMMSE = np.zeros((nr_of_users, nr_of_BS_antennas)) + 1j*np.zeros((nr_of_users, nr_of_BS_antennas))

    for i in range(nr_of_users):
        regularization_parameter_for_RZF_solution = 0
        path_loss = 0
        if path_loss_option:
            # regularized ZF (MMSE) precoding for multiuser MISO with non-homogeneous user SNR conditions
            path_loss = np.random.uniform(path_loss_range[0], path_loss_range[-1])
            regularization_parameter_for_RZF_solution = regularization_parameter_for_RZF_solution + 1/((10**(path_loss/10))*total_power)
        
        K = 0  # Rayleigh channel
        # K = np.random.randint(0,10)  # Rician channel
        h_LOS = np.random.uniform(0, 2*math.pi)
        result_real = np.sqrt(10**(path_loss/10))*np.sqrt(0.5)*(np.sqrt(K/(K+1))*math.cos(h_LOS) + np.sqrt(1/(K+1))*np.random.normal(size=(nr_of_BS_antennas,1)))
        result_imag = np.sqrt(10**(path_loss/10))*np.sqrt(0.5)*(np.sqrt(K/(K+1))*math.sin(h_LOS) + np.sqrt(1/(K+1))*np.random.normal(size=(nr_of_BS_antennas,1)))
        channel_WMMSE[i,:] = result_real.T + 1j*result_imag.T
        result_col_1 = np.vstack((result_real, result_imag))
        result_col_2 = np.vstack((-result_imag, result_real))
        result = np.hstack((result_col_1, result_col_2))  # real channel data
        initial_transmitter_precoder.append(result_col_1)  # MRT precoder: v_i = h_i^H
        channel_nn.append(result)

    initial_transmitter_precoder_array = np.array(initial_transmitter_precoder)
    initial_transmitter_precoder_array = np.sqrt(total_power)*initial_transmitter_precoder_array/np.linalg.norm(initial_transmitter_precoder_array)

    initial_transmitter_precoder = []
    for i in range(nr_of_users):
        initial_transmitter_precoder.append(initial_transmitter_precoder_array[i,:,:])

    return channel_nn, initial_transmitter_precoder, channel_WMMSE, regularization_parameter_for_RZF_solution


def compute_sinr_nn(channel, precoder, noise_power, user_id, nr_of_users):
    """Compute sinr of user applying to back propagation learning"""
    numerator = tf.reduce_sum((tf.matmul(tf.transpose(channel[user_id]), precoder[user_id]))**2)  # |(h_R^T-1j*H_I^T)(v_R+1j*V_I)|^2
    inter_user_interference = 0
    for user_index in range(nr_of_users):
        if user_index != user_id:
            inter_user_interference = inter_user_interference + tf.reduce_sum((tf.matmul(tf.transpose(channel[user_id]), precoder[user_index]))**2)
    denominator = inter_user_interference + noise_power

    result = numerator/denominator
    return result


def compute_WSR_nn(user_weights, channel, precoder, noise_power, nr_of_users, nr_of_samples_per_batch):
    result = 0
    for sample_index in range(nr_of_samples_per_batch):
        for user_index in range(nr_of_users):
            user_sinr = compute_sinr_nn(channel[sample_index], precoder[sample_index], noise_power, user_index, nr_of_users)
            result = result + user_weights[sample_index][user_index]*(tf.math.log(1+user_sinr))  # /tf.math.log(tf.cast(2.0, tf.float64)))
    return result


def PGD_step(init_u, init_w, name, user_weights, channel, initial_transmitter_precoder, total_power, nr_of_BS_antennas, nr_of_users, nr_of_samples_per_batch):
    """Builds one PGD iteration in the deep unfolded WMMSE network"""

    # $init_u \in (1, nr_of_users, 2, 1)$
    # $init_w \in (1, nr_of_users, 1)$
    # $user_weights \in (nr_of_samples_per_batch, nr_of_users, 1)$
    # $channel \in (nr_of_samples_per_batch, nr_of_users, 2*nr_of_BS_antennas, 2)$
    # $initial_transmitter_precoder \in (nr_of_samples_per_batch, nr_of_users, 2*nr_of_BS_antennas, 1)$

    with tf.variable_scope(name):

        receiver_precoder = tf.Variable(tf.constant(init_u, dtype=tf.float64), name="weight", dtype=tf.float64)
        mse_weights = tf.Variable(tf.constant(init_w, dtype=tf.float64), name="bias", dtype=tf.float64)

        # Computing weight matrix W        
        a1_exp = tf.tile(tf.expand_dims(mse_weights[:,0,:],-1),[nr_of_samples_per_batch,2*nr_of_BS_antennas,2*nr_of_BS_antennas])  # expand_dims is reverse of squeeze, tile is like kron(ones,a)
        a2_exp = tf.tile(tf.expand_dims(tf.reduce_sum((receiver_precoder[:,0,:,:])**2, axis=-2), -1), [nr_of_samples_per_batch,2*nr_of_BS_antennas,2*nr_of_BS_antennas])
        a3_exp = tf.tile(tf.expand_dims(user_weights[:,0,:],-1),[1,2*nr_of_BS_antennas,2*nr_of_BS_antennas])
        W = a1_exp*a2_exp*a3_exp*tf.matmul(channel[:,0,:,:],tf.transpose(channel[:,0,:,:],perm = [0,2,1]))  # \in (nr_of_samples_per_batch, 2*nr_of_BS_antennas,2*nr_of_BS_antennas)
        for i in range(1, nr_of_users):
            a1_exp = tf.tile(tf.expand_dims(mse_weights[:,i,:],-1),[nr_of_samples_per_batch,2*nr_of_BS_antennas,2*nr_of_BS_antennas])
            a2_exp = tf.tile(tf.expand_dims(tf.reduce_sum((receiver_precoder[:,i,:,:])**2, axis=-2), -1), [nr_of_samples_per_batch,2*nr_of_BS_antennas,2*nr_of_BS_antennas])
            a3_exp = tf.tile(tf.expand_dims(user_weights[:,i,:],-1),[1,2*nr_of_BS_antennas,2*nr_of_BS_antennas])
            W = W + a1_exp*a2_exp*a3_exp*tf.matmul(channel[:,i,:,:],tf.transpose(channel[:,i,:,:],perm = [0,2,1]))  # A = sum_{i=1}^N{alpha_i(w_i)abs(u_i)^2(h_i)(h_i^H)}

        # Compute bias b
        b = []
        augm_receiver_precoder = tf.tile(receiver_precoder, [nr_of_samples_per_batch,1,1,1])
        for i in range(nr_of_users):
            a1_exp = tf.tile(tf.expand_dims(mse_weights[:,i,:],-1),[nr_of_samples_per_batch,2*nr_of_BS_antennas,1])
            a2_exp = tf.tile(tf.expand_dims(user_weights[:,i,:],-1),[1,2*nr_of_BS_antennas,1])
            b.append(-1*a1_exp*a2_exp*tf.matmul(channel[:,i,:,:],augm_receiver_precoder[:,i,:,:]))

        temp = []
        for i in range(nr_of_users):
            temp.append(tf.add(tf.matmul(W,initial_transmitter_precoder[:,i,:,:]),b[i]))
        gradient = tf.transpose(tf.stack(temp), perm=[1,0,2,3])
        output_temp = initial_transmitter_precoder - gradient

        # projection operation
        output = []
        for i in range(nr_of_samples_per_batch):
            output.append(tf.cond((tf.linalg.norm(output_temp[i]))**2 < total_power, lambda: output_temp[i], lambda: tf.sqrt(tf.cast(total_power, tf.float64))*output_temp[i]/tf.linalg.norm(output_temp[i])))

        return tf.stack(output), receiver_precoder, mse_weights

