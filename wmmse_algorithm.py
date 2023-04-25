import numpy as np
import copy
import matplotlib.pyplot as plt
from utility_functions import compute_weighted_sum_rate


def compute_P(Phi_diag_elements, Sigma_diag_elements, mu):
    """Compute power for each step in bisection serach"""
    nr_of_BS_antennas = Phi_diag_elements.size
    mu_array = mu*np.ones(nr_of_BS_antennas)
    result = np.divide(Phi_diag_elements, (Sigma_diag_elements + mu_array)**2)
    result = np.sum(result)
    return result


def run_WMMSE(epsilon, power_tolerance, channel, selected_users, total_power, noise_power, user_weights, max_nr_of_iterations, log=False):
    """Function to run the WMMSE algorithm for WSR maximization in a MISO system"""

    nr_of_users, nr_of_BS_antennas = channel.shape
    break_condition = epsilon + 1  # break condition to stop the WMMSE iterations, i.e., exit the while

    # Init
    transimitter_precoder = np.zeros((nr_of_users,nr_of_BS_antennas)) + 1j*np.zeros((nr_of_users,nr_of_BS_antennas))
    for user_index in range(nr_of_users):
        if user_index in selected_users:
            transimitter_precoder[user_index,:] = channel[user_index,:]  # MRT precoder
    transimitter_precoder = np.sqrt(total_power)*transimitter_precoder/np.linalg.norm(transimitter_precoder)
    receiver_precoder = np.zeros(nr_of_users) + 1j*np.zeros(nr_of_users)
    mse_weights = np.finfo(float).eps*np.ones(nr_of_users)

    nr_of_iteration_counter = 0
    WSR = []  # to check if the WSR increases as each iteration of WMMSE
    WSR.append(compute_weighted_sum_rate(user_weights, channel, transimitter_precoder, noise_power, selected_users))
    new_receiver_precoder = np.zeros(nr_of_users) + 1j*np.zeros(nr_of_users)
    new_mse_weights = np.zeros(nr_of_users)
    new_transmitter_precoder = np.zeros((nr_of_users,nr_of_BS_antennas)) + 1j*np.zeros((nr_of_users,nr_of_BS_antennas))
    while break_condition >= epsilon and nr_of_iteration_counter <= max_nr_of_iterations:

        nr_of_iteration_counter = nr_of_iteration_counter + 1
        if log:
            print("WMMSE ITERATION:" + " " + str(nr_of_iteration_counter))


        # Optimize the receiver and MSE weights, i.e., variable u and w
        for user_index_1 in range(nr_of_users):
            if user_index_1 in selected_users:
                user_interference = 0
                for user_index_2 in range(nr_of_users):
                    if user_index_2 == user_index_1:
                        # user_signal = (np.absolute(np.matmul(np.conj(channel[user_index_1,:]),transimitter_precoder[user_index_1,:])))**2
                        user_signal = np.matmul(np.conj(channel[user_index_1,:]),transimitter_precoder[user_index_1,:])
                    elif user_index_2 in selected_users:
                        user_interference = user_interference + (np.absolute(np.matmul(np.conj(channel[user_index_1,:]),transimitter_precoder[user_index_2,:])))**2
            new_mse_weights[user_index_1] = (noise_power + user_interference + np.abs(user_signal)**2)/(noise_power + user_interference)
            new_receiver_precoder[user_index_1] = user_signal/(noise_power + user_interference + np.abs(user_signal)**2)


        # Optimize the precoder, i.e., variable v
        A = np.zeros((nr_of_BS_antennas, nr_of_BS_antennas)) + 1j*np.zeros((nr_of_BS_antennas, nr_of_BS_antennas))
        B = np.zeros((nr_of_users, nr_of_BS_antennas)) + 1j*np.zeros((nr_of_users, nr_of_BS_antennas))
        for user_index in range(nr_of_users):
            hh = np.matmul(channel[user_index,:][:,None],np.conj(channel[user_index,:][None,:]))
            A = A + user_weights[user_index]*new_mse_weights[user_index]*((np.absolute(new_receiver_precoder[user_index]))**2)*hh
            B[user_index,:] = user_weights[user_index]*new_mse_weights[user_index]*new_receiver_precoder[user_index]*channel[user_index,:]
        B = B.T

        Sigma_diag_elements_true, U = np.linalg.eigh(A)
        Sigma_diag_elements = copy.deepcopy(np.real(Sigma_diag_elements_true))
        Lambda = np.inner(B,np.conj(B))
        Phi = np.matmul(np.matmul(np.conj(U.T),Lambda),U)
        Phi_diag_elements_true = np.diag(Phi)
        Phi_diag_elements = copy.deepcopy(np.real(Phi_diag_elements_true))
        for i in range(len(Phi_diag_elements)):
            if Phi_diag_elements[i] < np.finfo(float).eps:
                Phi_diag_elements[i] = np.finfo(float).eps
            if Sigma_diag_elements[i] < np.finfo(float).eps:
                Sigma_diag_elements[i] = 0

        power = 0
        if np.prod(Sigma_diag_elements) != 0:  # det(A) != 0
            for user_index in range(nr_of_users):
                if user_index in selected_users:
                    v = np.matmul(np.linalg.pinv(A), B[:,user_index])
                    # v = np.linalg.solve(A,B[:,user_index])
                    power = power + np.linalg.norm(v)**2
        if np.prod(Sigma_diag_elements) != 0 and power <= total_power:
            mu_opt = 0
        else:  # Bisection search
            mu_low = np.sqrt(1/total_power*np.sum(Phi_diag_elements))
            mu_high = np.finfo(float).eps

            power_distance = []
            obtained_power = total_power + 2*power_tolerance
            while np.absolute(total_power - obtained_power) > power_tolerance:
                mu_new = (mu_high + mu_low)/2
                obtained_power = compute_P(Phi_diag_elements, Sigma_diag_elements, mu_new)
                power_distance.append(np.absolute(total_power - obtained_power))
                if obtained_power > total_power:
                    mu_high = mu_new
                if obtained_power <= total_power:
                    mu_low = mu_new

            mu_opt = mu_new
            if log:
                plt.title("Distance from the target value in bisection search")
                plt.plot(power_distance)
                plt.show()

        for user_index in range(nr_of_users):
            if user_index in selected_users:
                v = np.linalg.solve(A + mu_opt*np.eye(nr_of_BS_antennas),B[:,user_index])
                new_transmitter_precoder[user_index,:] = v;


        # Variable update
        mse_weights_selected_user = []
        new_mse_weights_selected_user = []
        for user_index in range(nr_of_users):
            if user_index in selected_users:
                mse_weights_selected_user.append(mse_weights[user_index])
                new_mse_weights_selected_user.append(new_mse_weights[user_index])
        break_condition = np.absolute(np.log(np.prod(new_mse_weights_selected_user)) - np.log(np.prod(mse_weights_selected_user)))  # mse_k = 1/w_k, -log2(mse_k) = log2(w_k)

        receiver_precoder = copy.deepcopy(new_receiver_precoder)
        mse_weights = copy.deepcopy(new_mse_weights)
        transimitter_precoder = copy.deepcopy(new_transmitter_precoder)
        WSR.append(compute_weighted_sum_rate(user_weights, channel, transimitter_precoder, noise_power, selected_users))

    if log:
        plt.title("Change of the WSR at each iteration of the WMMSE")
        plt.plot(WSR, 'b-o')
        plt.show()

    return transimitter_precoder, receiver_precoder, mse_weights, WSR[-1]
