from __future__ import division
from __future__ import print_function
import numpy as np
from scipy.linalg import solve
import random
from turtle import*
import pygame
import random

import sys

import matplotlib.pyplot as plt

# import subprocess
# with open("output.txt", "w+") as output:
#     subprocess.call(["python", "./RL2_PS2.py"], stdout=output);



def plot_graph(X,Y,XLabel,YLabel,Title):
    plt.plot(X,Y)
    plt.ylabel(YLabel)
    plt.xlabel(XLabel)
    plt.title(Title)
    plt.show()

def convertPair2single(x, y):
    return x + 10 * y


def convertSingle2Pair(s):
    return (s % 10, int((s - s % 10) / 10))

def weighted_random_by_dct(dct):
    rand_val = random.random()
    total = 0
    for k, v in dct.items():
        total += v
        if rand_val <= total:
            return k
    assert False, 'unreachable'

class Simulation:

    class Block:
        def __init__(self):
            self.cost = 0
            self.action = 'noAction'


    class Action:
        def __init__(self):
            self.prob = {}
            self.cost = {}


    class State:
        def __init__(self):
            self.actionSet = {}

    def __init__(self,num_of_states,terminal_state,States,discounted_factor,Matrix):
        self.num_of_states = num_of_states
        self.terminal_state = terminal_state
        self.States = States
        self.discounted_factor = discounted_factor
        self.StateSet = {}

        self.Plots_Display = False

        for state in Matrix:
            self.StateSet[state] = self.State()
            for action in Matrix[state]['prob']:
                X = self.StateSet[state]
                X.actionSet[action] = self.Action()
                for j in Matrix[state]['prob'][action]:
                    X.actionSet[action].prob[States[j]] = Matrix[state]['prob'][action][j]

    def next_state(self, src, action):
        prob_dict = self.StateSet[src].actionSet[action].prob
        return weighted_random_by_dct(prob_dict)


    def cost(self,src, dest):
        if src == terminal_state and dest == terminal_state:
            return 0
        elif dest == terminal_state:
            return 10
        else:
            return 0

    def policy_change(self,J1, J2):
        ctr = 0
        for state in J1:
            if (J1[state].action != J2[state].action):
                ctr += 1
        return ctr

    def J_num_plot(self,type,J,stage,state):
        X = []
        Y = []
        state_chosen = state
        for stage in range(stage):
            Y.append(J[stage][state_chosen].cost)
            X.append(stage + 1)

        plot_graph(X, Y, 'Iteration', 'J' + str(convertSingle2Pair(state_chosen)),
                   type+' Iteration, For Terminal state at ' + str(convertSingle2Pair(terminal_state)))

    def get_value_matrix(self, J,stages):
        matrix = np.zeros(shape = (stages,num_of_states))
        for i in range(stages):
            for j in range(num_of_states):
                matrix[i][j] = J[i][j].cost
        return matrix

    def get_policy_matrix(self, J, stages):
        matrix = np.empty([stages, num_of_states], dtype="S10")
        for i in range(stages):
            for j in range(num_of_states):
                matrix[i][j] = J[i][j].action
        return matrix

    def get_table(self,Matrix, stage):
        A = np.empty([10, 10], dtype="S10")
        for state in range(len(Matrix[stage])):
            A[convertSingle2Pair(state)[0],convertSingle2Pair(state)[1]] = Matrix[stage][state]
        print('<table align="center" border="1" cellpadding="10">')
        for j in range(10):
            print("<tr>")
            for i in range(10):
                print(' <td> '+"<img src='"+A[i][9-j]+".PNG' width='42' height='42' >" +' </td>')
            print("</tr>")
        print('</table>')

    def get_table_values(self,Matrix, stage):
        A = np.empty([10, 10], dtype="S10")
        for state in range(len(Matrix[stage])):
            A[convertSingle2Pair(state)[0],convertSingle2Pair(state)[1]] = Matrix[stage][state]
        print('<table align="center" border="1" cellpadding="10">')
        for j in range(10):
            print("<tr>")
            for i in range(10):
                print(' <td> '+A[i][9-j]+' </td>')
            print("</tr>")
        print('</table>')

    def value_iteration(self,threshold):

        J = [{x: self.Block() for x in States}]
        stage = 0
        converged = False
        while not converged:
            stage = stage + 1
            J.append({x: self.Block() for x in States})
            for state in J[stage]:
                J[stage][state].cost = -sys.maxsize + 1
                for action in self.StateSet[state].actionSet:
                    sum = 0
                    for state_next in self.StateSet[state].actionSet[action].prob:
                        sum = sum + self.StateSet[state].actionSet[action].prob[state_next] * (
                                self.cost(state, state_next) + discounted_factor * J[stage - 1][
                            state_next].cost)
                        # print ("detailed sum",stage, state, StateSet[state].actionSet[action].prob[state_next],sum)

                    if sum > J[stage][state].cost:
                        J[stage][state].cost = sum
                        J[stage][state].action = action
                # print (stage, state, J[stage][state].cost,J[stage][state].action)
            for state in J[stage]:
                if(abs(J[stage][state].cost-J[stage-1][state].cost)>threshold):
                    converged = False
                    break
            else:
                converged = True

        # print(self.get_policy_matrix(J,stage)[1])
        # print(self.get_policy_matrix(J,stage)[stage-1])
        k = stage-1
        print('<h2 align="center">Value Iteration, For '+str(k)+' iteration</h2>')
        self.get_table(self.get_policy_matrix(J, stage), k)
        print('<br></br>')
        self.get_table_values(self.get_value_matrix(J, stage), k)

        if True:
            X = []
            Y = []
            for stage in range(stage):
                x = max(abs(J[stage + 1][s].cost - J[stage][s].cost) for s in J[stage])
                Y.append(x)
                X.append(stage)

            plot_graph(X,Y,'Iteration','Value Difference','Value Iteration, For Terminal state at ' + str(convertSingle2Pair(terminal_state)))

            X = []
            Y = []
            for stage in range(stage-1):
                Y.append(self.policy_change(J[stage+1], J[stage + 2]))
                X.append(stage + 1)

            plot_graph(X, Y, 'Iteration', 'Policy Difference',
                       'Value Iteration, For Terminal state at ' + str(convertSingle2Pair(terminal_state)))

            self.J_num_plot('Value',J,stage,5)
            self.J_num_plot('Value',J,stage,12)
            self.J_num_plot('Value',J,stage,98)




    def policy_iteration(self):

        def policy_solver(J):
            p_pi = np.zeros(num_of_states)
            for i in range(num_of_states):
                for j in range(num_of_states):
                    p_pi[i] = p_pi[i] + self.StateSet[i].actionSet[J[i].action].prob.get(j, 0) * self.cost(i, j)

            T_pi = np.zeros(shape=(num_of_states, num_of_states))
            for i in range(num_of_states):
                for j in range(num_of_states):
                    T_pi[i][j] = self.StateSet[i].actionSet[J[i].action].prob.get(j, 0)

            T_pi = T_pi * discounted_factor
            A = T_pi - np.identity(num_of_states)
            sol = solve(A, -p_pi)

            for i in range(num_of_states):
                J[i].cost = sol[i]

        policy_iterate_num = 0
        J = [{x: self.Block() for x in States}]
        for state in J[policy_iterate_num]:
            if (self.StateSet[state].actionSet.get('UP') != None):
                J[policy_iterate_num][state].action = 'UP'
            else:
                J[policy_iterate_num][state].action = random.choice(list(self.StateSet[state].actionSet))
        policy_solver(J[policy_iterate_num])

        while True:
            policy_iterate_num = policy_iterate_num + 1
            J.append({x: self.Block() for x in States})
            for state in J[policy_iterate_num]:
                J[policy_iterate_num][state].cost = -sys.maxsize + 1
                for action in self.StateSet[state].actionSet:
                    sum = 0
                    for state_next in self.StateSet[state].actionSet[action].prob:
                        sum = sum + self.StateSet[state].actionSet[action].prob[state_next] * (
                                self.cost(state, state_next) + discounted_factor * J[policy_iterate_num - 1][
                            state_next].cost)
                        # print ("detailed sum",stage, state, StateSet[state].actionSet[action].prob[state_next],sum)

                    if sum > J[policy_iterate_num][state].cost:
                        J[policy_iterate_num][state].cost = sum
                        J[policy_iterate_num][state].action = action
                # print (stage, state, J[stage][state].cost,J[stage][state].action)

            if self.policy_change(J[policy_iterate_num - 1], J[policy_iterate_num]) == 0 or policy_iterate_num == 100:
                break

            policy_solver(J[policy_iterate_num])

        # print(self.get_value_matrix(J, policy_iterate_num))
        # print(self.get_policy_matrix(J, policy_iterate_num))
        k = policy_iterate_num-1
        print('<h2 align="center">Policy Iteration, For '+str(k)+' iteration</h2>')
        self.get_table(self.get_policy_matrix(J, policy_iterate_num), k)
        print('<br></br>')
        self.get_table_values(self.get_value_matrix(J, policy_iterate_num), k-1)

        if self.Plots_Display:
            X = []
            Y = []
            for stage in range(policy_iterate_num-1):
                x = max(abs(J[stage + 1][s].cost - J[stage][s].cost) for s in J[stage])
                Y.append(x)
                X.append(stage)

            plot_graph(X,Y,'Iteration','Value Difference','Policy Iteration, For Terminal state at ' + str(convertSingle2Pair(terminal_state)))

            X = []
            Y = []
            for stage in range(policy_iterate_num):
                Y.append(self.policy_change(J[stage], J[stage+1]))
                X.append(stage+1)

            plot_graph(X, Y, 'Iteration', 'Policy Difference',
                       'Policy Iteration, For Terminal state at ' + str(convertSingle2Pair(terminal_state)))

            self.J_num_plot('Policy',J,stage,5)
            self.J_num_plot('Policy',J,stage,12)
            self.J_num_plot('Policy',J,stage,98)

    def random_policy(self):
        r_policy = {}
        for state in self.StateSet:
            r_policy[state] = random.choice(self.StateSet[state].actionSet.keys())

        return r_policy


    def QRL(self, policy, alpha, str_state , epsilon, threshold):
        J = [{x: self.Block() for x in States}]
        J.append({x: self.Block() for x in States})
        episode = 0
        converged = False


        while not converged:
            episode = episode + 1
            cur_state = str_state

            for state in J[0]:
                J[1][state].cost = J[0][state].cost

            step = 0
            while not (cur_state == self.terminal_state):

                step = step +1

                if(random.random()<= epsilon):
                    curr_action = random.choice(self.StateSet[cur_state].actionSet.keys())
                else:
                    curr_action = policy[cur_state]

                prob_dict = self.StateSet[cur_state].actionSet[curr_action].prob

                next_state = weighted_random_by_dct(prob_dict)

                J[1][cur_state].cost = J[1][cur_state].cost + alpha * ( self.cost(cur_state,next_state) + self.discounted_factor*J[1][next_state].cost - J[1][cur_state].cost)



                cur_state = next_state

            print(step)

            max = -sys.maxsize + 1
            converged = True
            for state in J[1]:
                if max < abs(J[0][state].cost - J[1][state].cost):
                    max = abs(J[0][state].cost - J[1][state].cost)
                if (max > threshold):
                    converged = False
                    break

            for state in J[1]:
                J[0][state].cost = J[1][state].cost

        # print(self.get_policy_matrix(J,stage)[1])
        print(self.get_value_matrix(J,1))

        print(episode)

        # k = episode - 1
        # print('<h2 align="center">Value Iteration, For ' + str(k) + ' iteration</h2>')
        # self.get_table(self.get_policy_matrix(J, stage), k)
        # print('<br></br>')
        # self.get_table_values(self.get_value_matrix(J, stage), k)
        #
        # if False:
        #     X = []
        #     Y = []
        #     for stage in range(stage):
        #         x = max(abs(J[stage + 1][s].cost - J[stage][s].cost) for s in J[stage])
        #         Y.append(x)
        #         X.append(stage)
        #
        #     plot_graph(X, Y, 'Iteration', 'Value Difference',
        #                'Value Iteration, For Terminal state at ' + str(convertSingle2Pair(terminal_state)))
        #
        #     X = []
        #     Y = []
        #     for stage in range(stage - 1):
        #         Y.append(self.policy_change(J[stage + 1], J[stage + 2]))
        #         X.append(stage + 1)
        #
        #     plot_graph(X, Y, 'Iteration', 'Policy Difference',
        #                'Value Iteration, For Terminal state at ' + str(convertSingle2Pair(terminal_state)))
        #
        #     self.J_num_plot('Value', J, stage, 5)
        #     self.J_num_plot('Value', J, stage, 12)
        #     self.J_num_plot('Value', J, stage, 98)


    def Q_learning(self, policy, alpha, str_state ,str_action , epsilon, threshold):
        Q = [{x: {a: self.Block() for a in self.StateSet[x].actionSet} for x in States}]
        Q.append({x: {a: self.Block() for a in self.StateSet[x].actionSet} for x in States})
        episode = 0
        converged = False


        while not converged:
            episode = episode + 1
            cur_state,curr_action = str_state, str_action

            for state in Q[0]:
                for action in Q[0][state]:
                    Q[1][state][action].cost = Q[0][state][action].cost

            step = 0

            if (random.random() <= epsilon):
                curr_action = random.choice(self.StateSet[cur_state].actionSet.keys())
            else:
                curr_action = policy[cur_state]

            while not (cur_state == self.terminal_state):

                step = step +1

                prob_dict = self.StateSet[cur_state].actionSet[curr_action].prob

                next_state = weighted_random_by_dct(prob_dict)

                if (random.random() <= epsilon):
                    next_action = random.choice(self.StateSet[next_state].actionSet.keys())
                else:
                    next_action = policy[next_state]


                Q[1][cur_state][curr_action].cost = Q[1][cur_state][curr_action].cost + alpha * \
                                                    ( self.cost(cur_state,next_state) + self.discounted_factor*Q[1]
                                                    [next_state][next_action].cost - Q[1][cur_state][curr_action].cost)



                cur_state = next_state
                curr_action = next_action

            # print(step)

            # for state in Q[1]:
            #     for action in Q[1][state]:
            #         print(Q[1][state][action].cost, end=' ')
            # print()

            max = -sys.maxsize + 1
            converged = True
            for state in Q[1]:
                if converged:
                    for action in Q[1][state]:
                        if max < abs(Q[0][state][action].cost - Q[1][state][action].cost):
                            max = abs(Q[0][state][action].cost - Q[1][state][action].cost)
                        if (max > threshold):
                            converged = False
                            break

            print(max, "max_diff")

            for state in Q[0]:
                for action in Q[0][state]:
                    Q[0][state][action].cost = Q[1][state][action].cost

        # print(self.get_policy_matrix(J,stage)[1])

        J = []
        J.append({})
        for state in Q[0]:
            max = -sys.maxsize + 1
            for action in Q[0][state]:
                if max < Q[0][state][action].cost:
                    max = Q[0][state][action].cost
            J[0][state] = self.Block()
            J[0][state].cost = max

        # print(self.get_value_matrix(J,1))

        print(episode)

        # k = episode - 1
        # print('<h2 align="center">Value Iteration, For ' + str(k) + ' iteration</h2>')
        # self.get_table(self.get_policy_matrix(J, stage), k)
        # print('<br></br>')
        # self.get_table_values(self.get_value_matrix(J, stage), k)
        #
        # if False:
        #     X = []
        #     Y = []
        #     for stage in range(stage):
        #         x = max(abs(J[stage + 1][s].cost - J[stage][s].cost) for s in J[stage])
        #         Y.append(x)
        #         X.append(stage)
        #
        #     plot_graph(X, Y, 'Iteration', 'Value Difference',
        #                'Value Iteration, For Terminal state at ' + str(convertSingle2Pair(terminal_state)))
        #
        #     X = []
        #     Y = []
        #     for stage in range(stage - 1):
        #         Y.append(self.policy_change(J[stage + 1], J[stage + 2]))
        #         X.append(stage + 1)
        #
        #     plot_graph(X, Y, 'Iteration', 'Policy Difference',
        #                'Value Iteration, For Terminal state at ' + str(convertSingle2Pair(terminal_state)))
        #
        #     self.J_num_plot('Value', J, stage, 5)
        #     self.J_num_plot('Value', J, stage, 12)
        #     self.J_num_plot('Value', J, stage, 98)

    def TD(self, policy, alpha, str_state , epsilon, threshold):
        J = [{x: self.Block() for x in States}]
        J.append({x: self.Block() for x in States})
        episode = 0
        converged = False


        while not converged:
            episode = episode + 1
            cur_state = str_state

            for state in J[0]:
                J[1][state].cost = J[0][state].cost

            step = 0
            while not (cur_state == self.terminal_state):

                step = step +1

                if(random.random()<= epsilon):
                    curr_action = random.choice(self.StateSet[cur_state].actionSet.keys())
                else:
                    curr_action = policy[cur_state]

                prob_dict = self.StateSet[cur_state].actionSet[curr_action].prob

                next_state = weighted_random_by_dct(prob_dict)

                J[1][cur_state].cost = J[1][cur_state].cost + alpha * ( self.cost(cur_state,next_state) + self.discounted_factor*J[1][next_state].cost - J[1][cur_state].cost)



                cur_state = next_state

            print(step)

            max = -sys.maxsize + 1
            converged = True
            for state in J[1]:
                if max < abs(J[0][state].cost - J[1][state].cost):
                    max = abs(J[0][state].cost - J[1][state].cost)
                if (max > threshold):
                    converged = False
                    break

            for state in J[1]:
                J[0][state].cost = J[1][state].cost

        # print(self.get_policy_matrix(J,stage)[1])
        print(self.get_value_matrix(J,1))

        print(episode)

        # k = episode - 1
        # print('<h2 align="center">Value Iteration, For ' + str(k) + ' iteration</h2>')
        # self.get_table(self.get_policy_matrix(J, stage), k)
        # print('<br></br>')
        # self.get_table_values(self.get_value_matrix(J, stage), k)
        #
        # if False:
        #     X = []
        #     Y = []
        #     for stage in range(stage):
        #         x = max(abs(J[stage + 1][s].cost - J[stage][s].cost) for s in J[stage])
        #         Y.append(x)
        #         X.append(stage)
        #
        #     plot_graph(X, Y, 'Iteration', 'Value Difference',
        #                'Value Iteration, For Terminal state at ' + str(convertSingle2Pair(terminal_state)))
        #
        #     X = []
        #     Y = []
        #     for stage in range(stage - 1):
        #         Y.append(self.policy_change(J[stage + 1], J[stage + 2]))
        #         X.append(stage + 1)
        #
        #     plot_graph(X, Y, 'Iteration', 'Policy Difference',
        #                'Value Iteration, For Terminal state at ' + str(convertSingle2Pair(terminal_state)))
        #
        #     self.J_num_plot('Value', J, stage, 5)
        #     self.J_num_plot('Value', J, stage, 12)
        #     self.J_num_plot('Value', J, stage, 98)

if __name__ == "__main__":

    num_of_states = 100
    terminal_state = convertPair2single(3, 0)
    discounted_factor = 0.7
    threshold = 0.1

    print ('Terminal State:', terminal_state)

    States = [i for i in range(100)]

    Matrix = {}

    # convention is 'i' is X axis ans 'j' is Y axis
    # order is UP DOWN LEFT RIGHT
    for j in range(10):
        for i in range(10):
            UP = {convertPair2single(i, j + 1): 8 / 10, convertPair2single(i - 1, j): 2 / 30,
                  convertPair2single(i + 1, j): 2 / 30, convertPair2single(i, j - 1): 2 / 30}
            DOWN = {convertPair2single(i, j - 1): 8 / 10, convertPair2single(i - 1, j): 2 / 30,
                    convertPair2single(i + 1, j): 2 / 30, convertPair2single(i, j + 1): 2 / 30}
            LEFT = {convertPair2single(i, j + 1): 2 / 30, convertPair2single(i, j - 1): 2 / 30,
                    convertPair2single(i - 1, j): 8 / 10, convertPair2single(i + 1, j): 2 / 30}
            RIGHT = {convertPair2single(i, j + 1): 2 / 30, convertPair2single(i, j - 1): 2 / 30,
                     convertPair2single(i + 1, j): 8 / 10, convertPair2single(i - 1, j): 2 / 30}

            Mat_prob = {'UP': UP, 'DOWN': DOWN, 'LEFT': LEFT, 'RIGHT': RIGHT}
            Matrix[convertPair2single(i, j)] = {'prob': Mat_prob}

    i = 0
    for j in range(10):
        dictionary = Matrix[convertPair2single(i, j)]['prob']['LEFT']
        dictionary[convertPair2single(i, j)] = dictionary.get(convertPair2single(i, j), 0) + dictionary.pop(
            convertPair2single(i - 1, j))

        dictionary = Matrix[convertPair2single(i, j)]['prob']['UP']
        dictionary[convertPair2single(i, j)] = dictionary.get(convertPair2single(i, j), 0) + dictionary.pop(
            convertPair2single(i - 1, j))

        dictionary = Matrix[convertPair2single(i, j)]['prob']['DOWN']
        dictionary[convertPair2single(i, j)] = dictionary.get(convertPair2single(i, j), 0) + dictionary.pop(
            convertPair2single(i - 1, j))

        dictionary = Matrix[convertPair2single(i, j)]['prob']['RIGHT']
        dictionary[convertPair2single(i, j)] = dictionary.get(convertPair2single(i, j), 0) + dictionary.pop(
            convertPair2single(i - 1, j))

    i = 9
    for j in range(10):
        dictionary = Matrix[convertPair2single(i, j)]['prob']['RIGHT']
        dictionary[convertPair2single(i, j)] = dictionary.get(convertPair2single(i, j), 0) + dictionary.pop(
            convertPair2single(i + 1, j))

        dictionary = Matrix[convertPair2single(i, j)]['prob']['UP']
        dictionary[convertPair2single(i, j)] = dictionary.get(convertPair2single(i, j), 0) + dictionary.pop(
            convertPair2single(i + 1, j))

        dictionary = Matrix[convertPair2single(i, j)]['prob']['DOWN']
        dictionary[convertPair2single(i, j)] = dictionary.get(convertPair2single(i, j), 0) + dictionary.pop(
            convertPair2single(i + 1, j))

        dictionary = Matrix[convertPair2single(i, j)]['prob']['LEFT']
        dictionary[convertPair2single(i, j)] = dictionary.get(convertPair2single(i, j), 0) + dictionary.pop(
            convertPair2single(i + 1, j))

    j = 0
    for i in range(10):
        dictionary = Matrix[convertPair2single(i, j)]['prob']['DOWN']
        dictionary[convertPair2single(i, j)] = dictionary.get(convertPair2single(i, j), 0) + dictionary.pop(
            convertPair2single(i, j - 1))

        dictionary = Matrix[convertPair2single(i, j)]['prob']['LEFT']
        dictionary[convertPair2single(i, j)] = dictionary.get(convertPair2single(i, j), 0) + dictionary.pop(
            convertPair2single(i, j - 1))

        dictionary = Matrix[convertPair2single(i, j)]['prob']['RIGHT']
        dictionary[convertPair2single(i, j)] = dictionary.get(convertPair2single(i, j), 0) + dictionary.pop(
            convertPair2single(i, j - 1))

        dictionary = Matrix[convertPair2single(i, j)]['prob']['UP']
        dictionary[convertPair2single(i, j)] = dictionary.get(convertPair2single(i, j), 0) + dictionary.pop(
            convertPair2single(i, j - 1))

    j = 9
    for i in range(10):
        dictionary = Matrix[convertPair2single(i, j)]['prob']['UP']
        dictionary[convertPair2single(i, j)] = dictionary.get(convertPair2single(i, j), 0) + dictionary.pop(
            convertPair2single(i, j + 1))

        dictionary = Matrix[convertPair2single(i, j)]['prob']['RIGHT']
        dictionary[convertPair2single(i, j)] = dictionary.get(convertPair2single(i, j), 0) + dictionary.pop(
            convertPair2single(i, j + 1))

        dictionary = Matrix[convertPair2single(i, j)]['prob']['LEFT']
        dictionary[convertPair2single(i, j)] = dictionary.get(convertPair2single(i, j), 0) + dictionary.pop(
            convertPair2single(i, j + 1))

        dictionary = Matrix[convertPair2single(i, j)]['prob']['DOWN']
        dictionary[convertPair2single(i, j)] = dictionary.get(convertPair2single(i, j), 0) + dictionary.pop(
            convertPair2single(i, j + 1))

    Matrix[convertPair2single(7, 9)]['prob'] = {'TELEPORT': {convertPair2single(7, 1): 1}}
    Matrix[convertPair2single(0, 0)]['prob'] = {
        'TELEPORT': {convertPair2single(2, 3): 1 / 4, convertPair2single(2, 4): 1 / 4, convertPair2single(2, 5): 1 / 4,
                     convertPair2single(2, 6): 1 / 4}}

    Matrix[terminal_state]['prob'] = {'HALT': {terminal_state: 1}}

    simulation = Simulation(num_of_states,terminal_state,States,discounted_factor,Matrix)


    # simulation.value_iteration(threshold)
    # simulation.policy_iteration()

    policy = simulation.random_policy()
    simulation.TD(policy,0.1,0,0.01, threshold)
    # simulation.Q_learning(policy,0.1,0,'UP',0.01, threshold)






