Info_Is = []
for i in range(5):
    weight = 'weights_net1_shuffle/rnn_1515tanh512_checkpoint39_0_9'
    Info_A, Info_I, Info_P = Memory(weight, k_action = 1, k_stim = 1, k_internal = 1., epsilon = 0)
    print ('decay', Info_I)
    Info_Is.append(Info_I)
