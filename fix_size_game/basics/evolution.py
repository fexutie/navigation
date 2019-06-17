trial = 0
Performance_0 = []
for i in range(3):
    performance = []
    for j in range(11):
        if j == 0:
            Pretest = PretrainTest(weight_write = 'weights_cpu1/rnn_1515tanh512_checkpoint{}'.format(trial))
        else:
            Pretest = PretrainTest(weight_write = 'weights_fix/weights1/rnn_1515tanh512_checkpoint{}_{}_{}'.format(trial, i, j-1))
        Pretest.TestAllSizes(size_range = [15], limit_set = 4, test_size = [0])
        performance.append(Pretest.Performance)
    print (performance)
    Performance_0.append(performance)
