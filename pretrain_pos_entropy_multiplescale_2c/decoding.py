for trial in [0]: 
    size = 15
    Pretest =  PretrainTest(holes = 0, weight_write =  'weights_cpu/rnn_1515tanh512_checkpoint{}'.format(trial), inputs_type=(0, 0))        
    weight = 'weights_cpu/rnn_1515tanh512_checkpoint{}'.format(trial)
    Prec0, prec_matrix0 = Pretest.decode(weight = weight, size_range = [size], size_test = [size], epsilon = 1)
    np.save('Dist{}'.format(trial), prec_matrix0)
