
-- return function that returns network definition
return function(params)
    -- get number of classes from external parameters
    local nclasses = params.nclasses or 1

    -- get number of channels from external parameters
    local channels = 1
    -- params.inputShape may be nil during visualization
    if params.inputShape then
        channels = params.inputShape[1]
        assert(params.inputShape[2]==28 and params.inputShape[3]==28, 'Network expects 28x28 images')
    end

    if pcall(function() require('cudnn') end) then
       print('Using CuDNN backend')
       backend = cudnn
       convLayer = cudnn.SpatialConvolution
       convLayerName = 'cudnn.SpatialConvolution'
    else
       print('Failed to load cudnn backend (is libcudnn.so in your library path?)')
       if pcall(function() require('cunn') end) then
           print('Falling back to legacy cunn backend')
       else
           print('Failed to load cunn backend (is CUDA installed?)')
           print('Falling back to legacy nn backend')
       end
       backend = nn -- works with cunn or nn
       convLayer = nn.SpatialConvolutionMM
       convLayerName = 'nn.SpatialConvolutionMM'
    end

    require('rnn')

    -- hook
    function DataHook(dbinput, dblabel)
      input = {}
      label = {}

      strip_width = 9
      --strip_width = 28

      bs = dbinput:size(1)
      channels = dbinput:size(2)
      height = dbinput:size(3)
      width = dbinput:size(4)

      for i=1,(width - strip_width + 1) do
        input[i] = dbinput:sub(1, bs, 1, channels, 1, height, i, i+strip_width-1)
        label[i] = dblabel
      end

      return input, label
    end

    -- -- This is a variant of LeNet

    local recurrent = true
    local use_lstm = true
    local net

    -- convolutional part
    net = nn.Sequential()
      :add(nn.MulConstant(0.00390625))
      :add(backend.SpatialConvolution(channels,20,5,5,1,1,0)) -- channels*28*28 -> 20*24*24
      :add(backend.ReLU())
      :add(backend.SpatialConvolution(20,50,5,5,1,1,0)) -- 20*24*24 -> 50*20*20
      :add(backend.ReLU())
      :add(nn.View(-1):setNumInputDims(3))  -- 50*20*20 -> 20000
    -- insert LSTM or Linear layer between cnn and classifier
    n_conv_features = 1000
    n_linear_features = 500
    if recurrent then
      if use_lstm then
        net:add(nn.LSTM(n_conv_features, n_linear_features))
      else
        -- simple linear recurrent layer
        net:add(
          nn.Recurrence(
            nn.Sequential()
              :add(nn.ParallelTable()
                :add(nn.Linear(n_conv_features, n_linear_features))
                :add(nn.Linear(n_linear_features, n_linear_features)))
              :add(nn.CAddTable())
              :add(backend.ReLU()),
            n_linear_features,
            1))
        end
    else
      net:add(nn.Linear(n_conv_features, n_linear_features)):add(backend.ReLU())
    end
    -- classification part
    net:add(nn.Linear(n_linear_features, nclasses))
       :add(nn.LogSoftMax())
    -- wrap in a Sequencer
    net = nn.Sequencer(net)
    -- objective function (wrapped in a SequencerCriterion)
    local criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())

    return {
        model = net,
        loss = criterion,
        trainBatchSize = 8,
        validationBatchSize = 32,
        dataHook = DataHook,
    }
end

