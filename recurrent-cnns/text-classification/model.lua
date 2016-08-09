assert(pcall(function() require('dpnn') end), 'dpnn module required: luarocks install dpnn')
require('rnn')

-- return function that returns network definition
return function(params)
    -- get number of classes from external parameters (default to 14)
    local nclasses = params.nclasses or 14

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

    local feature_len = 1
    if params.inputShape then
        assert(params.inputShape[1]==1, 'Network expects 1xHxW images')
        params.inputShape:apply(function(x) feature_len=feature_len*x end)
    end

    local alphabet_len = 71 -- max index in input samples



    local net = nn.Sequential()
    -- feature_len x 1 x 1
    net:add(nn.View(-1,feature_len))
    -- feature_len
    net:add(nn.OneHot(alphabet_len))
    -- feature_len x alphabet_len                                 -- 123 x 71
    net:add(backend.TemporalConvolution(alphabet_len, 256, 7))    -- 117 x 256
    net:add(nn.Threshold())
    net:add(nn.TemporalMaxPooling(3, 3))                          -- 39 x 256
    -- 336 x 256
    net:add(backend.TemporalConvolution(256, 256, 7))             -- 33 x 256
    net:add(nn.Threshold())
    net:add(nn.TemporalMaxPooling(3, 3))                          -- 11 x 256
    -- 110 x 256
    net:add(backend.TemporalConvolution(256, 256, 3))             --  9 x 256
    net:add(nn.Threshold())
    -- 108 x 256
    net:add(backend.TemporalConvolution(256, 256, 3))
    net:add(nn.Threshold())
    -- 106 x 256
    net:add(backend.TemporalConvolution(256, 256, 3))
    net:add(nn.Threshold())
    -- 104 x 256
    net:add(backend.TemporalConvolution(256, 256, 3))
    net:add(nn.Threshold())
    net:add(nn.TemporalMaxPooling(3, 3))
    -- 34 x 256
    net:add(nn.Reshape(8704))
    -- 8704
    net:add(nn.Linear(8704, 1024))
    net:add(nn.Threshold())
    net:add(nn.Dropout(0.5))
    -- 1024
    net:add(nn.Linear(1024, 1024))
    net:add(nn.Threshold())
    net:add(nn.Dropout(0.5))
    -- 1024
    net:add(nn.Linear(1024, nclasses))
    net:add(backend.LogSoftMax())

    -- weight initialization
    local w,dw = net:getParameters()
    w:normal():mul(5e-2)

    local strip_width = 123
    local stride = 16
    local n_strips = math.floor((feature_len - strip_width) / stride + 1)
    print('strip_width', strip_width, 'stride', stride, 'n_strips', n_strips)

    local function fineTuneHook(net)

        local n_linear_features = 1024
        local n_conv_features = 1*256
        -- remove classifier part
        for i=26,18,-1 do net:remove(i) end
        -- fix weights of conv layers
        local function dummyFunc() end
        for i=1,15 do net:get(i).updateGradInput =  dummyFunc end
        net:add(nn.View(-1,n_conv_features))
        -- remove view layer
        net:remove(1)
        -- wrap in a sequencer
        net = nn.Sequential():add(nn.Sequencer(net))
        net:get(1).accGradParameters =  dummyFunc
        net:get(1).updateGradInput =  dummyFunc
        -- add recurrent layer
        net:add(nn.Contiguous())
        --net:add(cudnn.LSTM(n_conv_features,n_linear_features,1))
        --net:add(cudnn.RNN(n_conv_features,n_linear_features,1))
        net:add(cudnn.GRU(n_conv_features,n_linear_features,1))
        net:add(nn.Contiguous())
        net:add(nn.Sequencer(nn.Linear(n_linear_features,nclasses)))
        net:add(nn.Sequencer(backend.LogSoftMax()))

        return net
    end

    -- hook
    function DataHook(dbinput, dblabel)

      bs = dbinput:size(1)


      local X = dbinput:view(-1,feature_len)


      input = torch.CudaTensor(n_strips, bs, strip_width)
      label = torch.CudaTensor(n_strips, bs)
      for i=1,n_strips do
        input[i] = X:sub(1,bs, 1 + (i-1)*stride, (i-1)*stride + strip_width)
        if dblabel then
            label[i] = dblabel
        end
      end

      --print('modified input size',input:size())

      return input, label
    end

    return {
        model = net,
        --loss = nn.ClassNLLCriterion(),
        loss = nn.SequencerCriterion(nn.ClassNLLCriterion()),
        trainBatchSize = 8,
        validationBatchSize = 16,
        fineTuneHook =fineTuneHook,
        dataHook = DataHook,
    }
end
