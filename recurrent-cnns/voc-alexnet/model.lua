if pcall(function() require('cudnn') end) then
   print('Using CuDNN backend')
   backend = cudnn
   convLayer = cudnn.SpatialConvolution
   convLayerName = 'cudnn.SpatialConvolution'
   cudnn.fastest = true
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

function createModel(nGPU, channels, nClasses)



    local recurrent = true
    local use_lstm = false
    local use_gru = true
    local net
    local v = 4

    local n_conv_features
    local strip_width
    local stride

    -- this is alexnet as presented in Krizhevsky et al., 2012
    if v == 1 then
      net = nn.Sequential()
      net:add(convLayer(channels,96,11,11,4,4,0,0))       -- 224 ->  55
      net:add(backend.ReLU(true))
      --features:add(backend.SpatialMaxPooling(3,3,2,2))         --  55 ->  27
      net:add(convLayer(96,256,5,5,1,1,0,0))              --  27 ->  27
      net:add(backend.ReLU(true))
      --features:add(backend.SpatialMaxPooling(3,3,2,2))         --  27 ->  13
      net:add(convLayer(256,384,3,3,1,1,0,0))             --  13 ->  13
      net:add(backend.ReLU(true))
      net:add(convLayer(384,384,3,3,1,1,0,0))             --  13 ->  13
      net:add(backend.ReLU(true))
      net:add(convLayer(384,256,3,3,1,1,0,0))             --  13 ->  13
      net:add(backend.ReLU(true))
      --features:add(backend.SpatialMaxPooling(3,3,2,2))         --  13 ->  6
      n_conv_features = 11264 -- 495616 -- 11264
      strip_width = 52 -- 224
      stride = 40
    elseif v==2 then
      net = nn.Sequential()
      net:add(convLayer(channels,96,11,11,4,4,2,2)) --     -- 224 x 128 ->  55 x 31
      net:add(backend.ReLU(true))
      net:add(backend.SpatialMaxPooling(3,3,2,2))   --50     --  ->  27 x 15
      net:add(convLayer(96,256,5,5,1,1,2,2))        --24     --  ->  27 x 15
      net:add(backend.ReLU(true))
      net:add(backend.SpatialMaxPooling(3,3,2,2))   --20     --  ->  13 x 7
      net:add(convLayer(256,384,3,3,1,1,1,1))       --9     --  ->  13 x 7
      net:add(backend.ReLU(true))
      net:add(convLayer(384,384,3,3,1,1,1,1))       --7     --  ->  13 x 7
      net:add(backend.ReLU(true))
      net:add(convLayer(384,256,3,3,1,1,1,1))       --5     --  ->  13 x 7
      net:add(backend.ReLU(true))
      net:add(backend.SpatialMaxPooling(3,3,2,2))   --3     --  ->  6 x 3
      -- bs x 256 x 6 x 2
      n_conv_features = 256*6*6 -- 256*6*3
      strip_width = 224 -- 128 -- 224
      stride = 100 -- (224-strip_width)/3
    elseif v==3 or v==4 then
      net = nn.Sequential()
      net:add(convLayer(channels,96,11,11,2,2,2,2)) -- 110    -- 224 x 128 ->  55 x 31
      net:add(backend.ReLU(true))
      net:add(backend.SpatialMaxPooling(3,3,2,2))   --50     --  ->  27 x 15
      net:add(convLayer(96,256,5,5,1,1,2,2))        --24     --  ->  27 x 15
      net:add(backend.ReLU(true))
      net:add(backend.SpatialMaxPooling(3,3,2,2))   --20     --  ->  13 x 7
      net:add(convLayer(256,384,3,3,1,1,1,1))       --9     --  ->  13 x 7
      net:add(backend.ReLU(true))
      net:add(convLayer(384,384,3,3,1,1,1,1))       --7     --  ->  13 x 7
      net:add(backend.ReLU(true))
      net:add(convLayer(384,256,3,3,1,1,1,1))       --5     --  ->  13 x 7
      net:add(backend.ReLU(true))
      net:add(backend.SpatialMaxPooling(3,3,2,2))   --3     --  ->  6 x 3
      -- bs x 256 x 6 x 2
      n_conv_features = 256*12*6 -- 256*6*3
      strip_width = 128 -- 128 -- 224
      stride = 48 -- (224-strip_width)/3
    end

    net:add(nn.Dropout(0.5))


    --local classifier = nn.Sequential()
    --classifier:add(nn.View(256*6*6))
    --classifier:add(nn.Dropout(0.5))
    --classifier:add(nn.Linear(256*6*6, 4096))
    --classifier:add(nn.Threshold(0, 1e-6))
    --classifier:add(nn.Dropout(0.5))
    --classifier:add(nn.Linear(4096, 4096))
    --classifier:add(nn.Threshold(0, 1e-6))
    --classifier:add(nn.Linear(4096, nClasses))
    --classifier:add(backend.LogSoftMax())

    -- insert LSTM or Linear layer between cnn and classifier

    local n_linear_features_1 = 4096
    local n_linear_features_2 = 4096
    net:add(nn.View(-1):setNumInputDims(3))
    if recurrent then
      if use_lstm then
        --net:add(nn.Dropout(0.5))
        --   :add(nn.LSTM(n_conv_features, n_linear_features_1))
        --   :add(nn.Threshold(0, 1e-6))
        --   :add(nn.Dropout(0.5))
        --   :add(nn.LSTM(n_linear_features_1, n_linear_features_2))
        --   :add(nn.Threshold(0, 1e-6))

        net:add(nn.LSTM(n_conv_features, n_linear_features_1))
           :add(backend.ReLU(true))

        if v~=4 then
           net:add(nn.LSTM(n_linear_features_1, n_linear_features_2))
              :add(backend.ReLU(true))
        end
      elseif use_gru then
        -- 0.5 dropout
        net:add(nn.GRU(n_conv_features, n_linear_features_2, 9999, 0.5))
              :add(backend.ReLU(true))
      else
        net:add(
          nn.Recurrence(
            nn.Sequential()
              :add(nn.ParallelTable()
                :add(nn.Linear(n_conv_features, n_linear_features_2))
                :add(nn.Linear(n_linear_features_2, n_linear_features_2)))
              :add(nn.CAddTable())
              :add(backend.ReLU()),
            n_linear_features_2,
            1))
      end
    else
      net:add(nn.Linear(n_conv_features, n_linear_features_1))
         :add(backend.ReLU())
         :add(nn.Linear(n_linear_features_1, n_linear_features_2))
         :add(backend.ReLU(true))
    end
    -- classification part
    net:add(nn.Linear(n_linear_features_2, nClasses))
       :add(nn.LogSoftMax())
    -- wrap in a Sequencer
    net = nn.Sequencer(net)
    -- objective function (wrapped in a SequencerCriterion)
    local criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())

    -- hook
    function DataHook(dbinput, dblabel)
      input = {}
      label = {}

      bs = dbinput:size(1)
      channels = dbinput:size(2)
      height = dbinput:size(3)
      width = dbinput:size(4)

      i = 0
      while i < width do
        if stride*i + strip_width <= width then
          input[i+1] = dbinput:sub(1, bs, 1, channels, 1, height, 1+stride*i, 1+stride*i+strip_width-1)
          label[i+1] = dblabel
        end
        i = i+1
      end
      --print(#input, input[#input]:size())

      return input, label
    end

   return net, criterion
end

-- return function that returns network definition
return function(params)
    -- get number of classes from external parameters
    local nclasses = params.nclasses or 1
    -- adjust to number of channels in input images
    local channels = 1
    -- params.inputShape may be nil during visualization
    if params.inputShape then
        channels = params.inputShape[1]
        assert(params.inputShape[2]==256 and params.inputShape[3]==256, 'Network expects 256x256 images')
    end
    model, criterion = createModel(params.ngpus, channels, nclasses)
    return {
        model = model,
        loss = criterion,
        croplen = 224,
        trainBatchSize = 16,
        validationBatchSize = 32,
        dataHook = DataHook,
    }
end

