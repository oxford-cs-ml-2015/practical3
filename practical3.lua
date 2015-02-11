---------------------------------------------------------------------------------------
-- Practical 3 - Learning to use different optimizers with logistic regression
---------------------------------------------------------------------------------------

require 'torch'
require 'math'
require 'nn'
require 'optim'
require 'gnuplot'
require 'dataset-mnist'

------------------------------------------------------------------------------
-- INITIALIZATION AND DATA
------------------------------------------------------------------------------

torch.manualSeed(1)    -- fix random seed so program runs the same every time

-- NOTE: see below for optimState, storing optimiser settings
local opt = {}         -- these options are used throughout
opt.optimization = 'sgd'
opt.batch_size = 5
opt.train_size = 8000  -- set to 0 or 60000 to use all 60000 training data
opt.test_size = 0      -- 0 means load all data

local optimState
local optimMethod
if opt.optimization == 'lbfgs' then
  optimState = {
    learningRate = 1e-1,
    maxIter = 2,
    nCorrection = 10
  }
  optimMethod = optim.lbfgs
elseif opt.optimization == 'sgd' then
  optimState = {
    learningRate = 1e-1,
    weightDecay = 0,
    momentum = 0,
    learningRateDecay = 1e-7
  }
  optimMethod = optim.sgd
elseif opt.optimization == 'adagrad' then
  optimState = {
    learningRate = 1e-1,
  }
  optimMethod = optim.adagrad
else
  error('Unknown optimizer')
end

mnist.download()       -- download dataset if not already there

-- load dataset using dataset-mnist.lua into tensors (first dim of data/labels ranges over data)
local function load_dataset(train_or_test, count)
    -- load
    local data
    if train_or_test == 'train' then
        data = mnist.loadTrainSet(count, {32, 32})
    else
        data = mnist.loadTestSet(count, {32, 32})
    end

    -- shuffle the dataset
    local shuffled_indices = torch.randperm(data.data:size(1)):long()
    -- creates a shuffled *copy*, with a new storage
    data.data = data.data:index(1, shuffled_indices):squeeze()
    data.labels = data.labels:index(1, shuffled_indices):squeeze()

    -- UNCOMMENT to display a training example
    -- for more, see torch gnuplot package documentation:
    -- https://github.com/torch/gnuplot#plotting-package-manual-with-gnuplot
    --gnuplot.imagesc(data.data[20])

    -- vectorize each 2D data point into 1D
    data.data = data.data:reshape(data.data:size(1), 32*32)

    print('--------------------------------')
    print(' loaded dataset "' .. train_or_test .. '"')
    print('inputs', data.data:size())
    print('targets', data.labels:size())
    print('--------------------------------')

    return data
end

local train = load_dataset('train', opt.train_size)
local test = load_dataset('test', opt.test_size)

------------------------------------------------------------------------------
-- MODEL
------------------------------------------------------------------------------

local n_data = train.data:size(1)       -- number of training data
local n_inputs = train.data:size(2)     -- number of cols = number of dims of input
local n_outputs = train.labels:max()    -- highest label = # of classes

print(train.labels:max())
print(train.labels:min())

local lin_layer = nn.Linear(n_inputs, n_outputs)
local softmax = nn.LogSoftMax() 
model = nn.Sequential()
model:add(lin_layer)
model:add(softmax)

------------------------------------------------------------------------------
-- LOSS FUNCTION
------------------------------------------------------------------------------

local criterion = nn.ClassNLLCriterion()

------------------------------------------------------------------------------
-- TRAINING
------------------------------------------------------------------------------

local parameters, gradParameters = model:getParameters()

------------------------------------------------------------------------
-- Define closure with mini-batches 
------------------------------------------------------------------------

local counter = 0
local feval = function(x)
  if x ~= parameters then
    parameters:copy(x)
  end

  -- get start/end indices for our minibatch (here called "batch")
  local start_index = counter * opt.batch_size + 1
  local end_index = math.min(n_data, (counter + 1) * opt.batch_size)
  if end_index == n_data then
    counter = 0
  end
  counter = counter + 1

  local batch_inputs = train.data[{{start_index, end_index}, {}}]
  local batch_targets = train.labels[{{start_index, end_index}}]
  gradParameters:zero()

  -- In order, these lines compute:
  -- 1. compute outputs (log probabilities) for each data point
  local batch_outputs = model:forward(batch_inputs)
  -- 2. compute the loss of these outputs, measured against the true labels in batch_target
  local batch_loss = criterion:forward(batch_outputs, batch_targets)
  -- 3. compute the derivative of the loss wrt the outputs of the model
  local dloss_doutput = criterion:backward(batch_outputs, batch_targets) 
  -- 4. use gradients to update weights, we'll understand this step more next week
  model:backward(batch_inputs, dloss_doutput)

  -- optim expects us to return
  --     loss, (gradient of loss with respect to the weights that we're optimizing)
  return batch_loss, gradParameters
end
  
------------------------------------------------------------------------
-- Optimize 
------------------------------------------------------------------------
local epochs = 10  -- number of full passes over all the training data
local losses = {}
local epoch_loss = 0  
for epoch = 1, epochs do
  epoch_loss = 0

  -- loop over however many minibatches there are (by convention _ means a variable we ignore)
  for _ = 1, train.data:size(1), opt.batch_size do
    -- remember optimMethod is something like optim.sgd/optim.adagrad, see top of this file
    local __, minibatch_loss = optimMethod(feval, parameters, optimState)
    epoch_loss = epoch_loss + minibatch_loss[1]
  end

  losses[#losses+1] = epoch_loss
  print("epoch = " .. epoch .. ', full epoch loss = ' .. epoch_loss)
end

-- turn list of losses into a Tensor, and plot it
gnuplot.plot({
  torch.range(1, #losses),        -- x-coordinates for data to plot
  torch.Tensor(losses),           -- y-coordinates
  '-'})

------------------------------------------------------------------------------
-- TESTING THE LEARNED MODEL
------------------------------------------------------------------------------

local logProbs = model:forward(test.data)
local classProbabilities = torch.exp(logProbs)
local _, classPredictions = torch.max(classProbabilities,2)

-- TODO: compute test error here


