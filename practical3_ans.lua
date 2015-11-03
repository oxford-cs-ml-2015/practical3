---------------------------------------------------------------------------------------
-- 실습 3 - 로지스틱 회귀를 사용한 학습, 여러가지 최적화기를 사용해보기 위한.
--
-- 실행 방법: th -i practical3.lua
-- 또는:     luajit -i practical3.lua
---------------------------------------------------------------------------------------

require('mobdebug').start()
require 'torch'
require 'math'
require 'nn'
require 'optim'
require 'gnuplot'
require 'dataset-mnist'

------------------------------------------------------------------------------
-- 초기화 그리고 데이터
------------------------------------------------------------------------------

torch.manualSeed(1)

local opt = {}
opt.optimization = 'sgd'
opt.batch_size = 1000
opt.train_size = 60000
opt.test_size = 0
opt.epochs = 1

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

mnist.download()

local function load_dataset(train_or_test, count)
    local data
    if train_or_test == 'train' then
        data = mnist.loadTrainSet(count, {32, 32})
    else
        data = mnist.loadTestSet(count, {32, 32})
    end

    local shuffled_indices = torch.randperm(data.data:size(1)):long()
    data.data = data.data:index(1, shuffled_indices):squeeze()
    data.labels = data.labels:index(1, shuffled_indices):squeeze()

    --gnuplot.imagesc(data.data[10])

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
-- 모델
------------------------------------------------------------------------------

local n_train_data = train.data:size(1)
local n_inputs = train.data:size(2)
local n_outputs = train.labels:max()

print(train.labels:max())
print(train.labels:min())

local lin_layer = nn.Linear(n_inputs, n_outputs)
local softmax = nn.LogSoftMax() 
local model = nn.Sequential()
model:add(lin_layer)
model:add(softmax)

local n_test_data= test.data:size(1)

------------------------------------------------------------------------------
-- 손실 함수
------------------------------------------------------------------------------

local criterion = nn.ClassNLLCriterion()

------------------------------------------------------------------------------
-- 훈련
------------------------------------------------------------------------------

local parameters, gradParameters = model:getParameters()

------------------------------------------------------------------------
-- 미니 배치를 가진 클로저(함수 안에 정의된 함수) 정의
------------------------------------------------------------------------

local counter = 0
local feval = function(x)
  if x ~= parameters then
    parameters:copy(x)
  end

  local start_index = counter * opt.batch_size + 1
  local end_index = math.min(n_train_data, (counter + 1) * opt.batch_size + 1)
  if end_index == n_train_data then
    counter = 0
  else
    counter = counter + 1
  end

  local batch_inputs = train.data[{{start_index, end_index}, {}}]
  local batch_targets = train.labels[{{start_index, end_index}}]
  gradParameters:zero()

  local batch_outputs = model:forward(batch_inputs)
  local batch_loss = criterion:forward(batch_outputs, batch_targets)
  local dloss_doutput = criterion:backward(batch_outputs, batch_targets) 
  model:backward(batch_inputs, dloss_doutput)

  return batch_loss, gradParameters
end
  
------------------------------------------------------------------------
-- 최적화: 첫 번째 제출 항목
------------------------------------------------------------------------
local losses = {}
local epochs = opt.epochs
local iterations = epochs * math.ceil(n_train_data / opt.batch_size)
for i = 1, iterations do
  local _, minibatch_loss = optimMethod(feval, parameters, optimState)
  if i % 10 == 0 then
      print(string.format("(trn) minibatches processed: %6s, loss = %6.6f", i, minibatch_loss[1])) 
  end
  losses[#losses + 1] = minibatch_loss[1]
end

-- 할 일(TODO): 첫 번째 제출 항목을 위해, 위의 시험 손실을 평가(계산)하십시오.
--              그리고 아래 그림에 추가하십시오.
local losses_test = {}
local epochs = opt.epochs
local iterations = 1 * math.ceil(n_test_data / opt.batch_size)
for i = 1, iterations do
  local _, minibatch_loss = optimMethod(feval, parameters, optimState)
  print(string.format("(tst) minibatches processed: %6s, loss = %6.6f", i, minibatch_loss[1]))
  losses_test[#losses_test + 1] = minibatch_loss[1] 
end


------------------------------------------------------------------------------
-- 그 학습된 모델을 시험: 두 번째 제출 항목
------------------------------------------------------------------------------

local logProbs = model:forward(test.data) -- 10000개 데이터의 로그 확률들
local classProbabilities = torch.exp(logProbs) -- 그냥 확률들
local _, classPredictions = torch.max(classProbabilities, 2) -- 각 행에서 가장 큰 확률이 있는 인덱스들 추출 (예측).

-- 할 일(TODO): 두 번째 제출 항목을 위한 시험 분류 오차를 계산하십시오.
n_correct_lab= 0
n_testData= #(test.labels)
for i = 1, n_testData[1] do
  if classPredictions[i][1] == test.labels[i] then
    n_correct_lab= n_correct_lab + 1
  end
end

-- 정답률 계산
print('(tst) # of correct examples = ' .. tostring(n_correct_lab) )
correct_rate= n_correct_lab/n_testData[1]*100
print('Correct Rate = ' .. tostring(correct_rate) .. '%')

x= torch.range(1, #losses)
x2= torch.range(1, #losses_test)
gnuplot.plot({'Trn', x, torch.Tensor(losses), '-'}, 
             {'Tst', x2, torch.Tensor(losses_test), '-'})
gnuplot.xlabel('# of minibatches')
gnuplot.ylabel('loss')
gnuplot.title('opt : ' ..
              'opt=' .. opt.optimization .. ', ' ..
              'batch_siz=' .. tostring(opt.batch_size) .. ', ' ..
              'trn_siz=' .. tostring(opt.train_size) .. ', ' ..
              'tst_siz=' .. tostring(opt.test_size) .. ', ' ..
              'epochs=' .. tostring(opt.epochs) .. ', ' ..
              'Corr=' .. tostring(correct_rate) .. '%'
             )

