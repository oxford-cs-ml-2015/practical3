---------------------------------------------------------------------------------------
-- 실습 3 - 로지스틱 회귀로 다른 최적화기를 사용하기 위한 학습
--
-- 실행 방법: th -i practical3.lua
-- 또는:      luajit -i practical3.lua
---------------------------------------------------------------------------------------

require 'torch'
require 'math'
require 'nn'
require 'optim'
require 'gnuplot'
require 'dataset-mnist'

------------------------------------------------------------------------------
-- 초기화 그리고 데이터
------------------------------------------------------------------------------

torch.manualSeed(1)    -- 프로그램이 매번 같게 실행되도록 랜덤 시드를 고정합니다.

-- 할 일(TODO): 이 최적화기 옵션들을 가지고 노십시오. 문서에 설명되어 있는 두 번째 제출 숙제임.
-- 노트: 최적화기 설정들을 저장하는 optimState를 위해 아래를 보십시오.
local opt = {}         -- 이 옵션들은 도처에 사용됩니다.
opt.optimization = 'sgd'
opt.batch_size = 3
opt.train_size = 8000  -- 0 또는 60000으로 설정, 모든 60000개 훈련 데이털르 사용하기 위해
opt.test_size = 0      -- 0은 모든 데이터를 로드함을 의미합니다.
opt.epochs = 2         -- 훈련 데이터 전체를 한 번씩 읽어 처리하는 **근사적** 횟수, (`반복(iterations)` 변수를 위해 아래를 보십시오, 그 변수가 이 값으로 계산됩니다)

-- 노트: 아래 코드는 사용되는 최적화 알고리즘과 그 설정들을 바꿉니다
local optimState       -- 최적화 알고리즘의 설정들과 iteration들 동안의 상태을 가진 루아 테이블 하나를 저장
local optimMethod      -- 그 최적화 루틴에 상응하는 함수 하나를 저장
-- 기억하십시오, 아래 기본값들이 꼭 좋은 것은 아닙니다.
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

mnist.download()       -- 만약 없다면, 데이터세트를 내려받습니다.

-- dataset-mnist.lua를 사용하여 데이터세트를 텐서들로 로드합니다 (데이터의 첫 차원/데이터에 대한 정답(labels)들)
local function load_dataset(train_or_test, count)
    -- 로드
    local data
    if train_or_test == 'train' then
        data = mnist.loadTrainSet(count, {32, 32})
    else
        data = mnist.loadTestSet(count, {32, 32})
    end

    -- 데이터세트 섞기
    local shuffled_indices = torch.randperm(data.data:size(1)):long()
    -- 새 스토리지를 가진 섞인 *복사본* 생성 
    data.data = data.data:index(1, shuffled_indices):squeeze()
    data.labels = data.labels:index(1, shuffled_indices):squeeze()

    -- 할 일(TODO): (선택적) 훈련 예제를 디스플레이하기 위해 주석 해제(UNCOMMENT)하십시오.
    -- 더 자세한 사항은 torch gnuplot 패키지 문서를 보십시오:
    -- https://github.com/torch/gnuplot#plotting-package-manual-with-gnuplot
    --gnuplot.imagesc(data.data[10])

    -- 각 2차원 데이터 포인트를 1차원으로 벡터화합니다.
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

local n_train_data = train.data:size(1) -- 훈련 데이터 개수
local n_inputs = train.data:size(2)     -- 열의 개수 = 입력의 차원들의 개수
local n_outputs = train.labels:max()    -- 가장 높은 레이블 = 부류들의 개수

print(train.labels:max())
print(train.labels:min())

local lin_layer = nn.Linear(n_inputs, n_outputs)
local softmax = nn.LogSoftMax() 
local model = nn.Sequential()
model:add(lin_layer)
model:add(softmax)

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

  -- 우리의 미니배치를 위한 시작 또는 끝 인덱스들을 얻습니다 (이 코드에서 우리는 미니배치를 "배치"라 부를 것입니다)
  --           ------- 
  --          |  ...  |
  --        ^ ---------<- start index = i * batchsize + 1
  --  batch | |       |
  --   size | | batch |       
  --        v |   i   |<- end index (마지막 인덱스도 포함) = start index + batchsize
  --          ---------                                    = (i + 1) * batchsize + 1
  --          |  ...  |                 (마지막 미니배치를 제외하고, 데이터 범위를 넘어서 
  --          --------                   접근하기 못하도록, min()을 취합니다)
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

  -- 순서대로, 이 줄들은 계산합니다:
  -- 1. 각 데이터 포인트를 위한 출력 (로그 확률) 계산
  local batch_outputs = model:forward(batch_inputs)
  -- 2. 이 출력들의 손실 계산, barch_target에 있는 정답과 대조하여 측정된
  local batch_loss = criterion:forward(batch_outputs, batch_targets)
  -- 3. 그 모델의 출력들에 대한 편미분 계산
  local dloss_doutput = criterion:backward(batch_outputs, batch_targets) 
  -- 4. 가중치들을 갱신하기 위해 기울기들을 사용, 우리는 다음 주에 이 단계를 더 깊게 이해할 것입니다.
  model:backward(batch_inputs, dloss_doutput)

  -- optim은 우리에게 다음을 리턴하길 기대합니다
  --     손실, (우리가 최적화하고 있는 가중치들에 대한 손실의 기울기)
  return batch_loss, gradParameters
end
  
------------------------------------------------------------------------
-- 최적화: 첫 번째 제출 항목
------------------------------------------------------------------------
local losses = {}          -- 각 iteration/미니배치를 위한 훈련 손실들
local epochs = opt.epochs  -- 에포크 횟수
local iterations = epochs * math.ceil(n_train_data / opt.batch_size) -- 처리할 미니배치들의 정수 숫자
-- (노트: 훈련 데이터의 수는 배치 크기로 나눌 수 없을 수도 있습니다, 그래서 우리는 올림 합니다)

-- 각 iteration에서, 우리는:
--    1. 그 최적화 루틴을 호출합니다, 그 루틴은
--      a. feval(parameters)을 호출합니다, 그 feval은
--          i. 다음 미니배치를 집습니다
--         ii. 그 미니배치에서 평가된 파라미터들에 대한 손실의 기울기와 손실 값을 리턴합니다.
--      b. 손실을 줄이기 위해, 그 최적화 루틴은 이 기울기를 그 파라미터를 조절하는 데 사용합니다.
--    2. 그리고 우리는 그 손실을 한 테이블(리스트)에 덧붙이고, 그것을 출력합니다.
for i = 1, iterations do
  -- optimMethod는 함수를 저장하는 변수입니다, 그 변수는 optim.sgd 또는 optim.adagrad 또는 ...입니다.
  -- 이 함수들이 무엇을 하고 리턴하는지에 대한 더 자세한 정보는 다음 문서를 보십시오:
  --   https://github.com/torch/optim
  -- it returns (new_parameters, table), where table[0] is the value of the function being optimized
  -- and we can ignore new_parameters because `parameters` is updated in-place every time we call 
  -- the optim module's function. It uses optimState to hide away its bookkeeping that it needs to do
  -- between iterations.
  local _, minibatch_loss = optimMethod(feval, parameters, optimState)

  -- Our loss function is cross-entropy, divided by the number of data points,
  -- therefore the units (units in the physics sense) of the loss is "loss per data sample".
  -- Since we evaluate the loss on a different minibatch each time, the loss will sometimes 
  -- fluctuate upwards slightly (i.e. the loss estimate is noisy).
  if i % 10 == 0 then -- don't print *every* iteration, this is enough to get the gist
      print(string.format("minibatches processed: %6s, loss = %6.6f", i, minibatch_loss[1]))
  end
  -- TIP: use this same idea of not saving the test loss in every iteration if you want to increase speed.
  -- Then you can get, 10 (for example) times fewer values than the training loss. If you do this,
  -- you just have to be careful to give the correct x-values to the plotting function, rather than
  -- Tensor{1,2,...,#losses}. HINT: look up the torch.linspace function, and note that torch.range(1, #losses)
  -- is the same as torch.linspace(1, #losses, #losses).

  losses[#losses + 1] = minibatch_loss[1] -- append the new loss
end

-- TODO: for the first handin item, evaluate test loss above, and add to the plot below
--       see TIP/HINT above if you want to make the optimization loop faster

-- Turn table of losses into a torch Tensor, and plot it
gnuplot.plot({
  torch.range(1, #losses),        -- x-coordinates for data to plot, creates a tensor holding {1,2,3,...,#losses}
  torch.Tensor(losses),           -- y-coordinates (the training losses)
  '-'})

------------------------------------------------------------------------------
-- TESTING THE LEARNED MODEL: 2ND HANDIN ITEM
------------------------------------------------------------------------------

local logProbs = model:forward(test.data)
local classProbabilities = torch.exp(logProbs)
local _, classPredictions = torch.max(classProbabilities, 2)
-- classPredictions holds predicted classes from 1-10

-- TODO: compute test classification error here for the second handin item

