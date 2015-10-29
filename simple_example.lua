require 'torch'
require 'optim'


--------------------------------------------------------------------------------
-- 한 함수를 최소화하는 간단한 예제,
--    f(x) = 1/2 x^2 + x sin(x)
-- 그 함수의 미분
--    f'(x) = x + x cos(x) + sin(x).
--
-- 그 함수의 그래프:
-- http://www.wolframalpha.com/input/?i=plot+0.5*x%5E2+%2B+x+sin%28x%29+from+-2+to+4
--
--------------------------------------------------------------------------------

-- 미분값(gradient)을 위한 공간을 미리 할당, 1차원 요소 1개
-- (f의 변수가 하나이므로)
local grad = torch.Tensor{0}

-- 한 주어진 포인트 x_vec에서, 함수의 값과 미분값을 리턴합니다.
local function feval(x_vec)
    -- 노트: x_vec은 1차원 크기 1인 텐서입니다, 그래서
    -- 우리는 오직 그것의 요소 하나를 얻습니다:
    local x = x_vec[1]

    -- 함수 값(스칼라)과 미분값(텐서)을 계산하여 리턴합니다.
    f = 0.5*x^2 + x*torch.sin(x)
    grad[1] = x + torch.sin(x) + x*torch.cos(x)
    return f, grad
end

-- 어디에서 알고리즘이 시작할지 (보통은 랜덤이지만, 이것은 데모이므로 우리는 그렇게 하지 않을 것입니다)
-- 노트: 점들을 위한 플롯(plot)을 사용하여, 몇 개의 다른 시작 점들로 시도해보십시오.
local x = torch.Tensor{5}

-- optim 함수들은 기록 저장과 읽기 옵션들을 위해 이 테이블을 사용합니다.
local state = { learningRate = 1e-2 }

-- 반복 횟수가 너무 많아지거나 미분값이 0에 가까워지면 멈춥니다.
local iter = 0
while true do
    -- optim은 adagrad, sgd, lbfgs 등과 같은 여러 함수들을 가집니다.
    -- 더 자세한 사항은 문서를 보십시오.
    optim.adagrad(feval, x, state)

    -- gradient norm은 때때로 우리가 얼마나 최적에 가까운지에 대한 좋은 측정입니다. 
    -- 그러나 종종 그렇지 않기도 합니다. 이 이슈는 x^3을 위해 x=0 같은 지점들에서 멈출지와 같은 것입니다.
    if grad:norm() < 0.005 or iter > 50000 then 
        break 
    end
    iter = iter + 1
end

print(string.format("%.6f", x[1]))


