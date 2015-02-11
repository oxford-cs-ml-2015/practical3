require 'torch'
require 'optim'


--------------------------------------------------------------------------------
-- Simple example of minimizing a function,
--    f(x) = 1/2 x^2 + x sin(x)
-- with derivative
--    f'(x) = x + x cos(x) + sin(x).
--
-- Here's a plot:
-- http://www.wolframalpha.com/input/?i=plot+0.5*x%5E2+%2B+x+sin%28x%29+from+-2+to+4
--
--------------------------------------------------------------------------------

-- preallocate space for gradient, which is 1-dim and 1 element
-- (since f is univariate)
local grad = torch.Tensor{0}

-- return function's value and the gradient, at a given point x_vec
local function feval(x_vec)
    -- note: x_vec is a Tensor of 1-dim and size 1, so 
    -- we get its one and only element:
    local x = x_vec[1]

    -- compute and return func val (scalar), and gradient (Tensor)
    f = 0.5*x^2 + x*torch.sin(x)
    grad[1] = x + torch.sin(x) + x*torch.cos(x)
    return f, grad
end

-- where to start the algorithm (usually random, but here we won't since it's a demo)
-- NOTE: try a few starting points, using the plot for pointers
local x = torch.Tensor{5}

-- optim functions use this table for bookkeeping and for reading options
local state = { learningRate = 1e-2 }

-- stop when the gradient is close to 0, or after many iterations
local iter = 0
while true do
    -- optim has multiple functions, such as adagrad, sgd, lbfgs, and others
    -- see documentation for more details
    optim.adagrad(feval, x, state)

    -- gradient norm is SOMETIMES a good measure of how close we are to the optimum, but often not.
    -- the issue is that we'd stop at points like x=0 for x^3
    if grad:norm() < 0.005 or iter > 50000 then 
        break 
    end
    iter = iter + 1
end

print(string.format("%.6f", x[1]))


