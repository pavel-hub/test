function out = trustRegionSolve(xStart,Delta,eta,stepLngthBnd,F_obj,F_obj_grad,opts)
% Initiate logs
dists = [];
gNorms = [];
nGradEvals = 0;
% Params
xOpt = opts.xOpt;
s = opts.sVals;
b = opts.b;
s0 = opts.s0;
bvecs = opts.bvecs;
Nmax = opts.Nmax;
J_D = opts.JD;
% initial gradient
df0 = F_obj_grad(xStart,s,s0,b,bvecs');
xk = xStart;
tic
for it = 1:Nmax
    g = F_obj_grad(xk,s,s0,b,bvecs');
    nGradEvals = nGradEvals +1;
    gNorms(it) = norm(g);
    % find hessian approximation
    J_xk = J_D(xk,bvecs',b);
    B = J_xk'*J_xk;
    if norm(g) <= 1.0E-8*norm(df0)
        % converged!
        break;
    end
    p = solve_trust(g,B,Delta);
    %  check the actual reduction
    rho = (F_obj(xk,s,s0,b,bvecs') - F_obj(xk + p,s,s0,b,bvecs')) / (-g'*p - p'*B*p/2);
    % update the trust-region radius
    if rho < 0.25
        Delta = 0.25*Delta;
    elseif rho > 0.75 && Delta-norm(p) >= -1.0E-06*Delta
        Delta = min(2*Delta, stepLngthBnd);
    end
    % accept the step, if the ratio is OK
    if rho > eta
        xk = xk+p;
    end
    dists(it) = norm(xOpt-xk);
end
elapsedTotalTime = toc;
out = struct();
out.nIterations = it;
out.dists = dists;
out.elapsedTotalTime = elapsedTotalTime;
out.Sol = xk;
out.gNorms = gNorms;
out.nGradEvals = nGradEvals;
end