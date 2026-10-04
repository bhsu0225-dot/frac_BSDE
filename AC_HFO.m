% ============================================================
% High-dimensional semilinear PDE on cube
% Phi(x) = (prod_j sin(k*pi*x_j), |x|^2), kappa = 2.
%
% Domain: (-L,L)^d, zero boundary/exterior payoff.
% Manufactured exact solution:
%   u(t,x) = cos(lambda_t * t) * prod_{j=1}^d sin(k*pi*x_j)
%
% PDE:
%   u_t + Delta u + f(u) = q(t,x)
% with
%   f(u) = cos(u) + exp(sin(u^2))
% and
%   q = u_t + Delta u + f(u).
% Thus the BSDE source is f(y)-q(t,x).
% ============================================================

clc; clear;

%% ---------------- User parameters ----------------
d   = 11;
L   = 1.0;
T   = 0.5;
t0  = 0.0;
N   = 64000;
M   = 2000;
K   = 201;

k_list   = [3, 7, 11, 15];
lambda_t = 1;

% ----- 2D binning CE parameters -----
NBIN1    = 500;
NBIN2    = 500;
SMOOTHIT = 2;

% ----- Solver, parallel computation, and output -----
SCALAR_TOL   = 1e-12;
NEWTON_MAXIT = 50;
USE_PARALLEL = true;
NUM_WORKERS  = 10;       % Limit concurrent path histories to control memory.
BASE_SEED    = 2026;
SAVE_FIGURES = true;
output_dir   = fullfile(fileparts(mfilename('fullpath')), 'AC_HFO_results');

assert(d >= 2 && d == floor(d), 'd must be an integer >= 2.');
assert(T > t0 && L > 0, 'Require T > t0 and L > 0.');
assert(all([N,M,K] >= 1) && all([N,M,K] == floor([N,M,K])), ...
    'N, M, and K must be positive integers.');
assert(all(k_list > 0) && all(abs(k_list*L-round(k_list*L)) < 1e-12), ...
    'Require k*L to be an integer for the manufactured zero boundary data.');
assert(all([NBIN1,NBIN2] >= 2) && all([NBIN1,NBIN2] == floor([NBIN1,NBIN2])) ...
    && SMOOTHIT >= 0 && SMOOTHIT == floor(SMOOTHIT), 'Invalid feature grid.');
assert(SCALAR_TOL > 0 && NEWTON_MAXIT >= 1 && NEWTON_MAXIT == floor(NEWTON_MAXIT), ...
    'Invalid scalar-solver parameters.');
assert(NUM_WORKERS >= 1 && NUM_WORKERS == floor(NUM_WORKERS), 'Invalid worker count.');
dt = (T - t0) / N;
rng(BASE_SEED, 'twister');

nWorkers = 0;
if USE_PARALLEL
    assert(license('test','Distrib_Computing_Toolbox'), ...
        'Parallel Computing Toolbox is required. Set USE_PARALLEL=false for serial execution.');
    pool = gcp('nocreate');
    if isempty(pool)
        pool = parpool('local', NUM_WORKERS);
    end
    nWorkers = min(NUM_WORKERS, pool.NumWorkers);
end
if SAVE_FIGURES && ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

fprintf('d=%d, cube=[-%.1f,%.1f]^d, T=%.3f, N=%d, M=%d, K=%d, lambda=%.2f\n', ...
    d, L, L, T, N, M, K, lambda_t);
fprintf('2D binning CE: NBIN1=%d, NBIN2=%d, SMOOTHIT=%d\n', ...
    NBIN1, NBIN2, SMOOTHIT);
fprintf('Parallel workers: %d | Newton residual tolerance: %.1e\n', nWorkers, SCALAR_TOL);

%% ---------------- Loop over k ----------------
for kk = 1:numel(k_list)
    kfreq = k_list(kk);

    % Choose diagonal points so sin(k*pi*a)=±1 when possible
    a  = make_a_grid(K, kfreq, 0.98*L);
    X0 = repmat(a(:), 1, d);

    fprintf('\n==================== k = %d ====================\n', kfreq);

    results    = zeros(K, 4);  % [u_exact, u_num, abs_err, rel_err]
    u_num_list = zeros(K, 1);
    u_ex_list  = zeros(K, 1);
    % ordered print queue
    orderedPrint([], K);
    progressQueue = [];
    if USE_PARALLEL
        progressQueue = parallel.pool.DataQueue;
        afterEach(progressQueue, @(data) orderedPrint(data, K));
    end

    tic;
    parfor (kpt = 1:K, nWorkers)
        rng(BASE_SEED + (kk-1)*K + kpt, 'twister');
        x0 = X0(kpt, :);

        u_hat = fk_estimate( ...
            x0, t0, T, L, d, N, M, dt, ...
            NBIN1, NBIN2, SMOOTHIT, kfreq, lambda_t, ...
            SCALAR_TOL, NEWTON_MAXIT);

        u_ex   = u_exact(t0, x0, kfreq, lambda_t);
        abs_er = abs(u_hat - u_ex);
        rel_er = abs_er / max(1e-15, abs(u_ex));

        results(kpt,:)   = [u_ex, u_hat, abs_er, rel_er];
        u_num_list(kpt)  = u_hat;
        u_ex_list(kpt)   = u_ex;
        data = [kpt, a(kpt), u_ex, u_hat];
        if USE_PARALLEL
            send(progressQueue, data);
        else
            orderedPrint(data, K);
        end
    end
    t_elapsed = toc;
    drawnow;

    fprintf('\n=== SUMMARY (k=%d) ===\n', kfreq);
    disp(array2table(results, 'VariableNames', {'u_exact','u_num','abs_err','rel_err'}));
    fprintf('Mean abs err = %.3e | Median abs err = %.3e | Mean rel err = %.3e\n', ...
        mean(results(:,3)), median(results(:,3)), mean(results(:,4)));
    fprintf('Total time: %.2fs\n', t_elapsed);

    %% ---------------- Plot ----------------
    fig = figure('Name', sprintf('Exact_vs_Numerical_k%d', kfreq), 'Color','w');

    plot(a, u_ex_list, 'r-', ...
        'LineWidth', 1.8, ...
        'MarkerSize', 6, ...
        'MarkerFaceColor','r');
    hold on;

    plot(a, u_num_list, 'b-.', ...
        'LineWidth', 1.8, ...
        'MarkerSize', 6, ...
        'MarkerFaceColor','w');

    grid on;
    box on;

    xlabel('$\bf{ a \quad (x_0=(a,\dots,a))}$','Interpreter','latex');
    ylabel('$\bf{ u(t_0,x_0)} $','Interpreter','latex');
    title('Exact vs Numerical Solution', 'Interpreter', 'latex');

    legend({'Exact','Numerical'}, ...
        'Location','best', ...
        'Interpreter','latex');

    set(gca, 'FontSize', 16, 'LineWidth', 1.0);

    %% ----- save figure -----
    if SAVE_FIGURES
        figName = fullfile(output_dir, sprintf('HFO%d_%d_%d_%d', M, N, d, kfreq));
        set(fig, 'PaperPositionMode', 'auto');
        savefig(fig, [figName '.fig']);
        print(fig, [figName '.eps'], '-depsc2', '-painters');
        print(fig, [figName '.pdf'], '-dpdf', '-bestfit', '-painters');
        fprintf('Figure saved as "%s.fig", ".eps", and ".pdf"\n', figName);
    end
end

%% =======================================================================
function u_hat = fk_estimate( ...
    x0, t0, T, L, d, N, M, dt, ...
    NBIN1, NBIN2, SMOOTHIT, kfreq, lambda_t, scalar_tol, itMax)

assert(numel(x0) == d && all(abs(x0) < L), 'The initial point must be interior.');
assert(abs(N*dt-(T-t0)) <= 1e-12*max(1,T-t0), 'Inconsistent time grid.');
X        = repmat(x0, M, 1);
features = cell(N,1);
alive    = cell(N+1,1);
alive{1} = true(M,1);
payoff = zeros(M,1);
Nf = 0;

for k = 1:N
    al = alive{k};
    Xk = X(al,:);
    features{k} = [prod_sin_kpi(Xk, kfreq), sum(Xk.^2,2)];

    % Exact Brownian increment for alpha=2 and generator Delta.
    Xnew = Xk + sqrt(2*dt)*randn(nnz(al),d);
    X(al,:) = Xnew;                     % Retain the exterior candidate.
    alive{k+1} = al;
    alive{k+1}(al) = all(abs(Xnew) < L,2);
    Nf = k;
    if ~any(alive{k+1}), break; end
end

% terminal payoff for survivors
idx_T = alive{Nf+1};
if any(idx_T)
    payoff(idx_T) = phi_fun(T, X(idx_T,:), kfreq, lambda_t);
end

% backward recursion
Y = payoff;
for k = Nf:-1:1
    Ik = find(alive{k});
    tk = t0 + (k-1)*dt;
    p1k = features{k}(:,1);
    r2k = features{k}(:,2);
    target = Y(Ik) - dt*q_fun_mms_general(tk, p1k, d, kfreq, lambda_t);
    Hk = ce_bin2d_sumcount(p1k,r2k,target,NBIN1,NBIN2,SMOOTHIT);

    % W*1=1 makes Hk+dt*f(y) exactly W[Ynext+dt*(f(y)-q)].
    % In particular, the spatial source q remains inside the fitted target.
    Y(Ik) = solve_implicit(Hk,dt, ...
        @(y) cos(y)+exp(sin(y.^2)), ...
        @(y) -sin(y)+2*y.*cos(y.^2).*exp(sin(y.^2)), scalar_tol, itMax);
end

u_hat = mean(Y);
end

%% ======================= Exact / Terminal ==============================
function val = u_exact(t, X, kfreq, lambda_t)
P = prod_sin_kpi(X, kfreq);
val = cos(lambda_t * t) .* P;
end

function val = phi_fun(T, X, kfreq, lambda_t)
val = u_exact(T, X, kfreq, lambda_t);
end

function P = prod_sin_kpi(X, kfreq)
S = sin(kfreq*pi*X);
absS = abs(S);
row0 = any(absS < 1e-300, 2);
P = zeros(size(X,1),1);

idx = ~row0;
if any(idx)
    logabs = sum(log(absS(idx,:)), 2);
    sgn    = prod(sign(S(idx,:)), 2);
    P(idx) = sgn .* exp(logabs);
end
end

%% ======================= Source term q for MMS =========================
function val = q_fun_mms_general(t, P, d, kfreq, lambda_t)
% P is the first feature, prod_j sin(k*pi*x_j), at the current position.
% Paper convention: q = u_t + Delta u + cos(u) + exp(sin(u^2)).
u = cos(lambda_t * t) .* P;

du_dt   = -lambda_t .* sin(lambda_t * t) .* P;
Delta_u = -d * (kfreq*pi)^2 .* u;
f       = cos(u) + exp(sin(u.^2));

val = du_dt + Delta_u + f;
end

%% ======================= Scalar Newton solve =============================
function y = solve_implicit(H, step_dt, f, df, tol, itMax)
H = H(:);
assert(isscalar(step_dt) && isfinite(step_dt) && step_dt >= 0, 'Invalid time step.');
y = H;
for it = 0:itMax
    residual = y - H - step_dt*f(y);
    if any(~isfinite(y)) || any(~isfinite(residual))
        error('IBSDE:ScalarSolve', 'Nonfinite Newton iterate. Check the time step and source.');
    end
    if norm(residual,inf) <= tol
        return;
    end
    if it == itMax, break; end
    jacobian = 1-step_dt*df(y);
    if any(~isfinite(jacobian)) || any(abs(jacobian) < 1e-12)
        error('IBSDE:ScalarSolve', 'Singular Newton derivative. Reduce the time step.');
    end
    y = y-residual./jacobian;
end
error('IBSDE:ScalarSolve', 'Newton residual exceeds tolerance. Reduce the time step or increase NEWTON_MAXIT.');
end


function H = ce_bin2d_sumcount(p1, r2, y, NBIN1, NBIN2, smooth_it)
% Apply one normalized target/count smoother to every column of y.
p1 = p1(:);
r2 = r2(:);
m = numel(p1);
if isvector(y) && numel(y) == m
    y = y(:);
end
assert(numel(r2) == m && size(y,1) == m, 'Incompatible particle arrays.');
assert(all(isfinite(p1)) && all(isfinite(r2)) && all(isfinite(y(:))), ...
    'Nonfinite feature or target.');
if m == 0
    H = zeros(0,size(y,2));
    return;
end
if m == 1
    H = y;
    return;
end
if all(p1 == p1(1)) && all(r2 == r2(1))
    H = repmat(mean(y,1),m,1);
    return;
end
p1lo = min(p1); p1hi = max(p1);
r2lo = min(r2); r2hi = max(r2);
pad1 = 1e-12 + 0.02*(p1hi-p1lo);
pad2 = 1e-12 + 0.02*(r2hi-r2lo);
p1lo = p1lo-pad1; p1hi = p1hi+pad1;
r2lo = r2lo-pad2; r2hi = r2hi+pad2;
i1 = floor((p1-p1lo)/(p1hi-p1lo)*(NBIN1-1))+1;
i2 = floor((r2-r2lo)/(r2hi-r2lo)*(NBIN2-1))+1;
i1 = max(1,min(NBIN1,i1));
i2 = max(1,min(NBIN2,i2));
lin = i1+(i2-1)*NBIN1;
cnt = accumarray(lin,1,[NBIN1*NBIN2,1],@sum,0);
C = reshape(cnt,[NBIN1,NBIN2]);
ker = [1;2;1]/4;
for it = 1:smooth_it
    C = conv2(conv2(C,ker,'same'),ker','same');
end
denom = C(lin);
assert(all(denom > 0), 'An occupied query must have positive smoothed count.');
H = zeros(m,size(y,2));
for col = 1:size(y,2)
    S = reshape(accumarray(lin,y(:,col),[NBIN1*NBIN2,1],@sum,0),[NBIN1,NBIN2]);
    for it = 1:smooth_it
        S = conv2(conv2(S,ker,'same'),ker','same');
    end
    H(:,col) = S(lin)./denom;
end
end


function a = make_a_grid(K, kfreq, margin)
% Choose a so sin(k*pi*a)=±1 and |a|<=margin:
% a_n = (2n+1)/(2k)
if nargin < 3, margin = 0.98; end

n_min = ceil((-2*kfreq*margin - 1)/2);
n_max = floor(( 2*kfreq*margin - 1)/2);
n_all = n_min:n_max;

if numel(n_all) >= K
    idx = round(linspace(1, numel(n_all), K));
    nvec = n_all(idx);
    a = (2*nvec + 1) / (2*kfreq);
else
    a = linspace(-margin, margin, K);
end
a = a(:);
end

%% =================== ordered print callback ============================
function orderedPrint(data, K)
% data = [point index, diagonal coordinate, exact value, computed value].
persistent next_printed results_buffer
if isempty(data)
    next_printed = 1;
    results_buffer = cell(K,1);
    return;
end
results_buffer{data(1)} = data;
while next_printed <= K && ~isempty(results_buffer{next_printed})
    row = results_buffer{next_printed};
    abs_er = abs(row(4)-row(3));
    rel_er = abs_er/max(1e-15,abs(row(3)));
    fprintf('pt %3d/%3d | a=% .6f | exact=%.8e | num=%.8e | abs=%.2e | rel=%.2e\n', ...
        row(1), K, row(2), row(3), row(4), abs_er, rel_er);
    results_buffer{next_printed} = [];
    next_printed = next_printed+1;
end
end
