% ============================================================
% High-dimensional semilinear PDE on cube
% 2D-binning conditional expectation on (x1*x2, |x|^2)
%
% Domain: [-L,L]^d, Dirichlet boundary u=0 on boundary.
% Manufactured exact solution:
%   u(t,x) = cos(lambda_t * t) * prod_{j=1}^d sin(k*pi*x_j)
%
% PDE:
%   u_t + Delta u + f(u) + q(t,x) = 0
% with
%   f(u) = cos(u) + exp(sin(u^2))
% and
%   q = -(u_t + Delta u + f(u)).
% ============================================================

clc; clear; rng(2025);

%% ---------------- User parameters ----------------
d   = 11;
L   = 1.0;
T   = 0.5;
t0  = 0.0;
N   = 10000;
M   = 2000;
K   = 201;

k_list   = 10;
lambda_t = 1;

dt    = (T - t0) / N;
sqrt2 = sqrt(2);

% ----- 2D binning CE parameters -----
NBIN1    = 501;
NBIN2    = 501;
SMOOTHIT = 2;

% features
p1_fun = @(X) (X(:,1).*X(:,2));
r2_fun = @(X) sum(X.^2,2);

fprintf('d=%d, cube=[-%.1f,%.1f]^d, T=%.3f, N=%d, M=%d, K=%d, lambda=%.2f\n', ...
    d, L, L, T, N, M, K, lambda_t);
fprintf('2D binning CE: NBIN1=%d, NBIN2=%d, SMOOTHIT=%d\n', ...
    NBIN1, NBIN2, SMOOTHIT);

%% ---------------- Loop over k ----------------
for kk = 1:numel(k_list)
    kfreq = k_list(kk);

    % Choose diagonal points so sin(k*pi*a)=±1 when possible
    a  = make_a_grid(K, kfreq, 0.98);
    X0 = repmat(a(:), 1, d);

    fprintf('\n==================== k = %d ====================\n', kfreq);

    results    = zeros(K, 4);  % [u_exact, u_num, abs_err, rel_err]
    u_num_list = zeros(K, 1);
    u_ex_list  = zeros(K, 1);
    stderr_list = zeros(K,1);

    % ordered print queue
    q = parallel.pool.DataQueue;
    afterEach(q, @(data) orderedPrint(data, K));

    tic;
    parfor kpt = 1:K
        x0 = X0(kpt, :);

        [u_hat, stderr] = fk_estimate( ...
            x0, t0, T, L, d, N, M, dt, sqrt2, ...
            NBIN1, NBIN2, SMOOTHIT, p1_fun, r2_fun, ...
            kfreq, lambda_t);

        u_ex   = u_exact(t0, x0, kfreq, lambda_t);
        abs_er = abs(u_hat - u_ex);
        rel_er = abs_er / max(1e-15, abs(u_ex));

        results(kpt,:)   = [u_ex, u_hat, abs_er, rel_er];
        u_num_list(kpt)  = u_hat;
        u_ex_list(kpt)   = u_ex;
        stderr_list(kpt) = stderr;

        send(q, struct( ...
            'kpt',    kpt, ...
            'a',      a(kpt), ...
            'u_ex',   u_ex, ...
            'u_num',  u_hat, ...
            'stderr', stderr ));
    end
    t_elapsed = toc;

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
    % figName = 'HFO_bin2d_2000_64000_11_15';
    % savefig(fig, [figName '.fig']);
    % print(fig, [figName '.eps'], '-depsc2');
    % fprintf('Figure saved as "%s.fig" and "%s.eps"\n', figName, figName);
end

%% =======================================================================
function [u_hat, stderr] = fk_estimate( ...
    x0, t0, T, L, d, N, M, dt, sqrt2, ...
    NBIN1, NBIN2, SMOOTHIT, p1_fun, r2_fun, ...
    kfreq, lambda_t)

X      = cell(N+1,1);
alive  = cell(N+1,1);
dteff  = cell(N,1);
incq   = cell(N,1);
exited = cell(N,1);
bpay   = cell(N,1);

X{1}     = repmat(x0, M, 1);
alive{1} = true(M,1);
t        = t0 * ones(M,1);

payoff = zeros(M,1);
last_k = 1;

for k = 1:N
    tk = t0 + (k-1)*dt;
    Xk = X{k};
    al = alive{k};

    exited{k} = false(M,1);
    bpay{k}   = zeros(M,1);
    incq{k}   = zeros(M,1);
    dteff{k}  = zeros(M,1);
    dteff{k}(al) = dt;

    % Euler proposal
    step = zeros(M,d);
    step(al,:) = sqrt2*sqrt(dt)*randn(nnz(al),d);
    Xnew = Xk + step;
    tnew = t + dt;

    % cube exit detection
    was_in  = all(abs(Xk) < L, 2);
    now_out = any(abs(Xnew) >= L, 2);
    ex      = al & was_in & now_out;
    idx_ok  = al & ~ex;

    % ---- survivors: full-step midpoint quadrature ----
    if any(idx_ok)
        Xmid = Xk(idx_ok,:) + 0.5 * step(idx_ok,:);
        tmid = tk + 0.5 * dt;
        incq{k}(idx_ok) = q_fun_mms_general(tmid, Xmid, d, kfreq, lambda_t) * dt;
    end

    % ---- exiting particles: first-hit interpolation ----
    if any(ex)
        idx = find(ex);

        Xk_ex   = Xk(idx,:);
        Xnew_ex = Xnew(idx,:);
        V_ex    = step(idx,:);

        theta_hit = cube_hit_theta(Xk_ex, Xnew_ex, L);
        theta_hit = min(max(theta_hit, 0), 1);

        ttau = t(idx) + theta_hit * dt;

        % truncated midpoint quadrature
        Xmid_ex = Xk_ex + 0.5 * theta_hit .* V_ex;
        tmid_ex = tk + 0.5 * theta_hit * dt;

        incq{k}(idx) = q_fun_mms_general(tmid_ex, Xmid_ex, d, kfreq, lambda_t) ...
                     .* (theta_hit * dt);

        dteff{k}(idx) = theta_hit * dt;

        % Dirichlet boundary value = 0
        g_now = zeros(size(ttau));
        payoff(idx)    = g_now;
        exited{k}(idx) = true;
        bpay{k}(idx)   = g_now;

        % Retain the exterior candidate and record only the reconstructed time.
        % No increments are generated for this label in later steps.
        t(idx) = ttau;
    end

    X{k+1} = Xnew;

    al_next     = al;
    al_next(ex) = false;
    alive{k+1}  = al_next;

    if any(al_next)
        t(al_next) = tnew(al_next);
    end

    last_k = k+1;
    if ~any(al_next), break; end
end

% terminal payoff for survivors
idx_T = alive{last_k};
if any(idx_T)
    payoff(idx_T) = phi_fun(T, X{last_k}(idx_T,:), kfreq, lambda_t);
end

% backward recursion
Nf = last_k - 1;
Y  = cell(Nf+1,1);
Y{Nf+1} = payoff;

for k = Nf:-1:1
    Ik = find(alive{k});
    if isempty(Ik)
        Y{k} = Y{k+1};
        continue;
    end

    XkI = X{k}(Ik,:);
    dtk = dteff{k}(Ik);

    Ynext_eff = Y{k+1};
    exk = exited{k};
    if any(exk)
        tmp = Ynext_eff;
        tmp(exk) = bpay{k}(exk);
        Ynext_eff = tmp;
    end

    target = Ynext_eff(Ik) + incq{k}(Ik);

    % ===== 2D binning CE on (p1, r2) =====
    p1k = p1_fun(XkI);
    r2k = r2_fun(XkI);
    Hk = ce_bin2d_sumcount(p1k,r2k,[target,dtk],NBIN1,NBIN2,SMOOTHIT);

    % implicit nonlinear solve
    Yk = solve_implicit(Hk(:,1),Hk(:,2), ...
        @(y) cos(y)+exp(sin(y.^2)),[],200);

    y_full     = Y{k+1};
    y_full(Ik) = Yk;
    Y{k}       = y_full;
end

u_hat  = mean(Y{1});
stderr = NaN;
end

%% =======================================================================
function theta_hit = cube_hit_theta(Xk, Xnew, L)
[M,d] = size(Xk);
V = Xnew - Xk;
theta_hit = ones(M,1);

for j = 1:d
    x  = Xk(:,j);
    v  = V(:,j);
    xn = Xnew(:,j);

    idx = (v > 0) & (x < L) & (xn >= L);
    if any(idx)
        th = (L - x(idx)) ./ v(idx);
        theta_hit(idx) = min(theta_hit(idx), th);
    end

    idx = (v < 0) & (x > -L) & (xn <= -L);
    if any(idx)
        th = (-L - x(idx)) ./ v(idx);
        theta_hit(idx) = min(theta_hit(idx), th);
    end
end

theta_hit = min(max(theta_hit, 0), 1);
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
function val = q_fun_mms_general(t, X, d, kfreq, lambda_t)
% q = -(u_t + Delta u + cos(u) + exp(sin(u^2)))

P = prod_sin_kpi(X, kfreq);
u = cos(lambda_t * t) .* P;

du_dt   = -lambda_t .* sin(lambda_t * t) .* P;
Delta_u = -d * (kfreq*pi)^2 .* u;
f       = cos(u) + exp(sin(u.^2));

val = - (du_dt + Delta_u + f);
end

%% ======================= Scalar contraction solve =========================
function y = solve_implicit(H, step_dt, f, ~, itMax)
H = H(:);
if isscalar(step_dt)
    step_dt = repmat(step_dt,size(H));
else
    step_dt = step_dt(:);
end
assert(numel(step_dt) == numel(H) && all(step_dt >= 0), 'Invalid step duration.');
y = H;
tol = 1e-12;
for it = 1:max(200,itMax)
    y = H + step_dt.*f(y);
    residual = y - H - step_dt.*f(y);
    if any(~isfinite(y)) || any(~isfinite(residual))
        error('IBSDE:ScalarSolve', 'Nonfinite fixed-point iterate. Check the time step and source.');
    end
    if norm(residual,inf) <= tol
        return;
    end
end
error('IBSDE:ScalarSolve', 'Fixed-point residual exceeds tolerance. Reduce the time step.');
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
    persistent last_printed results_buffer

    if isempty(last_printed) || last_printed >= K
        last_printed = 0;
        results_buffer = containers.Map('KeyType','int32','ValueType','any');
    end

    results_buffer(data.kpt) = data;

    k = last_printed + 1;
    while isKey(results_buffer, k)
        d = results_buffer(k);

        abs_er = abs(d.u_num - d.u_ex);
        rel_er = abs_er / max(1e-15, abs(d.u_ex));

        fprintf('pt %3d/%3d | a=% .6f | exact=%.8e | num=%.8e | abs=%.2e | rel=%.2e\n', ...
            d.kpt, K, d.a, d.u_ex, d.u_num, abs_er, rel_er);

        last_printed = k;
        remove(results_buffer, k);
        k = k + 1;
    end
end
