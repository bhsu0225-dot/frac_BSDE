function frac_IBSDE_complex_nonhomog()
% =========================================================================
% Author: Bihao Su
% =========================================================================

close all; clc;

%% ---------------- reproducibility ----------------
seed = 10086;
try
    maxNumCompThreads(1);
catch
end
rng(seed, 'twister');

%% ================= user parameters =================
% ---- recommended runnable defaults ----
d       = 100;
s       = 0.25;              % if s=1 => Brownian case, generator L = Delta
alpha   = 2*s;

T       = 50;               
t0      = 0.0;
N       = 800;               
M_list  = [1000,10000];      
K       = 101;               

k1 = 1; 
k2 = 2;

beta = 0.3;
eta  = 0.7;

% domain scale
Rout = 2.0;

% outer ellipsoid
a_axes = Rout * ones(1,d);
a_axes(1) = 1.5 * Rout;
a_axes(2) = 2.0 * Rout;

% inner hole
c_inner = zeros(1,d);
c_inner(3) = 0.55 * Rout;
r_in = 0.25 * Rout;

% CE 2D binning
NBIN1    = 501;
NBIN2    = 501;
SMOOTHIT = 2;

NEWTON_IT = 20;
dt = (T - t0) / N;

 
usePar = true;

fprintf('=============================================================\n');
fprintf('[RUNNABLE-IBSDE] d=%d, s=%.3f (alpha=%.3f), T=%.2f, N=%d, dt=%.3e\n', ...
    d, s, alpha, T, N, dt);
fprintf('K=%d, M_list=[%s], line: a*(e_%d+e_%d)\n', ...
    K, num2str(M_list), k1, k2);
fprintf('Domain: outer ellipsoid minus inner ball\n');
fprintf('=============================================================\n');

%% ---------------- A(t) ----------------
Afun  = @(t) exp(cos(t));
dA_dt = @(t) -sin(t) .* exp(cos(t));

%% ---------------- nonlinearity ----------------
f  = @(u) eta * (exp(beta*u) - u);
df = @(u) eta * beta * exp(beta*u) - eta;

%% ---------- (-Delta)^s phi constant and 1F1 params ----------
% phi(x) = x_{k1} x_{k2} exp(-|x|^2)
% L_s phi := (+)(-Delta)^s phi  = Cds * x_{k1}x_{k2} * 1F1(a+2;b+2;-|x|^2)
b = d/2;
a = d/2 + s;
Cds = 2^(2*s) * gamma(a)/gamma(b) * (a*(a+1)) / (b*(b+1));
a2 = a + 2;
b2 = b + 2;

%% ---------------- helpers ----------------
inD = @(X) in_domain_complex(X, a_axes, c_inner, r_in);

phi_fun = @(X) (X(:,k1) .* X(:,k2) .* exp(-sum(X.^2, 2)));
p1_fun  = @(X) (X(:,k1) .* X(:,k2));
r2_fun  = @(X) sum(X.^2, 2);

F1F1_neg = @(r2) kummer1f1_neg_series_vec(a2, b2, r2);

% (+)(-Delta)^s phi
Ls_phi_fun = @(X) (Cds .* (X(:,k1).*X(:,k2)) .* F1F1_neg(sum(X.^2,2)));

u_exact_fun = @(t,X) (Afun(t) .* phi_fun(X));
g_fun       = @(t,X) u_exact_fun(t, X);    % nonhomogeneous exterior data

% source term q = -(u_t + L u + f(u)), with L = -(-Delta)^s
q_fun = @(t,X) q_manufactured_signfix(t, X, Afun, dA_dt, phi_fun, Ls_phi_fun, f);

%% ---------------- initial points: x0 = a*(e_k1 + e_k2) ----------------
Rgrid = 0.99 / sqrt((1/a_axes(k1)^2) + (1/a_axes(k2)^2));
avec  = linspace(-Rgrid, Rgrid, K).';

X0 = zeros(K, d);
X0(:,k1) = avec;
X0(:,k2) = avec;

for kk = 1:K
    x = X0(kk,:);
    shrink = 1.0;
    while true
        xt = shrink * x;
        if inD(xt)
            X0(kk,:) = xt;
            break;
        end
        shrink = 0.9 * shrink;
        if shrink < 1e-12
            error('Failed to place an initial point inside D. Adjust domain/hole.');
        end
    end
end

%% ---------------- exact curve at t0 ----------------
u_ex_ref = u_exact_fun(t0, X0);

%% ---------------- run for each M ----------------
numM = numel(M_list);
u_num_all      = zeros(K, numM);
kill_rate_all  = zeros(K, numM);
summary_all    = zeros(numM, 5);    
elapsed_all    = zeros(numM, 1);

for im = 1:numM
    M = M_list(im);
    fprintf('\n===== Running case M = %d (%d/%d) =====\n', M, im, numM);

    tCase = tic;

    results = zeros(K, 4);    
    u_num_list = zeros(K,1);
    kill_rate_list = zeros(K,1);

    if usePar
        pool = gcp('nocreate');
        if isempty(pool)
            warning('usePar=true but no parallel pool exists. Switch to serial.');
            usePar_local = false;
        else
            usePar_local = true;
        end
    else
        usePar_local = false;
    end

    if usePar_local
        dq = parallel.pool.DataQueue;
        nDone = 0;
        afterEach(dq, @onData);

        parfor kk = 1:K
            rng(seed + kk, 'twister');

            x0 = X0(kk,:);

            [p1_hist, r2_hist, phi_hist, incq, kill_step, bpay, lastN] = ...
                forward_subBM_stop_complex_store( ...
                    x0, d, N, dt, s, M, inD, q_fun, g_fun, p1_fun, r2_fun, phi_fun, t0);

            kill_rate = mean(kill_step > 0);

            Y = terminal_payoff_from_hist(phi_hist, kill_step, bpay, Afun, t0, dt, lastN);

            % backward recursion
            for n = lastN:-1:1
                alive_n = (kill_step == 0) | (kill_step >= n);
                if ~any(alive_n)
                    continue;
                end

                idx = find(alive_n);
                targ = Y(idx) + double(incq(idx, n));
                p1n  = double(p1_hist(idx, n));
                r2n  = double(r2_hist(idx, n));

                H = ce_bin2d_sumcount(p1n, r2n, targ, NBIN1, NBIN2, SMOOTHIT);
                Y(idx) = newton_implicit_general(H, dt, f, df, NEWTON_IT);
            end

            u_num = mean(Y);
            u_ex  = u_exact_fun(t0, x0);
            abs_err = abs(u_num - u_ex);
            rel_err = abs_err / max(1e-14, abs(u_ex));

            u_num_list(kk) = u_num;
            kill_rate_list(kk) = kill_rate;
            results(kk,:) = [u_ex, u_num, abs_err, rel_err];

            msg = sprintf(['M=%d | a=%+.4e | r=%.3f | exact=%+.6e | num=%+.6e | ', ...
                           'abs=%.2e | rel=%.2e | lastN=%d | kill=%.3f\n'], ...
                           M, x0(k1), norm(x0), u_ex, u_num, abs_err, rel_err, lastN, kill_rate);
            send(dq, msg);
        end

    else
        for kk = 1:K
            rng(seed + kk, 'twister');

            x0 = X0(kk,:);

            [p1_hist, r2_hist, phi_hist, incq, kill_step, bpay, lastN] = ...
                forward_subBM_stop_complex_store( ...
                    x0, d, N, dt, s, M, inD, q_fun, g_fun, p1_fun, r2_fun, phi_fun, t0);

            kill_rate = mean(kill_step > 0);
            kill_rate_list(kk) = kill_rate;

            Y = terminal_payoff_from_hist(phi_hist, kill_step, bpay, Afun, t0, dt, lastN);

            % backward recursion
            for n = lastN:-1:1
                alive_n = (kill_step == 0) | (kill_step >= n);
                if ~any(alive_n)
                    continue;
                end

                idx = find(alive_n);
                targ = Y(idx) + double(incq(idx, n));
                p1n  = double(p1_hist(idx, n));
                r2n  = double(r2_hist(idx, n));

                H = ce_bin2d_sumcount(p1n, r2n, targ, NBIN1, NBIN2, SMOOTHIT);
                Y(idx) = newton_implicit_general(H, dt, f, df, NEWTON_IT);
            end

            u_num = mean(Y);
            u_ex  = u_exact_fun(t0, x0);

            u_num_list(kk) = u_num;

            abs_err = abs(u_num - u_ex);
            rel_err = abs_err / max(1e-14, abs(u_ex));
            results(kk,:) = [u_ex, u_num, abs_err, rel_err];

            fprintf(['M=%d | pt %2d/%2d | a=%+.4e | r=%.3f | exact=%+.6e | num=%+.6e | ', ...
                     'abs=%.2e | rel=%.2e | lastN=%d | kill=%.3f\n'], ...
                     M, kk, K, x0(k1), norm(x0), u_ex, u_num, abs_err, rel_err, lastN, kill_rate);
        end
    end

    tElapsed = toc(tCase);
    elapsed_all(im) = tElapsed;

    L2err   = sqrt(mean(results(:,3).^2));
    MedAbs  = median(results(:,3));
    MeanRel = mean(results(:,4));
    MeanKill= mean(kill_rate_list);

    fprintf('\n=== SUMMARY (M=%d) ===\n', M);
    fprintf('L2 err          = %.3e\n', L2err);
    fprintf('Median abs err  = %.3e\n', MedAbs);
    fprintf('Mean rel err    = %.3e\n', MeanRel);
    fprintf('Mean kill rate  = %.3f\n', MeanKill);
    fprintf('Elapsed time    = %.2f s\n', tElapsed);

    u_num_all(:, im)     = u_num_list;
    kill_rate_all(:, im) = kill_rate_list;
    summary_all(im,:)    = [M, L2err, MedAbs, MeanRel, MeanKill];
end

%% ---------------- overall summary print ----------------
fprintf('\n================ Overall Summary by M ================\n');
for im = 1:numM
    fprintf('M=%-6d | L2=%.3e | MedAbs=%.3e | MeanRel=%.3e | MeanKill=%.3f | Time=%.2fs\n', ...
        summary_all(im,1), summary_all(im,2), summary_all(im,3), summary_all(im,4), summary_all(im,5), elapsed_all(im));
end
fprintf('======================================================\n');

%% ================= plot =================
fig = figure('Name', 'Exact_vs_Numerical_M_Comparison', 'Color', 'w');

plot(avec, u_ex_ref, 'k-', 'LineWidth', 2.0);
hold on;

colors = [0, 0.45, 0.74;
          0.85, 0.33, 0.10;
          0.93, 0.69, 0.13;
          0.49, 0.18, 0.56];

for im = 1:numM
    cidx = min(im, size(colors,1));
    plot(avec, u_num_all(:,im), '-.', ...
        'Color', colors(cidx,:), ...
        'LineWidth', 1.8);
end

grid on;
box on;
xlabel('$a \quad (x_0 = a(e_{k_1}+e_{k_2}))$', 'Interpreter', 'latex', 'FontSize', 16);
ylabel('$u(t_0,x_0)$', 'Interpreter', 'latex', 'FontSize', 16);
title(sprintf('Exact vs Numerical'));
xlim([min(avec), max(avec)]);

ymin = min([u_ex_ref(:); u_num_all(:)]);
ymax = max([u_ex_ref(:); u_num_all(:)]);
ypad = 0.08 * max(1e-8, ymax - ymin);
ylim([0, 0.7]);

legtxt = cell(1, numM + 1);
legtxt{1} = 'Exact';
for im = 1:numM
    legtxt{im+1} = sprintf('Numerical (M=%d)', M_list(im));
end
legend(legtxt, 'Location', 'best', 'Interpreter', 'none');

set(gca, 'FontSize', 16, 'LineWidth', 1.5);

%% ---------------- save results ----------------
% tag = sprintf('complex_nonhomog_d%d_alpha%.2f_T%.2f_N%d', d, alpha, T, N);
% savefig(fig, [tag, '.fig']);
% print(fig, [tag, '.eps'], '-depsc2');
% save([tag, '.mat'], ...
%     'avec', 'X0', 'u_ex_ref', 'u_num_all', 'kill_rate_all', ...
%     'summary_all', 'elapsed_all', ...
%     'd', 's', 'alpha', 'T', 't0', 'N', 'dt', 'M_list', 'K', ...
%     'a_axes', 'c_inner', 'r_in', ...
%     'NBIN1', 'NBIN2', 'SMOOTHIT', 'NEWTON_IT', ...
%     'beta', 'eta', 'seed');
% 
% fprintf('Saved:\n');
% fprintf('  %s.fig\n', tag);
% fprintf('  %s.eps\n', tag);
% fprintf('  %s.mat\n', tag);

%% ---------------- nested callback ----------------
    function onData(msg)
        nDone = nDone + 1;
        fprintf('%s', msg);
        fprintf('Progress: %6.2f%% (%d/%d)\n', 100*nDone/K, nDone, K);
    end

end

% =========================================================================
% Manufactured source q
% =========================================================================
function qv = q_manufactured_signfix(t, X, Afun, dA_dt, phi_fun, Ls_phi_fun, f)
% L = -(-Delta)^s
% u(t,x) = A(t) phi(x)
% q = -(u_t + L u + f(u))
phi   = phi_fun(X);
A     = Afun(t);
At    = dA_dt(t);
Lsphi = Ls_phi_fun(X);      % (+)(-Delta)^s phi
u     = A .* phi;

% Since L u = -A * (+)(-Delta)^s phi,
% q = -( At*phi - A*Lsphi + f(u) )
qv = -(At .* phi - A .* Lsphi + f(u));
end

% =========================================================================
% Domain: ellipsoid minus inner ball
% =========================================================================
function inside = in_domain_complex(X, a_axes, c_inner, r_in)
ell  = sum((X ./ a_axes).^2, 2) < 1.0;
hole = sum((X - c_inner).^2, 2) <= r_in^2;
inside = ell & ~hole;
end

% =========================================================================
% Forward simulation and storage
% =========================================================================
function [p1_hist, r2_hist, phi_hist, incq, kill_step, bpay, lastN] = ...
    forward_subBM_stop_complex_store( ...
    x0, d, N, dt, s, M, inD, q_fun, g_fun, p1_fun, r2_fun, phi_fun, t0)

X = repmat(reshape(x0,1,[]), M, 1);
alive = true(M,1);

p1_hist  = zeros(M, N+1, 'single');
r2_hist  = zeros(M, N+1, 'single');
phi_hist = zeros(M, N+1, 'single');
incq     = zeros(M, N,   'single');

kill_step = zeros(M,1,'int32');   % 0 => survives to T
bpay      = zeros(M,1);           % payoff at exit

p1_hist(:,1)  = single(p1_fun(X));
r2_hist(:,1)  = single(r2_fun(X));
phi_hist(:,1) = single(phi_fun(X));

lastN = N;

for n = 1:N
    tn = t0 + (n-1)*dt;

    idx = find(alive);
    if isempty(idx)
        lastN = n - 1;
        for k = n:N
            p1_hist(:,k+1)  = p1_hist(:,k);
            r2_hist(:,k+1)  = r2_hist(:,k);
            phi_hist(:,k+1) = phi_hist(:,k);
        end
        break;
    end

    Xk = X(idx,:);

    % Left-endpoint quadrature of q over [tn, tn+dt]
    qv = q_fun(tn, Xk);
    incq(idx, n) = single(qv * dt);

    % Sample increment
    if abs(s - 1.0) < 1e-14
        % Brownian case: dS = dt, generator = Delta
        Z = randn(numel(idx), d);
        step = sqrt(2*dt) * Z;
    else
        dS = subordinator_incr(dt, s, [numel(idx), 1]);
        Z  = randn(numel(idx), d);
        step = sqrt(2) .* (sqrt(dS) .* ones(1,d)) .* Z;
    end

    Xcand = Xk + step;

    inside = inD(Xcand);
    surv_idx = idx(inside);
    died_idx = idx(~inside);

    if ~isempty(surv_idx)
        X(surv_idx,:) = Xcand(inside,:);
    end

    if ~isempty(died_idx)
        alive(died_idx) = false;
        kill_step(died_idx) = n;
        t_exit = tn + dt;
        bpay(died_idx) = g_fun(t_exit, Xcand(~inside,:));
    end

    % Carry forward stored features
    p1_hist(:, n+1)  = p1_hist(:, n);
    r2_hist(:, n+1)  = r2_hist(:, n);
    phi_hist(:, n+1) = phi_hist(:, n);

    if ~isempty(surv_idx)
        p1_hist(surv_idx, n+1)  = single(p1_fun(X(surv_idx,:)));
        r2_hist(surv_idx, n+1)  = single(r2_fun(X(surv_idx,:)));
        phi_hist(surv_idx, n+1) = single(phi_fun(X(surv_idx,:)));
    end
end
end

% =========================================================================
% Terminal payoff initialization from stored history
% =========================================================================
function Y = terminal_payoff_from_hist(phi_hist, kill_step, bpay, Afun, t0, dt, lastN)
M = size(phi_hist, 1);
Y = zeros(M,1);

aliveT = (kill_step == 0);
phiT = double(phi_hist(:, lastN+1));

if any(aliveT)
    Y(aliveT) = Afun(t0 + lastN*dt) .* phiT(aliveT);
end

died = ~aliveT;
if any(died)
    Y(died) = bpay(died);
end
end

% =========================================================================
% One-sided s-stable subordinator increment (CMS)
% valid only for 0 < s < 1
% =========================================================================
function dS = subordinator_incr(dt, s, sz)
if ~(s > 0 && s < 1)
    error('subordinator_incr is valid only for 0 < s < 1. For s=1 use Brownian branch.');
end

U = pi * rand(sz);
E = -log(rand(sz));
Sa = (sin(s*U) ./ (sin(U)).^(1/s)) .* (sin((1-s)*U) ./ E).^((1-s)/s);
dS = (dt.^(1/s)) .* Sa;
dS = max(dS, 0);
end

% =========================================================================
% 1F1(a;b;-x) via series recursion
% =========================================================================
function F = kummer1f1_neg_series_vec(a, b, x)
x = double(x(:));
x = max(x, 0);

m = numel(x);
F = ones(m,1);
term = ones(m,1);

% crude but practical truncation for large x
Xcut = 200;
large = (x > Xcut);
F(large) = 0;
active = ~large;

tol = 1e-12;
nmax = 1200;

for n = 0:nmax-1
    if ~any(active)
        break;
    end
    idx = find(active);

    term(idx) = term(idx) .* ((a+n)./(b+n)) .* (-x(idx)) ./ (n+1);
    F(idx)    = F(idx) + term(idx);

    stop = abs(term(idx)) <= tol .* max(1, abs(F(idx)));
    active(idx(stop)) = false;
end
end

% =========================================================================
% CE: 2D bin smoother on (p1,r2)
% =========================================================================
function H = ce_bin2d_sumcount(p1, r2, y, NBIN1, NBIN2, smooth_it)
p1 = p1(:);
r2 = r2(:);
y  = y(:);

m = numel(p1);
if m == 0
    H = zeros(0,1);
    return;
end
if m == 1
    H = y;
    return;
end

p1lo = min(p1); p1hi = max(p1);
r2lo = min(r2); r2hi = max(r2);

if (p1hi-p1lo) < 1e-14 || (r2hi-r2lo) < 1e-14 || ...
        ~isfinite(p1lo) || ~isfinite(p1hi) || ~isfinite(r2lo) || ~isfinite(r2hi)
    H = mean(y) * ones(m,1);
    return;
end

pad1 = 1e-12 + 0.02*(p1hi-p1lo);
pad2 = 1e-12 + 0.02*(r2hi-r2lo);

p1lo = p1lo - pad1; p1hi = p1hi + pad1;
r2lo = r2lo - pad2; r2hi = r2hi + pad2;

i1 = floor((p1 - p1lo) ./ (p1hi - p1lo) * (NBIN1-1)) + 1;
i2 = floor((r2 - r2lo) ./ (r2hi - r2lo) * (NBIN2-1)) + 1;

i1 = max(min(i1, NBIN1), 1);
i2 = max(min(i2, NBIN2), 1);

lin = i1 + (i2-1)*NBIN1;

sumY = accumarray(lin, y, [NBIN1*NBIN2, 1], @sum, 0);
cnt  = accumarray(lin, 1, [NBIN1*NBIN2, 1], @sum, 0);

S = reshape(sumY, [NBIN1, NBIN2]);
C = reshape(cnt,  [NBIN1, NBIN2]);

ker = [1;2;1]/4;
for it = 1:smooth_it
    S = conv2(conv2(S, ker, 'same'), ker', 'same');
    C = conv2(conv2(C, ker, 'same'), ker', 'same');
end

MU = S ./ max(C, 1e-12);
H = MU(lin);
end

% =========================================================================
% Implicit solve: y - dt f(y) = H
% =========================================================================
function y = newton_implicit_general(H, dt, f, df, itMax)
H = H(:);
y = H;

for it = 1:itMax
    F = y - dt .* f(y) - H;
    J = 1 - dt .* df(y);

    J = sign(J) .* max(abs(J), 1e-12);
    step = -F ./ J;

    % mild damping for robustness
    big = abs(step) > 2.0;
    y_try = y + step;
    if any(big)
        y_try(big) = y(big) + 0.5 * step(big);
    end

    y = y_try;

    if norm(F, inf) < 1e-10
        break;
    end
end

% clipping for robustness
y = max(min(y, 50), -50);
end