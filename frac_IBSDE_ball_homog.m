function frac_IBSDE_ball_homog()
% =========================================================================
% PDE:
%   u_t + L u + f(u) + q = 0, in [t0,T) x B_1,
%   u = 0,                    on [t0,T] x (R^d \ B_1),
%   u(T,x) = u_ex(T,x),       in B_1,
%
% where
%   f(u) = sin(u) + exp(cos(u))
%
% Exact solution:
%   u_ex(t,x) = A(t) * (1 - |x|^2)_+^{alpha/2},
%   A(t) = cos(t)/(1 + 10 t^2)
%
% 1) Exact-in-law stable increments, with a separate Brownian branch.
% 2) Detect exits only at grid endpoints and freeze each exited label.
% 3) Full-step left-endpoint source dt*q(tn,Xk), including the exit step.
% 4) Regress Y_next+dt*q by smoothing target sums and counts.
% 5) Solve y=H+dt*f(y) with a checked scalar residual.
% =========================================================================

close all; clc;

%% ---------------- reproducibility ----------------
seed = 20260306;
try
    maxNumCompThreads(1);
catch
end
rng(seed, 'twister');

%% ================= user parameters =================
d       = 100;
alpha   = 1.5;          % in (0,2]
s       = alpha / 2;    % in (0,1]

T       = 1.0;
t0      = 0.0;
N       = 640;
dt      = (T - t0) / N;

M_list  = [10000, 20000];
K       = 101;

SOLVE_MAXIT = 200;
SOLVE_TOL   = 1e-12;        % Bound on each scalar root error.

% This example implements kappa=2; the paper allows 2<=kappa<=d.
kappa = 2;

% CE binning
NBIN1    = 500;
NBIN2    = 500;
SMOOTHIT = 2;

% parallel option
usePar = true;              
numWorkers = 15;
save_results = true;        % Save FIG, EPS, PNG and the numerical MAT data.
PNG_DPI = 300;
outdir = fullfile(fileparts(mfilename('fullpath')), 'output_ball_endpoint');

assert(d >= 2 && kappa == 2, 'This standalone example implements kappa=2.');
assert(alpha > 0 && alpha <= 2 && T > t0);
assert(N >= 1 && N == floor(N) && all(M_list >= 1 & M_list == floor(M_list)));
assert(NBIN1 >= 2 && NBIN2 >= 2 && SMOOTHIT >= 0 && SMOOTHIT == floor(SMOOTHIT));
assert((1+exp(1))*dt < 1, 'Need (1+e)*dt<1 for the scalar contraction. Increase N.');
if save_results && ~exist(outdir,'dir'), mkdir(outdir); end
if usePar
    if isempty(ver('parallel')) || ~license('test','Distrib_Computing_Toolbox')
        warning('Parallel toolbox unavailable. Running serially.');
        usePar = false;
    else
        try
            if isempty(gcp('nocreate')), parpool('local',numWorkers); end
        catch err
            warning('Could not start parallel pool: %s. Running serially.',err.message);
            usePar = false;
        end
    end
end

fprintf('=============================================================\n');
fprintf('[FRAC-BALL-LINE-a1d-ENDPOINT] d=%d, alpha=%.3f, s=%.3f, T=%.2f, N=%d, dt=%.3e\n', ...
    d, alpha, s, T, N, dt);
fprintf('K=%d, M_list=[%s]\n', K, num2str(M_list));
fprintf('Path line: x = a * 1_d, a in [%g,%g]\n', -1.5/sqrt(d), 1.5/sqrt(d));
fprintf('Outside B_1: force solution value to boundary value 0\n');
fprintf('=============================================================\n');

%% ---------------- A(t), A'(t) ----------------
Afun  = @(t) cos(t) ./ (1 + 10*t.^2);
dA_dt = @(t) ((-sin(t)).*(1 + 10*t.^2) - 20*t.*cos(t)) ./ (1 + 10*t.^2).^2;

%% ---------------- nonlinearity ----------------
f  = @(u) sin(u) + exp(cos(u));
df = @(u) cos(u) - sin(u).*exp(cos(u));

%% ---------------- constant C_{d,alpha} ----------------
Cda = 2^alpha * gamma(1 + alpha/2) * gamma((d + alpha)/2) / gamma(d/2);

%% ---------------- helpers ----------------
inD = @(X) sum(X.^2, 2) < 1.0;

psi_fun     = @(X) max(1 - sum(X.^2, 2), 0).^(alpha/2);
u_exact_fun = @(t,X) Afun(t) .* psi_fun(X);

% exterior boundary value = 0
g_fun = @(t,X) zeros(size(X,1),1);

% source term
q_fun = @(t,X) q(t, X, Afun, dA_dt, alpha, Cda, f);

% Paper Example 1: axial coordinate and transverse radius (ball radius 1).
p1_fun = @(X) sum(X,2)/sqrt(d);
r2_fun = @(X) sqrt(max(sum(X.^2,2)-p1_fun(X).^2,0));

%% ---------------- x = a * 1_d ----------------
avec = linspace(-1.5*sqrt(1/d), 1.5*sqrt(1/d), K).';

X0 = zeros(K, d);
for kk = 1:K
    X0(kk,:) = avec(kk) * ones(1,d);
end

% exact values at t0; outside automatically gives 0
u_ex_ref = u_exact_fun(t0, X0);

%% ---------------- run for each M ----------------
numM = numel(M_list);
u_num_all   = zeros(K, numM);
summary_all = zeros(numM, 5);   % [M, L2, medAbs, meanRel, meanKill]
elapsed_all = zeros(numM, 1);
max_residual_all = zeros(numM,1);
kill_rate_all = zeros(K,numM);

for im = 1:numM
    M = M_list(im);
    fprintf('\n===== Running case M = %d (%d/%d) =====\n', M, im, numM);

    tCase = tic;

    results = zeros(K, 4); % [u_exact, u_num, abs_err, rel_err]
    kill_rate_list = zeros(K,1);
    residual_list = zeros(K,1);

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
            rng(seed + 1000*im + kk, 'twister');

            x0 = X0(kk,:);
            r0 = norm(x0);

            % Force outside points to boundary value 0
            if ~inD(x0)
                u_ex  = 0.0;
                u_num = 0.0;
                abs_err = 0.0;
                rel_err = 0.0;
                kill_rate = 1.0;

                results(kk,:) = [u_ex, u_num, abs_err, rel_err];
                kill_rate_list(kk) = kill_rate;

                msg = sprintf(['M=%d | a=%+.4e | r=%.3f | OUTSIDE -> force 0 | ', ...
                               'exact=%+.6e | num=%+.6e\n'], ...
                               M, avec(kk), r0, u_ex, u_num);
                send(dq, msg);
                continue;
            end

            [p1_hist, r2_hist, psi_hist, incq, kill_step, bpay, lastN] = ...
                forward_subBM( ...
                    x0, d, N, dt, s, M, inD, q_fun, g_fun, p1_fun, r2_fun, psi_fun, t0);

            kill_rate = mean(kill_step > 0);

            Y = terminal_payoff(psi_hist, kill_step, bpay, Afun, t0, dt, lastN);

            point_residual = 0;
            for n = lastN:-1:1
                alive_n = (kill_step == 0) | (kill_step >= n);
                if ~any(alive_n)
                    continue;
                end

                idx  = find(alive_n);
                targ = Y(idx) + double(incq(idx,n));
                p1n  = double(p1_hist(idx,n));
                r2n  = double(r2_hist(idx,n));

                H = ce_bin2d_sumcount(p1n, r2n, targ, NBIN1, NBIN2, SMOOTHIT);
                [Y(idx), scalar_residual] = solve_implicit(H, dt, f, df, SOLVE_MAXIT, SOLVE_TOL);
                point_residual = max(point_residual,scalar_residual);
            end

            residual_list(kk) = point_residual;
            u_num = mean(Y);
            u_ex  = u_exact_fun(t0, x0);

            abs_err = abs(u_num - u_ex);
            rel_err = abs_err / max(1e-14, abs(u_ex));

            results(kk,:) = [u_ex, u_num, abs_err, rel_err];
            kill_rate_list(kk) = kill_rate;

            msg = sprintf(['M=%d | a=%+.4e | r=%.3f | exact=%+.6e | num=%+.6e | ', ...
                           'abs=%.2e | rel=%.2e | lastN=%d | kill=%.3f\n'], ...
                           M, avec(kk), r0, u_ex, u_num, abs_err, rel_err, lastN, kill_rate);
            send(dq, msg);
        end

    else
        for kk = 1:K
            rng(seed + 1000*im + kk, 'twister');

            x0 = X0(kk,:);
            r0 = norm(x0);

            % Force outside points to boundary value 0
            if ~inD(x0)
                u_ex  = 0.0;
                u_num = 0.0;
                abs_err = 0.0;
                rel_err = 0.0;
                kill_rate = 1.0;

                results(kk,:) = [u_ex, u_num, abs_err, rel_err];
                kill_rate_list(kk) = kill_rate;

                fprintf(['M=%d | pt %3d/%3d | a=%+.4e | r=%.3f | OUTSIDE -> force 0 | ', ...
                         'exact=%+.6e | num=%+.6e\n'], ...
                         M, kk, K, avec(kk), r0, u_ex, u_num);
                continue;
            end

            [p1_hist, r2_hist, psi_hist, incq, kill_step, bpay, lastN] = ...
                forward_subBM( ...
                    x0, d, N, dt, s, M, inD, q_fun, g_fun, p1_fun, r2_fun, psi_fun, t0);

            kill_rate = mean(kill_step > 0);
            kill_rate_list(kk) = kill_rate;

            Y = terminal_payoff(psi_hist, kill_step, bpay, Afun, t0, dt, lastN);

            point_residual = 0;
            for n = lastN:-1:1
                alive_n = (kill_step == 0) | (kill_step >= n);
                if ~any(alive_n)
                    continue;
                end

                idx  = find(alive_n);
                targ = Y(idx) + double(incq(idx,n));
                p1n  = double(p1_hist(idx,n));
                r2n  = double(r2_hist(idx,n));

                H = ce_bin2d_sumcount(p1n, r2n, targ, NBIN1, NBIN2, SMOOTHIT);
                [Y(idx), scalar_residual] = solve_implicit(H, dt, f, df, SOLVE_MAXIT, SOLVE_TOL);
                point_residual = max(point_residual,scalar_residual);
            end

            residual_list(kk) = point_residual;
            u_num = mean(Y);
            u_ex  = u_exact_fun(t0, x0);

            abs_err = abs(u_num - u_ex);
            rel_err = abs_err / max(1e-14, abs(u_ex));
            results(kk,:) = [u_ex, u_num, abs_err, rel_err];

            fprintf(['M=%d | pt %3d/%3d | a=%+.4e | r=%.3f | exact=%+.6e | num=%+.6e | ', ...
                     'abs=%.2e | rel=%.2e | lastN=%d | kill=%.3f\n'], ...
                     M, kk, K, avec(kk), r0, u_ex, u_num, abs_err, rel_err, lastN, kill_rate);
        end
    end

    tElapsed = toc(tCase);
    elapsed_all(im) = tElapsed;
    max_residual_all(im) = max(residual_list);
    kill_rate_all(:,im) = kill_rate_list;

    L2err    = sqrt(mean(results(:,3).^2));
    MedAbs   = median(results(:,3));
    MeanRel  = mean(results(:,4));
    MeanKill = mean(kill_rate_list);

    summary_all(im,:) = [M, L2err, MedAbs, MeanRel, MeanKill];

    fprintf('\n=== SUMMARY (M=%d) ===\n', M);
    fprintf('L2 err (all query points) = %.3e\n', L2err);
    fprintf('Max scalar residual      = %.3e\n', max_residual_all(im));
    fprintf('Median abs err  = %.3e\n', MedAbs);
    fprintf('Mean rel err    = %.3e\n', MeanRel);
    fprintf('Mean kill rate  = %.3f\n', MeanKill);
    fprintf('Elapsed time    = %.2f s\n', tElapsed);

    u_num_all(:,im) = results(:,2);
end

%% ---------------- overall summary ----------------
fprintf('\n================ Overall Summary by M ================\n');
for im = 1:numM
    fprintf('M=%-6d | L2=%.3e | MedAbs=%.3e | MeanRel=%.3e | MeanKill=%.3f | Time=%.2fs\n', ...
        summary_all(im,1), summary_all(im,2), summary_all(im,3), summary_all(im,4), summary_all(im,5), elapsed_all(im));
end
fprintf('======================================================\n');

%% ---------------- single comparison plot ----------------
fig = figure('Name', 'Exact_vs_Numerical', 'Color', 'w');

plot(avec, u_ex_ref, 'k-', 'LineWidth', 2.0);
hold on;

colors = lines(numM);
for im = 1:numM
    plot(avec, u_num_all(:,im), '-.', ...
        'Color', colors(im,:), ...
        'LineWidth', 1.8);
end

grid on;
box on;
xlabel('$a \quad (x = a\,\mathbf{1}_d)$', 'Interpreter', 'latex', 'FontSize', 16);
ylabel('$u(t_0,x)$', 'Interpreter', 'latex', 'FontSize', 16);
title('Exact vs Numerical', 'Interpreter', 'tex', 'FontSize', 16);

xlim([min(avec), max(avec)]);
ylim([0, 1.4]);

lgd = cell(1, numM+1);
lgd{1} = 'Exact';
for im = 1:numM
    lgd{im+1} = sprintf('Numerical (M=%d)', M_list(im));
end
legend(lgd, 'Location', 'NorthEast', 'Interpreter', 'none');

set(gca, 'FontSize', 16, 'LineWidth', 2.0);

%% ---------------- save ----------------
if save_results
    tag = sprintf('frac_ball_endpoint_d%d_alpha%.2f_T%.2f_N%d', d, alpha, T, N);
    savefig(fig, fullfile(outdir,[tag '.fig']));
    print(fig, fullfile(outdir,[tag '.eps']), '-depsc2');
    print(fig, fullfile(outdir,[tag '.png']), '-dpng', sprintf('-r%d',PNG_DPI));
    save(fullfile(outdir,[tag '.mat']), ...
        'avec','X0','u_ex_ref','u_num_all','summary_all','elapsed_all', ...
        'max_residual_all','kill_rate_all', ...
        'd','alpha','s','T','t0','N','dt','M_list','K','kappa', ...
        'NBIN1','NBIN2','SMOOTHIT','SOLVE_MAXIT','SOLVE_TOL','seed','Cda');
    fprintf('Saved FIG, EPS, PNG and MAT files in:\n%s\n',outdir);
end

%% ---------------- nested callback ----------------
    function onData(msg)
        nDone = nDone + 1;
        fprintf('%s', msg);
        fprintf('Progress: %6.2f%% (%d/%d)\n', 100*nDone/K, nDone, K);
    end
end

% =========================================================================
% Source term q for the unit-ball semilinear fractional example
% =========================================================================
function qv = q(t, X, Afun, dA_dt, alpha, Cda, f)
r2 = sum(X.^2, 2);
inside = (r2 < 1.0);

qv = zeros(size(r2));

if any(inside)
    psi = (1 - r2(inside)).^(alpha/2);
    A   = Afun(t);
    At  = dA_dt(t);
    if ~isscalar(A)
        A = A(inside);
        At = At(inside);
    end
    u   = A .* psi;

    % q = -u_t - L u - f(u), and L u = -C_{d,alpha} A(t)
    qv(inside) = -At .* psi + Cda .* A - f(u);
end
end

% =========================================================================
% Forward simulation and storage on the unit ball
% =========================================================================
function [p1_hist, r2_hist, psi_hist, incq, kill_step, bpay, lastN] = ...
    forward_subBM( ...
    x0, d, N, dt, s, M, inD, q_fun, g_fun, p1_fun, r2_fun, psi_fun, t0)

X = repmat(reshape(x0,1,[]), M, 1);
alive = true(M,1);

p1_hist  = zeros(M, N+1, 'single');
r2_hist  = zeros(M, N+1, 'single');
psi_hist = zeros(M, N+1, 'single');
incq     = zeros(M, N,   'single');

kill_step = zeros(M,1,'int32');
bpay      = zeros(M,1);

p1_hist(:,1)  = single(p1_fun(X));
r2_hist(:,1)  = single(r2_fun(X));
psi_hist(:,1) = single(psi_fun(X));

lastN = N;

for n = 1:N
    tn = t0 + (n-1)*dt;

    idx = find(alive);
    if isempty(idx)
        lastN = n - 1;
        for k = n:N
            p1_hist(:,k+1)  = p1_hist(:,k);
            r2_hist(:,k+1)  = r2_hist(:,k);
            psi_hist(:,k+1) = psi_hist(:,k);
        end
        break;
    end

    Xk = X(idx,:);

    % Full left-endpoint step for EVERY active label, including new exits.
    incq(idx,n) = single(dt*q_fun(tn,Xk));

    % ----- candidate increment -----
    if abs(s - 1.0) < 1e-14
        % alpha = 2, Brownian case
        Z = randn(numel(idx), d);
        step = sqrt(2*dt) * Z;
    else
        dS = subordinator_incr(dt, s, [numel(idx), 1]);
        Z  = randn(numel(idx), d);
        step = sqrt(2) .* (sqrt(dS) .* ones(1,d)) .* Z;
    end

    assert(all(isfinite(step(:))), 'Nonfinite stable increment. No tail cutoff is applied.');
    Xcand = Xk + step;

    inside = inD(Xcand);
    surv_loc = find(inside);
    died_loc = find(~inside);

    surv_idx = idx(surv_loc);
    died_idx = idx(died_loc);

    % Interior and exterior candidates share the same full-step source.
    X(idx,:) = Xcand;
    if ~isempty(died_idx)
        alive(died_idx) = false;
        kill_step(died_idx) = n;
        bpay(died_idx) = g_fun(tn+dt, Xcand(died_loc,:));
        % No more increments or sources are generated for these labels.
    end

    % ----- carry forward stored features -----
    p1_hist(:,n+1)  = p1_hist(:,n);
    r2_hist(:,n+1)  = r2_hist(:,n);
    psi_hist(:,n+1) = psi_hist(:,n);

    if ~isempty(surv_idx)
        p1_hist(surv_idx,n+1)  = single(p1_fun(X(surv_idx,:)));
        r2_hist(surv_idx,n+1)  = single(r2_fun(X(surv_idx,:)));
        psi_hist(surv_idx,n+1) = single(psi_fun(X(surv_idx,:)));
    end
end
end

% =========================================================================
% Terminal payoff initialization
% =========================================================================
function Y = terminal_payoff(psi_hist, kill_step, bpay, Afun, t0, dt, lastN)
M = size(psi_hist, 1);
Y = zeros(M,1);

aliveT = (kill_step == 0);
psiT = double(psi_hist(:, lastN+1));

if any(aliveT)
    Y(aliveT) = Afun(t0 + lastN*dt) .* psiT(aliveT);
end

died = ~aliveT;
if any(died)
    Y(died) = bpay(died);
end
end

% =========================================================================
% One-sided s-stable subordinator increment (0 < s < 1)
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
% CE: 2D bin smoother on (p1,r2)
% =========================================================================
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
p1lo = min(p1); p1hi = max(p1);
r2lo = min(r2); r2hi = max(r2);
if (p1hi-p1lo) < 1e-14 || (r2hi-r2lo) < 1e-14
    H = repmat(mean(y,1),m,1);
    return;
end
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

function [y,residual] = solve_implicit(H, dt, f, df, itMax, tol)
% Solve y-dt*f(y)=H. The bound |f'|<=1+e is global for this example.
% Newton is safeguarded by a contraction update if its residual increases.
H = H(:);
contraction = (1+exp(1))*dt;
assert(isscalar(dt) && dt > 0 && contraction < 1);
assert(itMax >= 1 && tol > 0 && all(isfinite(H)));
y = H;
for it = 1:itMax
    F = y-dt*f(y)-H;
    residual = norm(F,inf);
    if residual <= (1-contraction)*tol, return; end
    J = 1-dt*df(y);
    y_new = y-F./J;
    F_new = y_new-dt*f(y_new)-H;
    retry = ~isfinite(y_new) | ~isfinite(F_new) | abs(F_new) > abs(F);
    if any(retry)
        y_new(retry) = H(retry)+dt*f(y(retry));
    end
    y = y_new;
    if any(~isfinite(y))
        error('IBSDE:ScalarSolve','Nonfinite scalar iterate. Check dt and the source.');
    end
end
residual = norm(y-dt*f(y)-H,inf);
if residual > (1-contraction)*tol
    error('IBSDE:ScalarSolve','Residual %.3e exceeds tolerance. Increase SOLVE_MAXIT or reduce dt.',residual);
end
end
