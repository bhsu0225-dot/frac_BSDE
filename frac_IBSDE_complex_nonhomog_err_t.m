function frac_IBSDE_complex_nonhomog_err_t()

close all; clc;
try
    maxNumCompThreads(1);
catch
end

%% ================= USER PARAMETERS =================
d = 100; alpha = 1.5;
T = 50.0; t0 = 0.0; K = 101;
k1 = 1; k2 = 2;
beta = 0.3; eta = 0.7;
Rout = 2.0;
a_axes = Rout*ones(1,d);
a_axes(1) = 1.5*Rout; a_axes(2) = 2.0*Rout;
c_inner = zeros(1,d); c_inner(3) = 0.55*Rout;
r_in = 0.25*Rout;
N_start = 1000;
N_refine = 0:5;
N_list = N_start*2.^N_refine;
M_list = [1000 10000];       % Two particle counts, one curve each.
reference_height = 0.08;     % Reference at the COARSEST dt.
nRepeats = 1;               % Increase to assess Monte Carlo variability.
NBIN1 = 501; NBIN2 = 501; SMOOTHIT = 2;
NEWTON_IT = 20;           
usePar = true; numWorkers = 15;
seed = 10086;
figure_visible = 'on';
% Supplied feature map: (x_k1*x_k2, |x|^2).
kappa = 2;
outdir = fullfile(fileparts(mfilename('fullpath')),'output_complex_err_t');

%% ---------------- validation and exact data ----------------
assert(d>=3 && d==floor(d) && kappa==2);
assert(k1>=1 && k2>=1 && k1<=d && k2<=d && k1~=k2);
assert(k1==floor(k1) && k2==floor(k2));
assert(all(a_axes>0) && r_in>0 && beta>=0 && eta>=0);
assert(NEWTON_IT>=1 && NEWTON_IT==floor(NEWTON_IT));
assert(alpha>0 && alpha<=2 && T>t0 && K>=3 && K==floor(K));
assert(all(N_list>=1 & N_list==floor(N_list)));
assert(all(M_list>=1 & M_list==floor(M_list)));
assert(nRepeats>=1 && nRepeats==floor(nRepeats));
assert(NBIN1>=2 && NBIN2>=2 && NBIN1==floor(NBIN1) && NBIN2==floor(NBIN2));
assert(SMOOTHIT>=0 && SMOOTHIT==floor(SMOOTHIT));
assert(reference_height>0);
assert(numel(M_list)==2,'Specify two M values for the two comparison curves.');
num_scan = numel(N_list); num_curves = numel(M_list);
x_values = (T-t0)./N_list;
assert(num_scan>=2 && all(diff(N_list)>0));
reference = reference_height*x_values/x_values(1);
if ~exist(outdir,'dir'), mkdir(outdir); end
workers = start_parallel(usePar,numWorkers);
s = alpha/2;
Afun = @(t) exp(cos(t));
dA_dt = @(t) -sin(t).*exp(cos(t));
f = @(u) eta*(exp(beta*u)-u);
df = @(u) eta*beta*exp(beta*u)-eta;
b = d/2; a = d/2+s;
Cds = 2^(2*s)*gamma(a)/gamma(b)*(a*(a+1))/(b*(b+1));
assert(isfinite(Cds),'Nonfinite fractional-Laplacian coefficient.');
inD = @(X) in_domain_complex(X,a_axes,c_inner,r_in);
phi_fun = @(X) X(:,k1).*X(:,k2).*exp(-sum(X.^2,2));
p1_fun = @(X) X(:,k1).*X(:,k2);
r2_fun = @(X) sum(X.^2,2);
F1F1_neg = @(r2) kummer1f1_neg_series_vec(a+2,b+2,r2);
Ls_phi_fun = @(X) Cds*X(:,k1).*X(:,k2).*F1F1_neg(sum(X.^2,2));
u_exact_fun = @(t,X) Afun(t).*phi_fun(X);
g_fun = @(t,X) u_exact_fun(t,X);
q_fun = @(t,X) q_manufactured_signfix(t,X,Afun,dA_dt,phi_fun,Ls_phi_fun,f);
Rgrid = 0.99/sqrt(1/a_axes(k1)^2+1/a_axes(k2)^2);
avec = linspace(-Rgrid,Rgrid,K)';
X0 = zeros(K,d); X0(:,k1) = avec; X0(:,k2) = avec;
for kk = 1:K
    x = X0(kk,:); shrink = 1;
    while ~inD(shrink*x)
        shrink = 0.9*shrink;
        if shrink<1e-12
            error('Failed to place an initial point inside D. Adjust domain/hole.');
        end
    end
    X0(kk,:) = shrink*x;
end
u_ex_ref = u_exact_fun(t0,X0);
inside = inD(X0);

err = nan(num_curves,num_scan);
orders = nan(num_curves,num_scan);
err_runs = nan(num_curves,num_scan,nRepeats);
elapsed_all = nan(num_curves,num_scan);
max_residual_all = nan(num_curves,num_scan);
u_num_all = nan(K,num_scan,num_curves,nRepeats);
kill_rate_all = nan(K,num_scan,num_curves,nRepeats);
tag = sprintf('nonhomo_err_t_%dd_alpha%02d',d,round(10*alpha));

matfile = fullfile(outdir,[tag '.mat']);
assert(seed>=0 && seed+num_curves*num_scan*nRepeats*K<2^32);
fprintf('d=%d, alpha=%g, queries=%d (%d interior), repeats=%d\n', ...
    d,alpha,K,nnz(inside),nRepeats);
fprintf('Empirical RMS uses the exact solution at the interior queries. Reference lines are guides only.\n');

%% ---------------- compute every error from the solver ----------------
for icurve = 1:num_curves
    for iscan = 1:num_scan
        N = N_list(iscan); M = M_list(icurve);
        dt = (T-t0)/N;
        timer = tic; max_residual = 0;
        for rep = 1:nRepeats
            values = zeros(K,1); kills = zeros(K,1); residuals = zeros(K,1);
            run_id = ((icurve-1)*num_scan+(iscan-1))*nRepeats+(rep-1);
            parfor (kk = 1:K,workers)
                rng(seed+run_id*K+kk,'twister');
                [values(kk),kills(kk),residuals(kk)] = solve_one_point( ...
                    X0(kk,:),d,N,dt,s,M,inD,q_fun,g_fun,p1_fun,r2_fun, ...
                    phi_fun,Afun,t0,f,df,NBIN1,NBIN2,SMOOTHIT,NEWTON_IT);
            end
            u_num_all(:,iscan,icurve,rep) = values;
            kill_rate_all(:,iscan,icurve,rep) = kills;
            err_runs(icurve,iscan,rep) = sqrt(mean((values-u_ex_ref).^2));
            max_residual = max(max_residual,max(residuals));
        end
        err(icurve,iscan) = sqrt(mean(reshape(err_runs(icurve,iscan,:),[],1).^2));
        elapsed_all(icurve,iscan) = toc(timer);
        max_residual_all(icurve,iscan) = max_residual;
        j = iscan;
        if j>1 && err(icurve,j-1)>0 && err(icurve,j)>0
            orders(icurve,j) = log(err(icurve,j-1)/err(icurve,j))/log(x_values(j-1)/x_values(j));
        end
        fprintf('N=%d, dt=%.6g, M=%d: RMS=%.8e, order=%.4f, residual=%.2e, wall=%.2fs\n', ...
            N,dt,M,err(icurve,iscan),orders(icurve,iscan),max_residual,elapsed_all(icurve,iscan));
        if max_residual>1e-8
            warning('Implicit residual %.3e: Newton iteration or clipping may affect the measured error.',max_residual);
        end
        % Checkpoint after each completed (N,M) pair.
        save(matfile,'err','orders','err_runs','u_num_all','u_ex_ref','X0','avec','inside', ...
            'kill_rate_all','max_residual_all','elapsed_all','x_values','reference', ...
            'd','alpha','T','t0','K','N_list','M_list','nRepeats','kappa', ...
            'NBIN1','NBIN2','SMOOTHIT','NEWTON_IT','seed', ...
            'k1','k2','beta','eta','Rout','a_axes','c_inner','r_in', ...
            'reference_height');
    end
end

%% ---------------- plot in the supplied style ----------------
err_N_alpha = [err;reference]; % Third row is ONLY a reference, not computed error.

fig = figure('Color','w','Visible',figure_visible);
loglog(x_values,err(1,:),'Color',[0,0.45,0.74],'Marker','o', ...
    'MarkerSize',8,'MarkerFaceColor','w','LineStyle','-','LineWidth',2);
hold on
loglog(x_values,err(2,:),'Color',[0.85,0.33,0.10],'Marker','s', ...
    'MarkerSize',8,'MarkerFaceColor','w','LineStyle','-','LineWidth',2);
loglog(x_values,reference,'Color','black','Marker','*', ...
    'MarkerSize',8,'MarkerFaceColor','w','LineStyle','-','LineWidth',2);
xlim([min(x_values),max(x_values)]);
ylim('auto'); % Use the computed errors and reference curve to set the range.
legend_text = {sprintf('$M = %d$',M_list(1)), ...
    sprintf('$M = %d$',M_list(2)), '$\mathcal{O}(\Delta t)$'};
handle = legend(legend_text);
set(handle,'Interpreter','latex','FontSize',15,'Location','Northwest');
set(gca,'FontSize',18);
ylabel('$\|e^N\|_2$','Interpreter','latex','FontSize',18);
xlabel('$\Delta t$','Interpreter','latex','FontSize',18);
grid on
if any(~isfinite(err(:))) || any(err(:)<=0)
    warning('Nonpositive or nonfinite errors cannot be displayed on a logarithmic axis. Check the saved error values.');
end
savefig(fig,fullfile(outdir,[tag '.fig']));
print(fig,fullfile(outdir,[tag '.eps']),'-depsc2');
old_units = get(fig,'Units');
set(fig,'Units','inches');
fig_position = get(fig,'Position');
set(fig,'PaperUnits','inches','PaperSize',fig_position(3:4), ...
    'PaperPosition',[0 0 fig_position(3:4)]);
set(fig,'Units',old_units);
print(fig,fullfile(outdir,[tag '.pdf']),'-dpdf');
save(matfile,'err_N_alpha','-append');

fprintf('Saved FIG, EPS, PDF and MAT files in:\n%s\n',outdir);
end

function [u_num,kill_rate,max_residual] = solve_one_point(x0,d,N,dt,s,M, ...
    inD,q_fun,g_fun,p1_fun,r2_fun,phi_fun,Afun,t0,f,df, ...
    NBIN1,NBIN2,SMOOTHIT,NEWTON_IT)
[p1_hist,r2_hist,phi_hist,incq,kill_step,bpay,lastN] = ...
    forward_subBM_stop_complex_store(x0,d,N,dt,s,M,inD,q_fun,g_fun,p1_fun,r2_fun,phi_fun,t0);
kill_rate = mean(kill_step>0);
Y = terminal_payoff_from_hist(phi_hist,kill_step,bpay,Afun,t0,dt,lastN);
max_residual = 0;
for n = lastN:-1:1
    idx = find(kill_step==0 | kill_step>=n);
    if isempty(idx), continue; end
    targ = Y(idx)+double(incq(idx,n));
    H = ce_bin2d_sumcount(double(p1_hist(idx,n)),double(r2_hist(idx,n)), ...
        targ,NBIN1,NBIN2,SMOOTHIT);
    Y(idx) = newton_implicit_general(H,dt,f,df,NEWTON_IT);
    residual = abs(Y(idx)-dt*f(Y(idx))-H);
    if any(~isfinite(residual)) || any(~isfinite(Y(idx)))
        error('Nonfinite backward value or residual at level %d. Check dt and Newton settings.',n);
    end
    max_residual = max(max_residual,max(residual));
end
u_num = mean(Y);
end

function workers = start_parallel(usePar,numWorkers)
workers = 0;
if ~usePar, return; end
if isempty(ver('parallel')) || ~license('test','Distrib_Computing_Toolbox')
    warning('Parallel toolbox unavailable. Running serially.');
    return;
end
try
    pool = gcp('nocreate');
    if isempty(pool), pool = parpool('local',numWorkers); end
    workers = min(numWorkers,pool.NumWorkers);
    fprintf('Parallel enabled: %d workers.\n',workers);
catch err
    warning('Could not start parallel pool: %s. Running serially.',err.message);
end
end

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
