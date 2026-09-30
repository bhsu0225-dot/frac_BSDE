function frac_IBSDE_ball_semilinear_err_t()
close all; clc;
try
    maxNumCompThreads(1);
catch
end

%% ================= USER PARAMETERS =================
d = 100; alpha = 2.0;
T = 1.0; t0 = 0.0; K = 101;
N_start = 500;
N_refine = 0:5;
N_list = N_start*2.^N_refine;
M_list = [10000 20000];       % Two particle counts, one curve each.
reference_height = 0.08;     % Reference at the COARSEST dt, as in your figure.
nRepeats = 1;               % Increase to assess Monte Carlo variability.
NBIN1 = 500; NBIN2 = 500; SMOOTHIT = 2;
SOLVE_MAXIT = 200; SOLVE_TOL = 1e-12;
usePar = true; numWorkers = 15;
seed = 20260306;
figure_visible = 'on';
% This ball experiment uses the kappa=2 feature map from the manuscript.
kappa = 2;
outdir = fullfile(fileparts(mfilename('fullpath')),'output_ball_err_t');

%% ---------------- validation and exact data ----------------
assert(d>=2 && d==floor(d) && kappa==2);
assert(alpha>0 && alpha<=2 && T>t0 && K>=3 && K==floor(K));
assert(all(N_list>=1 & N_list==floor(N_list)));
assert(all(M_list>=1 & M_list==floor(M_list)));
assert(nRepeats>=1 && nRepeats==floor(nRepeats));
assert(NBIN1>=2 && NBIN2>=2 && NBIN1==floor(NBIN1) && NBIN2==floor(NBIN2));
assert(SMOOTHIT>=0 && SMOOTHIT==floor(SMOOTHIT));
assert(all((1+exp(1))*(T-t0)./N_list<1),'Increase N: need (1+e)*dt<1.');
assert(reference_height>0);
assert(numel(M_list)==2,'Specify two M values for the two comparison curves.');
num_scan = numel(N_list); num_curves = numel(M_list);
x_values = (T-t0)./N_list;
assert(num_scan>=2 && all(diff(N_list)>0));
reference = reference_height*x_values/x_values(1);
if ~exist(outdir,'dir'), mkdir(outdir); end
workers = start_parallel(usePar,numWorkers);
s = alpha/2;
Afun = @(t) cos(t)./(1+10*t.^2);
dA_dt = @(t) ((-sin(t)).*(1+10*t.^2)-20*t.*cos(t))./(1+10*t.^2).^2;
f = @(u) sin(u)+exp(cos(u));
df = @(u) cos(u)-sin(u).*exp(cos(u));
Cda = 2^alpha*gamma(1+alpha/2)*gamma((d+alpha)/2)/gamma(d/2);
assert(isfinite(Cda),'Nonfinite fractional-Laplacian coefficient.');
inD = @(X) sum(X.^2,2)<1;
psi_fun = @(X) max(1-sum(X.^2,2),0).^(alpha/2);
u_exact_fun = @(t,X) Afun(t).*psi_fun(X);
g_fun = @(t,X) zeros(size(X,1),1);
q_fun = @(t,X) q(t,X,Afun,dA_dt,alpha,Cda,f);
p1_fun = @(X) sum(X,2)/sqrt(d);
r2_fun = @(X) sqrt(max(sum(X.^2,2)-p1_fun(X).^2,0));
avec = linspace(-1.5/sqrt(d),1.5/sqrt(d),K)';
X0 = avec*ones(1,d);
u_ex_ref = u_exact_fun(t0,X0);
inside = inD(X0);

err = nan(num_curves,num_scan);
orders = nan(num_curves,num_scan);
err_runs = nan(num_curves,num_scan,nRepeats);
elapsed_all = nan(num_curves,num_scan);
max_residual_all = nan(num_curves,num_scan);
u_num_all = nan(K,num_scan,num_curves,nRepeats);
kill_rate_all = nan(K,num_scan,num_curves,nRepeats);
tag = sprintf('homo_err_t_%dd_alpha%02d',d,round(10*alpha));

matfile = fullfile(outdir,[tag '.mat']);
assert(seed>=0 && seed+num_curves*num_scan*nRepeats*K<2^32);
fprintf('d=%d, alpha=%g, queries=%d (%d interior), repeats=%d\n', ...
    d,alpha,K,nnz(inside),nRepeats);
fprintf('Empirical RMS includes prescribed exterior points. Reference lines are guides only.\n');

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
                    psi_fun,Afun,t0,f,df,NBIN1,NBIN2,SMOOTHIT,SOLVE_MAXIT,SOLVE_TOL);
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
        % Checkpoint after each completed (N,M) pair.
        save(matfile,'err','orders','err_runs','u_num_all','u_ex_ref','X0','avec','inside', ...
            'kill_rate_all','max_residual_all','elapsed_all','x_values','reference', ...
            'd','alpha','T','t0','K','N_list','M_list','nRepeats','kappa', ...
            'NBIN1','NBIN2','SMOOTHIT','SOLVE_MAXIT','SOLVE_TOL','seed', ...
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
    inD,q_fun,g_fun,p1_fun,r2_fun,psi_fun,Afun,t0,f,df, ...
    NBIN1,NBIN2,SMOOTHIT,SOLVE_MAXIT,SOLVE_TOL)
max_residual = 0;
if ~inD(x0)
    u_num = g_fun(t0,x0); kill_rate = 1;
    return;
end
[p1_hist,r2_hist,psi_hist,incq,kill_step,bpay,lastN] = ...
    forward_subBM(x0,d,N,dt,s,M,inD,q_fun,g_fun,p1_fun,r2_fun,psi_fun,t0);
kill_rate = mean(kill_step>0);
Y = terminal_payoff(psi_hist,kill_step,bpay,Afun,t0,dt,lastN);
for n = lastN:-1:1
    idx = find(kill_step==0 | kill_step>=n);
    if isempty(idx), continue; end
    targ = Y(idx)+double(incq(idx,n));
    p1n = double(p1_hist(idx,n));
    r2n = double(r2_hist(idx,n));
    H = ce_bin2d_sumcount(p1n,r2n,targ,NBIN1,NBIN2,SMOOTHIT);
    [Y(idx),residual] = solve_implicit(H,dt,f,df,SOLVE_MAXIT,SOLVE_TOL);
    max_residual = max(max_residual,residual);
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
% Match the degenerate-range branch described in the current manuscript.
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
