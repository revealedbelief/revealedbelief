%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SIMULATION & ESTIMATION CODE 
% "Revealed Beliefs & the Marriage Market Return to Education"
%  Alison Andrew & Abi Adams
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% We provide example code to apply the "revealed beliefs" methodology in
% practice. Our example code follows in 4 parts; 

% 0) Setup and examine the basic objects we need for this problem e.g.
% states, state transitions, payoff matrices
%
% 1a) Solve the problem for a particular "true" parameter vector containing
% both Sigma and the beliefs
%
% 1b) Use the implied choice probabilities from this solution to simulate
% experimental data for a revealed belief experiment
%
% 2) Use maximum likelihood estimation to estimate both Sigma and the
% belief parameters and compare to the truth
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
clear; clc;

%% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% STEP 0: setup states, utility and state transitions
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

% set globals
% --- edit the following within this function: T, D, discount factor
globals=step0a_set_globals;                                                        % <-- EDIT FUNCTION TO CHANGE T, D, BETA, Nsims

% set states 
% --- these are all the decision states. They are defined by the
%     time period (col 1) and then the complete history of the choices made in each
%     past period (cols 2:end). 
states=step0b_setup_states;                                                        % <-- EDIT FUNCTION TO CHANGE STATE DEFINITIONS
Nstates=size(states,1);

% -- table of states 
tab_statenumbers=array2table([1:Nstates]','VariableNames', {'State #'});
state_hist_names=[{'State -- t'},arrayfun(@(x) sprintf('State -- d%d',x),1:(globals.T-1),'UniformOutput',false)];
tab_statedescription=array2table(states,'VariableNames',state_hist_names);
[tab_statenumbers,tab_statedescription]


% state transition 
% -- for each state x action, give rownumber of state action implies at t+1
state_transition=step0c_setup_state_transition;                                    % <-- EDIT FUNCTION TO CHANGE STATE TRANSITIONS

% -- table of transitions 
state_transition_names=[arrayfun(@(x) sprintf('Transition (dt=%d)',x),1:(globals.D),'UniformOutput',false)];
tab_statetransitions=array2table(state_transition,'VariableNames',state_transition_names);
[tab_statenumbers tab_statedescription tab_statetransitions]

% setup utility lookup matrix for completed paths
prefparms=step0d_set_prefparms;
[u] = step0d_utility_f(states,prefparms);                                        % <-- EDIT FUNCTION TO CHANGE FUNCTIONAL FORM OF U()

% -- table of utility numbers
tab_terminalpayoffs=array2table(u,'VariableNames',{'Payoff -- U(q=L)', 'Payoff -- U(q=H)'});
[tab_statenumbers tab_statedescription tab_terminalpayoffs]


% set the "true" belief parameters & shock variance  
% -- this is only for simulation purposes as in real applications will be 
%    estimated
%    true_parms =[sigma, pi_{2},..... pi_{Nstates}]

true_parms =     [0.6, 0.8, 0.7, 0.6, 0.5, 0.5, 0.2];
if size(true_parms,2)~=Nstates
    error('Parameter vector must be of size 1 x Nstates. The first element corresponds to sigma. The rest contain pi_{2},... pi_{Nstates}.')
end
pi=step0e_makepi(states,true_parms(2:end))';                                       % <-- EDIT FUNCTION TO CHANGE FUNCTIONAL FORM OF BELIEFS

% -- table of probabilties
tab_pi=array2table(pi,'VariableNames',{'pi -- Pr(q=H)'})
[tab_statenumbers tab_statedescription tab_pi]

% For simulation purposes, set number of respondents & 
% number of experiments/respondent for simulations
n_resps=8000; % number of individuals
n_exps=5; % number of experiments                                               % <-- EDIT HERE TO CHANGE NUMBER OF ROUNDS

%% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% STEP 1: Simulate choice data 
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% for a given set of belief parms (pi), var of pref shocks (sigma) and
% prefs, solve model to give choice probabilies and value functions

% step 1a: solve model given set of prefs, belief parms (pi) & var of shocks (sigma) 
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
[pLow,pHigh,V,vCond_L,vCond_H] = step1a_solve_problem_Doptions(states,state_transition,prefparms,true_parms); 

% --- table for values conditional on q=L
table(states,vCond_L,pLow)
% --- table for values conditional on q=H
table(states,vCond_H,pHigh)

% step 1b) Simulate some experimental data using these choice probabilities 
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% randomly sample from the decision states with t<T

N=n_exps*n_resps; % no experimental rounds x respondents 
simdata.uniqueid=[1:N]';
N_states_not_T=size(states(states(:,1)<globals.T,:),1);  % number of states to sample from states(:,1)<globals.T

% sample equally from the states
rng(1234)
simdata.state_no=1+floor(N_states_not_T.*rand(N,1));
simdata.q=1.*(rand(N,1)>0.5); % give 50% rounds q=H

simdata.t=NaN(N,1);
simdata.pr=NaN(N,globals.D+1);

for ss=1:N_states_not_T
    % time period of randomly selected state
    simdata.t(simdata.state_no==ss)=states(ss,1);
    
    % choice probabilities at states (also given simulated low or high q)
    simdata.pr(simdata.state_no==ss & simdata.q==1,:)=repmat(pHigh(ss,:),size(simdata.pr(simdata.state_no==ss & simdata.q==1,:),1),1);
    simdata.pr(simdata.state_no==ss & simdata.q==0,:)=repmat(pLow(ss,:),size(simdata.pr(simdata.state_no==ss & simdata.q==0,:),1),1);
end

% given choice probs, simulate realised choices
simdata.cumulpr(:,1)=simdata.pr(:,1);
for d=2:(globals.D+1)
    simdata.cumulpr(:,d)=simdata.cumulpr(:,d-1)+simdata.pr(:,d);
end
rand2=rand(N,1);
simdata.ch(:,1)=rand2<simdata.cumulpr(:,1);
for d=2:(globals.D+1)
    simdata.ch(:,d)=(simdata.cumulpr(:,d-1)<rand2) & (rand2<=simdata.cumulpr(:,d));
end

%% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% STEP 2: Estimate Beliefs on Simulated Data
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

% check the ll function at the starting values
starting_vals=  [0.5.*ones(Nstates,1)]';
[err,pLow,pHigh,V,vCond_L,vCond_H] = step2_ll_exante_Doptions(states,state_transition,simdata,prefparms,starting_vals); 

% optimise using fminunc
options = optimoptions('fminunc','Display','iter');
fun=@(alpha0) step2_ll_exante_Doptions(states,state_transition,simdata,prefparms,alpha0); 

[x,~,~,~,~,Hessian]  = fminunc(fun,starting_vals,options);
estimate=x';
fmincon_estimate=x';
se=sqrt(diag(inv(Hessian)));
ci_lo=estimate-1.96*se;
ci_hi=estimate+1.96*se;

true_parms=true_parms';
Tunc=table(true_parms,estimate,se,ci_lo,ci_hi)

% % 2b*) Robustness to particleswarm optimizer 
% ub= [(1-eps).*ones(Nstates,1)]';
% lb= [eps.*ones(Nstates,1)]';
% nvars=size(lb,2);
% options = optimoptions('particleswarm','InitialSwarmMatrix',starting_vals,'UseParallel',true,'SwarmSize',200,'Display','iter');
% [ps_estimate,f] = particleswarm(fun,nvars,lb,ub,options);
% ps_estimate=ps_estimate';
% Tunc=table(true_parms,ps_estimate)



