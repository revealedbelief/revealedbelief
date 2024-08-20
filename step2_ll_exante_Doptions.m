function [err,pLow,pHigh,V,vCond_L,vCond_H] = step2_ll_exante_Doptions(states,state_transition,xadata,prefparms,alpha) 

    % set seed so simulate same eps on each iteration
    rng(1)
    globals=step0a_set_globals;

    % get implied choice probs and value functions given parameter guess
    %----------------
    [pLow,pHigh,V,vCond_L,vCond_H] = step1a_solve_problem_Doptions(states,state_transition,prefparms,alpha); 
     

    % assign probabilities to individual observations 
    %--------------------------------------
    pr=NaN(size(xadata.ch));   
    N_states_not_T=size(states(states(:,1)<globals.T,:),1);  % number of states to sample from  
    for ss=1:N_states_not_T
        pr(xadata.state_no==ss & xadata.q==1,:)=repmat(pHigh(ss,:),size(pr(xadata.state_no==ss & xadata.q==1,:),1),1);
        pr(xadata.state_no==ss & xadata.q==0,:)=repmat(pLow(ss,:),size(pr(xadata.state_no==ss & xadata.q==0,:),1),1);
    end
    
    % calculate log likelihood function
    %--------------------------------------
    % set bounds so objective function always defined
    pr=max(pr,0.00001);  
    ll=log(sum(pr.*xadata.ch,2));    
    err=-sum(ll,1); 

end