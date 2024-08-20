function [pLow,pHigh,V,vCond_L,vCond_H] = step1a_solve_problem_Doptions(states,state_transition,prefparms,alpha) 

    % set seed so simulate same shocks (eps) on each iteration
    rng(1)
    
    %-------------------------------------------------------------
    % Step 0: problem set up
    % - NB. states an input into function
    %-------------------------------------------------------------
    
    globals=step0a_set_globals;

    Nstates=size(states,1);

    % utility
    [u] = step0d_utility_f(states,prefparms);

    % beliefs
    sigma_ea=alpha(1);
    beliefparms=alpha(2:end);
    [pi] = step0e_makepi(states,beliefparms)';
    
    % initialise choice prob matrix - defined for each state x choice
    % --- separate matrices for if get high vs. low offer  
    pHigh=NaN(size(states,1),globals.D+1);
    pLow=NaN(size(states,1),globals.D+1);


    % initialise value function matrices
    % --- value of being unmarried at start of period state before know 
    %     type of marriage offer that materialised
    V=NaN(size(states,1),1);    
    % --- choice-specific conditional value functions
    %     v(q,state,choice) -- defined in equations (8) and (9)
    %     columns: conditional on choosing each action from d=1:D, d=0
    vCond_H=NaN(size(states,1),globals.D+1);
    vCond_L=NaN(size(states,1),globals.D+1);
    
    %-------------------------------------------------------------
    % Step 1: calculate conditional value function at T
    %-------------------------------------------------------------
    % for non-terminal actions = u(T,L)+Vbar
    vCond_H(states(:,1)==globals.T,:)=repmat(u(states(:,1)==globals.T,1)+prefparms.Vbar,1,globals.D+1);
    vCond_L(states(:,1)==globals.T,:)=repmat(u(states(:,1)==globals.T,1)+prefparms.Vbar,1,globals.D+1);
    
    % for terminal actions (last column) = u
    vCond_H(:,globals.D+1)=u(:,2);
    vCond_L(:,globals.D+1)=u(:,1)   ; 

    %-------------------------------------------------------------
    % Step 2: simulate errors to calculate choice probabilities 
    %-------------------------------------------------------------
    mu=zeros(1,globals.D+1);
    cov=diag(sigma_ea^2.*ones(1,globals.D+1));
    eps=mvnrnd(mu,cov,globals.Nsims);

    %-------------------------------------------------------------
    % Step 3: loop recursively through states to solve for value function 
    %-------------------------------------------------------------
    for ss=Nstates:-1:1
     
        % create indicator to mark choice we are interested in
        ind=zeros(Nstates,1);
        ind(ss)=1;
        a=states(ind==1,1); % age at state - need this for discounting
        
        %------------------------------------------------------------------
        % Step 3a: Choice probabilities conditional on low & high q 
        %------------------------------------------------------------------
        
        % --- shock mean & covariance for D+1 way choice 
        mu=zeros(1,globals.D);
        cov=sigma_ea^2.*ones(globals.D,globals.D)+diag(sigma_ea^2.*ones(1,globals.D));

        for d=1:(globals.D+1)
            % take the difference between the value of option k & other options
            % --- when q=H
            delta=vCond_H(ind==1,d)-vCond_H(ind==1,[1:(d-1) (d+1):(globals.D+1)]);
            pHigh(ind==1,d)=mvncdf([delta], mu , (globals.disc.^(a))^2.*cov);

            % --- when q=L
            delta=vCond_L(ind==1,d)-vCond_L(ind==1,[1:(d-1) (d+1):(globals.D+1)]);
            pLow(ind==1,d)=mvncdf([delta], mu , (globals.disc.^(a))^2.*cov);
        end
        
        %------------------------------------------------------------------
        % Step 3b: Form value functions
        %------------------------------------------------------------------
          
        if globals.D==1
            % when D=1, we can calculate this analytically which is a bit
            % faster plus more accurate
            z0=(vCond_L(ind==1,1)-vCond_L(ind==1,2))./(globals.disc^(a)); % val(reject)- val(accept) | q=Low
            z1=(vCond_H(ind==1,1)-vCond_H(ind==1,2))./(globals.disc^(a)); % val(reject)- val(accept) | q=High           

            % expected preference shock conditional on action being optimal
            E_eps_a_0=(globals.disc^(a)).*(1./(2.^0.5)).*sigma_ea.*normpdf(z0./(2.*sigma_ea.^2).^0.5)./(1-normcdf((z0./(2.*sigma_ea.^2).^0.5)));
            E_eps_a_1=(globals.disc^(a)).*(1./(2.^0.5)).*sigma_ea.*normpdf(z1./(2.*sigma_ea.^2).^0.5)./(1-normcdf((z1./(2.*sigma_ea.^2).^0.5)));
            E_eps_r_0=(globals.disc^(a)).*(1./(2.^0.5)).*sigma_ea.*normpdf(-z0./(2.*sigma_ea.^2).^0.5)./(1-normcdf((-z0./(2.*sigma_ea.^2).^0.5)));
            E_eps_r_1=(globals.disc^(a)).*(1./(2.^0.5)).*sigma_ea.*normpdf(-z1./(2.*sigma_ea.^2).^0.5)./(1-normcdf((-z1./(2.*sigma_ea.^2).^0.5)));

            % compute value function
            if a>1
            V(ind==1)=...
                (1-pi(ind==1)).*pLow(ind==1,2).*(vCond_L(ind==1,2)+E_eps_a_0)... % low offer, accept
                + pi(ind==1).*pHigh(ind==1,2).*(vCond_H(ind==1,2)+E_eps_a_1)... % high offer, accept
                + (1-pi(ind==1)).*pLow(ind==1,1).*(vCond_L(ind==1,1)+E_eps_r_0)... % low offer, reject
                + pi(ind==1).*pHigh(ind==1,1).*(vCond_H(ind==1,1)+E_eps_r_1); % high offer, reject

                % fill in vCond_q for each non-terminal action
                [state_t,action]=find(state_transition==ss);
                vCond_H(state_t,action)=V(ind==1);
                vCond_L(state_t,action)=V(ind==1);

            end

        end

        if globals.D>1

        
            % form complete utility inc. shocks for q=H and q=L
            total_High=vCond_H(ind==1,:)+eps.*globals.disc^a;
            total_Low=vCond_L(ind==1,:)+eps.*globals.disc^a;
            
            % maximum utility option on each shock draw 
            max_total_High=max(total_High,[],2);
            max_total_Low=max(total_Low,[],2);
            
            if a>1
                 V(ind==1)= ...
                     pi(ind==1).*mean(max_total_High,1) + (1-pi(ind==1)).*mean(max_total_Low,1);
              
                % fill in conditional value func for each non-terminal action
                [state_t,action]=find(state_transition==ss);
                vCond_H(state_t,action)=V(ind==1);
                vCond_L(state_t,action)=V(ind==1);
        
            end

        end

   end

end

  