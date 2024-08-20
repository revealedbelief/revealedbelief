function [u] = step0d_utility_f(states,prefparms)

    % U=(Good groom==1) + gamma * Sum(state==1) + kappa * Sum(state==2)
    u=zeros(size(states,1),2);
    u(:,1)=0;
    u(:,2)=1;
    
    u(:,1)=u(:,1)+sum(states(:,2:end)==1,2).*prefparms.gamma+sum(states(:,2:end)==2,2).*prefparms.kappa;
    u(:,2)=u(:,2)+sum(states(:,2:end)==1,2).*prefparms.gamma+sum(states(:,2:end)==2,2).*prefparms.kappa;
    
end

