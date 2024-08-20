function [states] = step0b_setup_states

 globals=step0a_set_globals;

 % when the order matters
 %-------------------------

 states=NaN(globals.D+1,globals.T);

 % current t - t1 action - t2 action - ....
 states(1,1)=1;
 states(2:(globals.D+1),1)=2;
 states(2:(globals.D+1),2)=[1:(globals.D)]';

for t=3:globals.T
    for k=1:globals.D
        a=states(states(:,1)==t-1 & states(:,t-1)>0,:);
        a(:,t)=k;
        a(:,1)=t;
        states=[states; a];
    end
end

end