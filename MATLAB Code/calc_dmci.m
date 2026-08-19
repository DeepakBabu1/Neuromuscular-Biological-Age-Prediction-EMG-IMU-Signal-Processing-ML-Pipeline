%% DMCI CODE
%
%dmci = 100+10( ((AVG_VAF_1sc) - (VAF_1se)) / (SD_VAF_1sc))
%
%where
% AVG_VAF_1sc = the average of the VAF of the one-synergy solution of the young adult control group
% VAF_1se = the VAF of the one-synergy solution for each individual in the experimental groups
% SD_VAF_1sc = the standard deviation of the VAF of the one-synergy solution of the young adult control group
% Steele et al. (2015)

function dmci = calc_dmci(VAF_controlgroup, VAF_allparticipants)

%walking
AVG_VAF_1sc = 85.933646
VAF_1se = 87.76443
SD_VAF_1sc =4.001535071

AVG_VAF_1sc = (sum(VAF_controlgroup))/length(VAF_controlgroup);
VAF_1se = VAF_allparticipants;
SD_VAF_1sc = std(VAF_controlgroup);

%dmci = 100+10*(((AVG_VAF_1sc) - (VAF_1se)) / (SD_VAF_1sc));

for i=1:length(VAF_allparticipants)
dmci(i) = 100+10*(((AVG_VAF_1sc) - (VAF_1se(i))) / (SD_VAF_1sc)); %dmci for each person's results....do we get the average dmci of each age group?

end