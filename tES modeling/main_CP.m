clear
clc
close all

%% Init and configuration
if ismember('D:\', pwd)      % working from C3N
    addpath(genpath('D:\OneDrive - Universidad Complutense de Madrid (UCM)\LNCC\LCCN _pipelines\RoastPipelines'))
    data_folder='D:\OneDrive - Universidad Complutense de Madrid (UCM)\LNCC\LCCN _data\';
else                           % working form laptop
    addpath(genpath('E:\OneDrive - Universidad Complutense de Madrid (UCM)\LNCC\LCCN _pipelines\RoastPipelines'))
    data_folder='E:\OneDrive - Universidad Complutense de Madrid (UCM)\LNCC\LCCN _data\';
end

cd(data_folder)

subjects=[35,49,50,58,59,64,65,71,75,77];
% protocols={'OzCz','P3P4','targetACC'}; % Define what models to calculate
protocols={'F3F4'};
padding=30;

%% Calculate Roast model
% Working on MATLAB 2016a; with GPD 3.3.0 in roast; and 30Gb RAM.

for i=1:length(subjects)
    
    subj_id=subjects(i);
    subj=strcat('NEMOS_0', int2str(subj_id));
    cd(subj)

    % Create main folder for Current Propagation models and add nifti
    created = exist(strcat(pwd,'\.roast'), 'dir');
    if created == 0
        mkdir .roast
        copyfile(strcat('anat/NEMOS-0',int2str(subj_id),'_3DT1.nii'), '.roast/') 
    end
    cd('.roast')

    wd=pwd;
    
    % Calculate current propagation model with Roast: OzCz situation
    if ismember('OzCz', protocols)
        currentProp_OzCz(strcat('NEMOS-0',int2str(subj_id),'_3DT1.nii'), wd, padding)
        cd(wd)
    
    % Calculate current propagation model with Roast: P3P4 situation
    % This disposition stimulates Precuneus, to desync it from ACC
    elseif ismember('P3P4', protocols)
        currentProp_P3P4(strcat('NEMOS-0',int2str(subj_id),'_3DT1.nii'), wd, padding)
        cd(wd)

    % Calculate current propagation model with Roast: P3P4 situation
    % This disposition stimulates Precuneus, to desync it from ACC
    elseif ismember('F3F4', protocols)
        currentProp_F3F4(strcat('NEMOS-0',int2str(subj_id),'_3DT1.nii'), wd, padding)
        cd(wd) 

    % Calculate current propagation model with Roast: target ACC situation
    % Optimize stimulation focality into ACC MNI coordinates
    elseif ismember('targetACC', protocols)
       currentProp_targetACC(subj_id, strcat('NEMOS-0',int2str(subj_id),'_3DT1.nii'), wd)
       cd(wd)
    end
    
    cd(data_folder)
    
end


