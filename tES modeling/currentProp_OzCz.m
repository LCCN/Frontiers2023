function model = currentProp_OzCz(nifti_file, folder, padding)

% Create folder structure
created = exist(strcat(pwd,'roast_OzCzModel'), 'dir');
if created == 0
    mkdir roast_OzCzModel
end

specific_folder=strcat(folder,'\roast_OzCzModel\');
copyfile(nifti_file, 'roast_OzCzModel/') 
file_path=strcat(specific_folder, nifti_file);


% Init Current Propagation model processsing
cd 'C:\Users\jescab01\Toolbox\roast-3.0'
addpath(genpath(specific_folder))


% ROAST model for alpha peak rising experiment: Oz-Cz positions
roast(file_path,...
{'Oz',1.5,'Cz',-1.5},...
'capType','1010',...
'electype','pad',...
'elecsize',{[70 50 3],[70 50 3]},... % mm
'elecori',{'si','ap'},...
'resampling','on',...
'zeropadding',padding)



