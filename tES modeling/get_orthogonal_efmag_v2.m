function txt_files = get_orthogonal_efmag_v2(subj, protocol, anat_folder, padding)

addpath(genpath('D:\Toolbox\iso2mesh-1.9.6'))
% add roast to path (we need NIFTI toolbox).
addpath('D:\Toolbox\roast-3.0\lib\NIFTI_20110921')

created = exist(strcat(pwd,'/orthogonalization_v2'), 'dir');
if created == 0
    mkdir orthogonalization_v2
end

% Load roast results: ef_all, ef_mag, vol_all.
disp('Loading current propagation model...')
load(dir('*Result.mat').name);
if contains(protocol, 'target')
   ef_all = r.ef_all; 
   ef_mag = r.ef_mag;
end

% Read white matter nifti
disp('Generating white matter mesh...')
nii=load_untouch_nii(dir('c2*_T1orT2.nii').name);
athreshold=graythresh(nii.img)*100; % automated thresholding with otsu method
if subj == 'NEMOS_071'
    disp('White matter thresholding exception for NEMOS 071')
    [datamesh.node, datamesh.face]=v2s(nii.img,athreshold-10,2); % thresholding exception
else
    [datamesh.node, datamesh.face]=v2s(nii.img,athreshold,2); % use iso2mesh; (3Dmatrix, threshold, ¿?) 
end

% plotmesh(datamesh.node,datamesh.face) % plot

mesh=triangulation(datamesh.face(:,1:3),datamesh.node); % Load triangulation
% saveas(trimesh(mesh),'orthogonalization_v2/0-mesh.png') % mesh
% close all

datamesh.centres=incenter(mesh); % Calculate centres of normals
datamesh.normvec=faceNormal(mesh); % Calculate normals to faces

% Invert normal vectors if necessary (pointing inward the brain). 
% ¿? v2s (volume2surface) function is not consistent with triangles 
% vertex order. faceNormal is consistent and will follow right hand rule.
[~,I]=maxk(datamesh.centres(:,3), 1000); % choose 1000 highest mesh triangles
if sum(datamesh.normvec(I,3))>0 % If they point upwards: invert them
    datamesh.normvec=-datamesh.normvec;
end

% Plot mesh with normal vectors drawn
fig=trimesh(mesh, 'FaceAlpha', 0.2);
hold on  
fig=quiver3(datamesh.centres(:,1),datamesh.centres(:,2),datamesh.centres(:,3), ...
     datamesh.normvec(:,1),datamesh.normvec(:,2),datamesh.normvec(:,3),'color','r');
title('Normal to white matter surface Vectors')

savefig('orthogonalization_v2/1-mesh_normals')
saveas(fig,'orthogonalization_v2/1-mesh_normals.png')
close all
 

%% Calculate ef_mag per mesh triangle

% Appoximate each center to a voxel
datamesh.voxel = round ( datamesh.centres );

% Find points at 0 index in any dimension to not consider them further
indexes=find(datamesh.voxel(:,1)==0 | datamesh.voxel(:,2)==0 | datamesh.voxel(:,3)==0);
% Save ef_vec per centre; make dot product (norm * ef_vec) to obtain magnitude
for i=1:size(datamesh.voxel, 1)
    if ismember(i, indexes)
        ef_vec=[0 0 0];
    else
        ef_vec = ef_all(datamesh.voxel(i,1),datamesh.voxel(i,2),datamesh.voxel(i,3),:);
    end
    datamesh.corref_vecs(i,:)=[ef_vec(1),ef_vec(2),ef_vec(3)];
    datamesh.ef_mag(i,1)=norm(datamesh.corref_vecs(i,:));
    datamesh.efnorm_mag(i,1)=dot(squeeze(ef_vec), datamesh.normvec(i,:));
end


fig=trimesh(mesh, 'FaceAlpha', 0.2);
hold on  
quiver3(datamesh.centres(:,1),datamesh.centres(:,2),datamesh.centres(:,3), ...
     datamesh.normvec(:,1),datamesh.normvec(:,2),datamesh.normvec(:,3),'color','r');
hold on
fig=quiver3(datamesh.centres(:,1), datamesh.centres(:,2),datamesh.centres(:,3), datamesh.corref_vecs(:,1), datamesh.corref_vecs(:,2), datamesh.corref_vecs(:,3), 'color','b');
fig.AutoScaleFactor=3;
title('Normal to wm surface (RED);  EF vectors (BLUE)')

savefig('orthogonalization_v2/2-mesh_normals_efvec.fig')
saveas(fig, 'orthogonalization_v2/2-mesh_normals_efvec.png')
close all

fig=trisurf(mesh,datamesh.efnorm_mag(:,1), 'EdgeColor','None');
title('Electric field magnitude on the WM surface')
savefig('orthogonalization_v2/3-efnorm_mag_perTriangle.fig')
saveas(fig, 'orthogonalization_v2/3-efnorm_mag_perTriangle.png')
close all


%% CHECK: not needed. Create efnorm_mag 3D volume from datamesh
% Get datamesh centres into a voxel space
efnorm_mag = zeros(size(ef_mag)); % Pre-allocate the new volume

% Unique as more than one mesh centre could correpond to a voxel
uniques_data=[];
[uniques_data.voxel, i_datamesh] = unique(datamesh.voxel, 'rows'); 
uniques_data.efnorm_mag = datamesh.efnorm_mag(i_datamesh);
uniques_data.centres = datamesh.centres(i_datamesh);
uniques_data.normvec = datamesh.normvec(i_datamesh);
uniques_data.corref_vecs = datamesh.corref_vecs(i_datamesh);

% Average multiple values per voxel
mask = true(size(datamesh.voxel,1), 1);
mask(i_datamesh) = false; % i_datamesh = non repeated
duplicated = unique(datamesh.voxel(mask,:), 'rows');

% Check on the duplication situations
report_duplicated = [];
report_duplicated.centres=[];
report_duplicated.normvec=[];
report_duplicated.corref_vecs=[];
report_duplicated.color=[];

% Takes 1 min approx. Look for more efficient solutions
for ia=1:length(duplicated)
    
    % Create a mask for efnorm_mag centres asociated to the voxel
    mask_datamesh = ismember(datamesh.voxel, duplicated(ia, :), 'rows');
    
    % Gather report vals
    report_duplicated.centres = [ report_duplicated.centres; datamesh.centres(mask_datamesh,:) ];
    report_duplicated.normvec = [ report_duplicated.normvec; datamesh.normvec(mask_datamesh,:) ];
    report_duplicated.corref_vecs = [ report_duplicated.corref_vecs; datamesh.corref_vecs(mask_datamesh,:) ];
     % random color
    report_duplicated.color = [ report_duplicated.color; rand(1, 3) .* ones(sum(mask_datamesh),3) ];  
   
    % Average efnorm_mags and save 
    avg = mean(datamesh.efnorm_mag(mask_datamesh));
    
    mask_unique = ismember(uniques_data.voxel, duplicated(ia, :), 'rows');
    uniques_data.efnorm_mag(mask_unique) = avg; 
    
end

% Check report
fig=trimesh(mesh, 'FaceAlpha', 0.15, 'FaceColor', 'g');
hold on 
scatter3(report_duplicated.centres(:,1),report_duplicated.centres(:,2),report_duplicated.centres(:,3), 50, report_duplicated.color, 'filled', 'MarkerEdgeColor','k')
hold on  
quiver3(report_duplicated.centres(:,1),report_duplicated.centres(:,2),report_duplicated.centres(:,3), ...
     report_duplicated.normvec(:,1),report_duplicated.normvec(:,2),report_duplicated.normvec(:,3),'color','r');
hold on
fig=quiver3(report_duplicated.centres(:,1), report_duplicated.centres(:,2),report_duplicated.centres(:,3), ...
    report_duplicated.corref_vecs(:,1), report_duplicated.corref_vecs(:,2), report_duplicated.corref_vecs(:,3), 'color','b');
title('non-needed check. Multiple centres per voxel(colorcoded in circles)')

savefig('orthogonalization_v2/4check-duplicated_inVoxel_normals_efvec.fig')
saveas(fig, 'orthogonalization_v2/4check-duplicated_inVoxel_normals_efvec.png')
close all


% Fill the new volume with the uniques and averaged data 
% uniques_data.sub2ind = sub2ind(size(efnorm_mag), uniques_data.voxel(:,1), uniques_data.voxel(:,2), uniques_data.voxel(:,3));
% efnorm_mag ( uniques_data.sub2ind ) = uniques_data.efnorm_mag;


%% Give ROI name to mesh triangle using Brainstom volatlas

% Load roi segmentation data
atlas = load(strcat(anat_folder,'subjectimage_AAL2_volatlas.mat'));
Cube = atlas.Cube;
roi_names = atlas.Labels(2:end, 2);

% Check number of rois
rois = unique(Cube);
rois = rois(2:end); % delete 0 --> non tissue (background)

% Padding as in roast model
Cube_padded=padarray(Cube,[padding-3 padding padding-3],0);

% Save roi per centre; 
for i=1:size(datamesh.voxel, 1)
    if ismember(i, indexes)
        datamesh.roi(i,1)=0;
    else
        datamesh.roi(i,1)=Cube_padded(datamesh.voxel(i,1),datamesh.voxel(i,2),datamesh.voxel(i,3));
    end
end

% Plot rois on mesh
fig=trisurf(mesh, datamesh.roi, 'EdgeColor','None');
title('AAL parcellation of subjects wm mesh')
savefig('orthogonalization_v2/5-roisOnSubjects.fig')
saveas(fig, 'orthogonalization_v2/5-roisOnSubjects.png')
close all

% Check a couple of regions: ACCl(35), OCCinfl(57), Prr(72), FSr(20)
mask = find(datamesh.roi == 35 | datamesh.roi == 36 | datamesh.roi == 37 | datamesh.roi == 39 | datamesh.roi == 35 | datamesh.roi == 72 |  datamesh.roi == 45 | datamesh.roi == 46 );
submesh=triangulation(datamesh.face(mask,1:3),datamesh.node); % Load triangulation

fig=trisurf(submesh, datamesh.efnorm_mag(mask,:), 'EdgeAlpha',0.2, 'FaceAlpha', 0.9);
hold on
quiver3(datamesh.centres(mask,1),datamesh.centres(mask,2),datamesh.centres(mask,3), ...
     datamesh.normvec(mask,1),datamesh.normvec(mask,2),datamesh.normvec(mask,3),'color','r','AutoScaleFactor',0.3); %norm in red
hold on
quiver3(datamesh.centres(mask,1), datamesh.centres(mask,2),datamesh.centres(mask,3), ...
    datamesh.corref_vecs(mask,1), datamesh.corref_vecs(mask,2), datamesh.corref_vecs(mask,3), 'color','b','AutoScaleFactor',0.5); % ef_mag in blue
title('Rois subset: ACC_l, FrontalSup_r, Pr_r, OCCmid_l; efnorm_mag as mesh color.')
savefig('orthogonalization_v2/6check-roiSelectionSubmesh_normals_efvec.fig')
saveas(fig, 'orthogonalization_v2/6check-roiSelectionSubmesh_normals_efvec.png')

fig2 = figure(2);
trisurf(submesh, datamesh.efnorm_mag(mask,:), 'EdgeAlpha',0.2, 'FaceAlpha', 0.9);
title('Rois subset: ACC_l, FrontalSup_r, Pr_r, OCCinf_l; efnorm_mag as mesh color.')
savefig(fig2, 'orthogonalization_v2/6check-roiSelectionSubmesh.fig')
saveas(fig2, 'orthogonalization_v2/6check-roiSelectionSubmesh.png')

close all


%% Get Average electric field magnitude per ROI
ROIvals = struct('efnorm_mags', [], 'avg_efnorm_mag', zeros(numel(rois),1));

% Pre-define figures to plot histograms with distributions of efnorm_mag
fig_hist_rh=figure('Position', [50, 50, 1800, 900]);
title('Right hemisphere rois')
fig_hist_lh=figure('Position', [50, 60, 1800, 900]);
title('Left hemisphere rois')
lim_x=0.2; % lim_y=50;

for iroi = 1:numel(rois)
    
    mask = find(datamesh.roi == rois(iroi));
    
    ROIvals(iroi).efnorm_mags = datamesh.efnorm_mag(mask);
    ROIvals(iroi).avg_efnorm_mag = mean(datamesh.efnorm_mag(mask));
        
    if rem(iroi,2)==0 && iroi<90 % Work on right hemisphere
        figure(fig_hist_rh)
        subplot(5,9,iroi/2);
        
        histogram(ROIvals(iroi).efnorm_mags)
        
        xlim([-lim_x,lim_x])
%         ylim([0,lim_y])
        if ROIvals(iroi).avg_efnorm_mag < 0
            xline(ROIvals(iroi).avg_efnorm_mag, 'r')
        else
            xline(ROIvals(iroi).avg_efnorm_mag, 'g')
        end
        title(strcat(roi_names{iroi}, ' = ', num2str(ROIvals(iroi).avg_efnorm_mag)))
    
    elseif iroi<90  % Work on left hemisphere
        figure(fig_hist_lh)
        subplot(5,9,iroi/2+0.5);
        
        histogram(ROIvals(iroi).efnorm_mags)
        
        xlim([-lim_x,lim_x])
%         ylim([0,lim_y])
        if ROIvals(iroi).avg_efnorm_mag < 0
            xline(ROIvals(iroi).avg_efnorm_mag, 'r')
        else
            xline(ROIvals(iroi).avg_efnorm_mag, 'g')
        end
        title(strcat(roi_names{iroi}, ' = ', num2str(ROIvals(iroi).avg_efnorm_mag)))
    end
end

% Save important variables. Use 
save(strcat('orthogonalization_v2/', subj,'-ROIvals_orth-', protocol, '.mat'), "ROIvals")
savefig(fig_hist_lh, "orthogonalization_v2/histograms_LeftHem.fig")
saveas(fig_hist_lh, "orthogonalization_v2/histograms_LeftHem.png")
savefig(fig_hist_rh, "orthogonalization_v2/histograms_RightHem.fig") 
saveas(fig_hist_rh, "orthogonalization_v2/histograms_RightHem.png")
close all


%% Save matrices for AAL2red
writematrix([ROIvals.avg_efnorm_mag], strcat('orthogonalization_v2/', subj,'-efnorm_mag-', protocol, '-AAL2.txt'));
writecell(roi_names, strcat('orthogonalization_v2/', subj,'-efnorm_mag-', protocol, '-AAL2_labels.txt'))

