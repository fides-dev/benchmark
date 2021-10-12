function run_Benchmark(model, optimizer)

output_folder = pwd;
copyfile(fullfile(pwd, 'arFit.m'), ...
         fullfile(pwd,'d2d','arFramework3','arFit.m'))
cd(fullfile(pwd,'d2d','arFramework3'))
arInit
d2d_folder = pwd;

cd(fullfile(d2d_folder, 'Examples', model))
    
if strcmp(model, 'Boehm_JProteomeRes2014')
    Setup_FullModel_Boehm2014
elseif strcmp(model,'Zheng_PNAS2012')
    Setup_Zheng
elseif strcmp(model,'Beer_MolBiosyst2014')
    Setup_IndiBac
elseif strcmp(model,'Hass_npjSysBio2017')
    cd('PanRTK_finalModel')
    Setup
elseif strcmp(model,'Lucarelli_TGFb_2017')
    cd('TGFb_ComplexModel_WithGenes_Reduced')
    Setup
else
    Setup
end
if strcmp(model, 'Fujita_SciSignal2010')
    ar.config.fiterrors = 0;
end

ar.config.atol = 1e-8;
ar.config.rtol = 1e-8;
ar.config.useFitErrorCorrection = 0;
ar.config.optim.TolX = 1e-6;
ar.config.optim.TolFun = 0;
ar.config.optim.PreconBandWidth = Inf;
ar.config.optim.MaxIter = 1e5;
ar.config.optim.MaxFunEvals = 1e5;

if strcmp(optimizer, 'lsqnonlin')
    % lsqnonlin
    ar.config.optimizer = 1;
    ar.config.optim.SubproblemAlgorithm = 'factorization';
else
    % fmincon
    ar.config.optimizer = 2;
end

mat_savefile = fullfile(output_folder, strcat(model, '_', optimizer, '.mat'));
if isfile(mat_savefile)
    load(mat_savefile)
else
    arFitLHS(1000, 0)
    save(mat_savefile, 'ar')
end
for field = {'iter', 'chi2s', 'ps', 'ps_start'}
    dlmwrite(fullfile(output_folder, strcat(model, '_', optimizer, '_', field{1},'.csv')), ar.(field{1}), 'delimiter', ',', 'precision', 12)
end
fid = fopen(fullfile(output_folder, strcat(model, '_', optimizer, '_pLabel.csv')), 'w');
for row = 1:length(ar.pLabel)
    fprintf(fid, '%s,', ar.pLabel{row});
end
fclose(fid);

cd(d2d_folder)

cd(output_folder);
end